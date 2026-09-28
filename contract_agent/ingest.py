"""Загрузка документов: инвентаризация, извлечение текстового слоя, OCR сканов.

Работает без LLM, кроме OCR: страницы без текстового слоя рендерятся в PNG и
распознаются vision-моделью (см. ``llm.LLMClient.ocr_pages``). Результаты кешируются
по хешу файла, повторный запуск не тратит токены.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from xml.etree import ElementTree
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf

if TYPE_CHECKING:
    from .config import Settings
    from .llm import LLMClient

PDF_EXT = {".pdf"}
TEXT_EXT = {".txt", ".md"}
DOCX_EXT = {".docx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = PDF_EXT | TEXT_EXT | DOCX_EXT | IMAGE_EXT
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
UNREADABLE_MARK = "[неразборчиво]"

# Форматы входных данных и доступные методы чтения — по ним агент выбирает метод в parse_document
FORMATS = {
    "pdf_text": "PDF с текстовым слоем → method=auto (или text_layer)",
    "pdf_scan": "PDF-скан без текстового слоя → method=ocr",
    "pdf_mixed": "PDF: часть страниц — сканы → method=auto (текст + OCR сканов)",
    "docx": "Word → method=auto (текст из XML документа)",
    "text": "Текстовый файл → method=auto",
    "image": "Изображение (скан страницы) → method=ocr",
    "unsupported": "Формат не поддерживается — документ помечается нераспознанным",
    "corrupted": "Файл повреждён или зашифрован — документ помечается нераспознанным",
}
METHODS = ("auto", "text_layer", "ocr")
# группы форматов — единственное место, где они перечислены (используются в ingest, tools, demo)
SCAN_FORMATS = frozenset({"pdf_scan", "pdf_mixed", "image"})      # есть страницы без текстового слоя
OCR_ONLY_FORMATS = frozenset({"pdf_scan", "image"})                # читаются только через OCR
PLAIN_TEXT_FORMATS = frozenset({"text", "docx"})                   # OCR не применяется
UNREADABLE_FORMATS = frozenset({"unsupported", "corrupted"})
PDF_OR_IMAGE_FORMATS = frozenset({"pdf_text"}) | SCAN_FORMATS      # можно перечитать через OCR


class ParseError(Exception):
    """Документ нельзя прочитать выбранным методом (формат, повреждение, нет OCR)."""

pymupdf.TOOLS.mupdf_display_errors(False)  # битые content stream'ы в PDF не должны засорять вывод


@dataclass
class DocumentInfo:
    doc_id: str
    path: str          # относительно input_dir
    folder: str
    file: str
    size_bytes: int
    pages: int
    text_pages: int    # страниц с нормальным текстовым слоем
    needs_ocr: bool
    format: str = "pdf_text"  # см. FORMATS


@dataclass
class PageText:
    page: int
    text: str
    source: str                 # text_layer | ocr | none
    legibility: str = "high"    # high | medium | low
    uncertain_fragments: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    doc_id: str
    path: str
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(f"=== Страница {p.page} ===\n{p.text}" for p in self.pages)

    @property
    def text_source(self) -> str:
        sources = {p.source for p in self.pages if p.text.strip()}
        if not sources:
            return "none"
        return sources.pop() if len(sources) == 1 else "mixed"

    @property
    def quality(self) -> dict:
        body = "".join(p.text for p in self.pages)
        letters = [c for c in body if c.isalpha()]
        cyr = sum(1 for c in letters if "а" <= c.lower() <= "я" or c.lower() == "ё")
        unreadable = body.count(UNREADABLE_MARK)
        low_pages = [p.page for p in self.pages if p.legibility == "low"]
        empty_pages = [p.page for p in self.pages if len(p.text.strip()) < 20]
        return {
            "chars": len(body),
            "cyrillic_ratio": round(cyr / len(letters), 3) if letters else 0.0,
            "unreadable_marks": unreadable,
            "low_legibility_pages": low_pages,
            "empty_pages": empty_pages,
            "source": self.text_source,
        }

    def is_recognized(self) -> tuple[bool, str]:
        """Минимальный порог «документ распознан». Возвращает (ok, причина)."""
        q = self.quality
        if q["chars"] < 200:
            return False, f"слишком мало текста ({q['chars']} символов)"
        if q["cyrillic_ratio"] < 0.5:
            return False, f"текст не похож на русский (доля кириллицы {q['cyrillic_ratio']})"
        if len(q["low_legibility_pages"]) > len(self.pages) / 2:
            return False, "больше половины страниц распознаны с низкой разборчивостью"
        return True, ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "ParsedDocument":
        return cls(
            doc_id=data["doc_id"],
            path=data["path"],
            pages=[PageText(**p) for p in data["pages"]],
            warnings=data.get("warnings", []),
        )


def make_doc_id(rel_path: str) -> str:
    """Стабильный короткий id: не зависит от порядка файлов и удобен для вызова tools."""
    return "doc_" + hashlib.sha1(rel_path.replace("\\", "/").encode("utf-8")).hexdigest()[:8]


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx_text(path: Path) -> str:
    """Текст DOCX без сторонних библиотек: абзацы и ячейки таблиц из word/document.xml."""
    with zipfile.ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("word/document.xml"))
    lines = []
    for para in root.iter(f"{_W_NS}p"):
        text = "".join(t.text or "" for t in para.iter(f"{_W_NS}t"))
        if text.strip():
            lines.append(text)
    return clean_text("\n".join(lines))


def is_scan_page(page: "pymupdf.Page", min_chars: int) -> bool:
    """Скан = почти нет текстового слоя, но есть изображение. Пустая страница без картинок сканом не считается."""
    return len(page.get_text().strip()) < min_chars and len(page.get_images()) > 0


def detect_format(path: Path, min_chars_per_page: int) -> tuple[str, int, int]:
    """Определяет формат файла: (format, pages, text_pages)."""
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return "text", 1, 1
    if ext in DOCX_EXT:
        try:
            with zipfile.ZipFile(path) as z:
                z.getinfo("word/document.xml")
            return "docx", 1, 1
        except (zipfile.BadZipFile, KeyError):
            return "corrupted", 0, 0
    if ext in IMAGE_EXT:
        return "image", 1, 0
    if ext in PDF_EXT:
        try:
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass:
                    return "corrupted", 0, 0
                pages = len(pdf)
                text_pages = sum(not is_scan_page(p, min_chars_per_page) for p in pdf)
        except Exception:
            return "corrupted", 0, 0
        if pages == 0:
            return "corrupted", 0, 0
        kind = "pdf_text" if text_pages == pages else "pdf_scan" if text_pages == 0 else "pdf_mixed"
        return kind, pages, text_pages
    return "unsupported", 0, 0


def scan_inventory(input_dir: Path, min_chars_per_page: int) -> list[DocumentInfo]:
    """Все файлы пакета, включая неподдерживаемые: они не пропускаются молча, а попадают в инвентарь с format=unsupported."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Папка с документами не найдена: {input_dir}")
    docs: list[DocumentInfo] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.name.lower() in IGNORED_NAMES or path.name.startswith("~$"):
            continue
        rel = path.relative_to(input_dir).as_posix()
        folder = path.parent.relative_to(input_dir).as_posix()
        fmt, pages, text_pages = detect_format(path, min_chars_per_page)
        docs.append(
            DocumentInfo(
                doc_id=make_doc_id(rel),
                path=rel,
                folder=folder,
                file=path.name,
                size_bytes=path.stat().st_size,
                pages=pages,
                text_pages=text_pages,
                needs_ocr=fmt in SCAN_FORMATS,
                format=fmt,
            )
        )
    return docs


_WS = re.compile(r"[ \t ]+")


def clean_text(text: str) -> str:
    """Схлопываем пробелы и «лесенки» из одиночных слов, которые даёт выравнивание по ширине."""
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln:
            if out and out[-1] != "":
                out.append("")
            continue
        # короткие строки-обрывки (одно слово на строке) склеиваем с предыдущей
        if out and out[-1] and len(ln) < 25 and not re.match(r"^(\d+(\.\d+)*\.?|[-•])\s", ln) and not out[-1].endswith((".", ":", ";")):
            out[-1] = out[-1] + " " + ln
        else:
            out.append(ln)
    return "\n".join(out).strip()


class DocumentLoader:
    """Инкапсулирует доступ к файлам, кешу текста и OCR."""

    def __init__(self, settings: "Settings", llm: "LLMClient | None" = None):
        self.settings = settings
        self.llm = llm
        self.text_cache = settings.cache_dir / "text"
        self.text_cache.mkdir(parents=True, exist_ok=True)

    def inventory(self) -> list[DocumentInfo]:
        return scan_inventory(self.settings.input_dir, self.settings.min_chars_per_page)

    def resolve(self, info: DocumentInfo) -> Path:
        return self.settings.input_dir / info.path

    def parse(self, info: DocumentInfo, method: str = "auto") -> ParsedDocument:
        """method: auto — текстовый слой, а сканы через OCR; text_layer — только текстовый слой; ocr — всё через OCR."""
        if method not in METHODS:
            raise ParseError(f"неизвестный метод {method!r}, допустимо: {METHODS}")
        fmt = info.format
        if fmt in UNREADABLE_FORMATS:
            raise ParseError(FORMATS[fmt])
        if fmt in PLAIN_TEXT_FORMATS and method == "ocr":
            raise ParseError(f"для формата {fmt} OCR не применяется — используйте method=auto")
        if fmt == "image" and method == "text_layer":
            raise ParseError("у изображения нет текстового слоя — используйте method=ocr")
        path = self.resolve(info)
        suffix = {"auto": "", "ocr": "_ocr", "text_layer": "_text"}[method]
        cache_file = self.text_cache / f"{info.doc_id}_{file_sha1(path)[:12]}{suffix}.json"
        if cache_file.exists():  # уже распознанный текст (в т.ч. OCR) доступен и без LLM
            return ParsedDocument.from_json(json.loads(cache_file.read_text(encoding="utf-8")))
        if method != "text_layer" and fmt in SCAN_FORMATS and self.llm is None:
            raise ParseError("для распознавания сканов нужен LLM (OCR недоступен без ключа API)")

        if fmt == "text":
            parsed = ParsedDocument(info.doc_id, info.path, [PageText(1, clean_text(path.read_text(encoding="utf-8", errors="replace")), "text_layer")])
        elif fmt == "docx":
            parsed = ParsedDocument(info.doc_id, info.path, [PageText(1, read_docx_text(path), "text_layer")])
        else:  # pdf_* и image: PyMuPDF открывает изображение как одностраничный документ
            parsed = self._parse_pdf(info, path, method)

        cache_file.write_text(json.dumps(parsed.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
        return parsed

    def _parse_pdf(self, info: DocumentInfo, path: Path, method: str) -> ParsedDocument:
        warnings: list[str] = []
        pages: list[PageText] = []
        scan_pages: list[int] = []
        try:
            pdf = pymupdf.open(path)
        except Exception as exc:
            raise ParseError(f"файл не открывается: {exc}") from exc
        with pdf:
            for i, page in enumerate(pdf):
                raw = page.get_text()
                is_scan = info.format == "image" or is_scan_page(page, self.settings.min_chars_per_page)
                if method == "ocr" or (method == "auto" and is_scan):
                    scan_pages.append(i)
                    pages.append(PageText(i + 1, "", "none", "low"))
                else:
                    pages.append(PageText(i + 1, clean_text(raw), "text_layer" if raw.strip() else "none"))

            if scan_pages:
                if self.llm is None:
                    warnings.append(f"Страницы {[p + 1 for p in scan_pages]} без текстового слоя, OCR недоступен (нет LLM)")
                else:
                    images = [
                        (i + 1, pdf[i].get_pixmap(dpi=self.settings.ocr_dpi).tobytes("png")) for i in scan_pages
                    ]
                    step = self.settings.ocr_pages_per_request
                    for start in range(0, len(images), step):
                        batch = images[start : start + step]
                        try:
                            result = self.llm.ocr_pages(batch, doc_path=info.path)
                        except Exception as exc:  # OCR упал — страницы остаются пустыми, но с явной пометкой
                            warnings.append(f"OCR страниц {[n for n, _ in batch]} не удался: {exc}")
                            continue
                        by_num = {p.page: p for p in result.pages}
                        for num, _ in batch:
                            ocr = by_num.get(num)
                            if ocr is None:
                                warnings.append(f"OCR не вернул страницу {num}")
                                continue
                            pages[num - 1] = PageText(
                                num, clean_text(ocr.text), "ocr", ocr.legibility, ocr.uncertain_fragments
                            )
        total = sum(len(p.text) for p in pages)
        if total > self.settings.max_doc_chars:
            warnings.append(
                f"Документ очень большой ({total} символов): в LLM будут переданы первые {self.settings.max_doc_chars}"
            )
        return ParsedDocument(info.doc_id, info.path, pages, warnings)
