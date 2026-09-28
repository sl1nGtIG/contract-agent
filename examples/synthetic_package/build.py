"""Генерация синтетического пакета: PDF-документы, эталон для evaluate и кассета для офлайн-демо.

    python examples/synthetic_package/build.py

Создаёт:
  examples/synthetic_package/docs/**         — PDF (текстовые и сканы без текстового слоя) + неподдерживаемый .xlsx
  examples/synthetic_package/gold.json       — эталон: поля документов, изменения, цепочки (генерируется ДО любого прогона)
  examples/demo/cassette/extractions.json    — «ответы модели» для ReplayLLM, с намеренно внесёнными ошибками
  examples/demo/cassette/ocr.json            — расшифровка сканов для ReplayLLM
"""
from __future__ import annotations

import copy
import json
import shutil
from html import escape
from pathlib import Path

import pymupdf

from spec import AS_OF, CHAINS, DOCS, INVALIDATED, OUR_PARTY, UNSUPPORTED

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CASSETTE = HERE.parent / "demo" / "cassette"
PAGE = pymupdf.Rect(56, 56, 540, 800)


def _html(paragraphs: list[str]) -> str:
    return "".join(f"<p style='margin:0 0 6px 0'>{escape(p)}</p>" for p in paragraphs)


def write_text_pdf(path: Path, paragraphs: list[str]) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_htmlbox(PAGE, _html(paragraphs))
    doc.save(path)


def write_scan_pdf(path: Path, paragraphs: list[str]) -> None:
    """Страница рендерится в картинку и вставляется изображением: текстового слоя нет, нужен OCR."""
    src = pymupdf.open()
    src.new_page().insert_htmlbox(PAGE, _html(paragraphs))
    png = src[0].get_pixmap(dpi=110, colorspace=pymupdf.csGRAY).tobytes("png")
    doc = pymupdf.open()
    doc.new_page().insert_image(doc[0].rect if len(doc) else pymupdf.Rect(0, 0, 595, 842), stream=png)
    doc.save(path)


def ev(pair, conf=0.95):
    if not pair:
        return {"value": None, "quote": None, "confidence": 0.0}
    value, quote = pair
    return {"value": value, "quote": quote, "confidence": conf}


def to_extraction(e: dict) -> dict:
    """Спецификация документа → ответ модели в формате DocumentExtraction."""
    return {
        "doc_type": e["doc_type"], "title": e["title"], "number": ev(e.get("number")), "date": ev(e.get("date")),
        "city": e.get("city"), "parent_number": ev(e.get("parent_number")), "parent_date": ev(e.get("parent_date")),
        "parent_title": e.get("parent_title"), "counterparty": ev(e.get("counterparty")),
        "counterparty_legal_form": e.get("counterparty_legal_form", "unknown"), "signatory_branch": None,
        "subject_category": e["subject_category"], "subject_summary": e["subject_summary"],
        "effective_from": ev(e.get("effective_from")), "valid_until": ev(e.get("valid_until")),
        "key_terms": [{**k, "confidence": 0.9} for k in e.get("key_terms", [])],
        "changes": [{**c, "confidence": 0.9} for c in e.get("changes", [])],
        "invalidates": e.get("invalidates", []), "related_contracts": e.get("related_contracts", []),
        "anonymized_fields": [], "extraction_notes": [],
    }


def gold_record(d: dict) -> dict:
    e = d["extraction"]
    value = lambda key: (e.get(key) or (None, None))[0]  # noqa: E731
    rec = {
        "doc_type": e["doc_type"], "number": value("number"), "date": value("date"),
        "parent_number": value("parent_number"), "parent_date": value("parent_date"),
        "effective_from": value("effective_from"), "valid_until": value("valid_until"),
        "counterparty_legal_form": e.get("counterparty_legal_form", "unknown"),
        "invalidates": [r["number"] for r in e.get("invalidates", [])],
        "changes": [{"target_document": c["target_document"], "target_clause": c["target_clause"], "action": c["action"]}
                    for c in e.get("changes", [])],
    }
    rec.update(d.get("gold", {}))
    return rec


# Намеренные ошибки «модели» в кассете: каждая должна быть поймана встроенными проверками
def inject_errors(extractions: dict[str, dict]) -> None:
    by_number = {x["number"]["value"]: x for x in extractions.values()}
    # 1. дата окончания у договора с автопролонгацией
    by_number["АР-2021/015"]["valid_until"] = {"value": "31.12.2023", "quote": "действует до 31.12.2023", "confidence": 0.9}
    # 2. пересказ вместо цитаты
    by_number["П-118/22"]["key_terms"].append(
        {"term": "reporting", "summary": "Поставщик ежеквартально направляет акт сверки.", "clause_ref": "п. 7.1",
         "quote": "Поставщик ежеквартально направляет Покупателю акт сверки взаиморасчётов", "confidence": 0.85})
    # 3. настоящая цитата, но чужое название контрагента
    by_number["П-118/22-1"]["counterparty"] = {"value": "ООО «Ромашка»", "quote": "ООО «ТехноСнаб» (Поставщик)", "confidence": 0.9}
    # 4. настоящая цитата, но чужая сумма
    by_number["У-77/5"]["changes"].append(
        {"target_document": "У-77", "target_clause": "п. 3.2", "action": "restate", "affected_terms": ["remuneration"],
         "new_value_summary": "Стоимость услуг 150 000 руб. в месяц.", "effective_from": "01.02.2024",
         "quote": "Стоимость услуг составляет 135 000 (сто тридцать пять тысяч) рублей в месяц", "confidence": 0.9})
    # 5. изменение документа, которого в тексте нет
    by_number["П-118/22-4"]["changes"].append(
        {"target_document": "П-118/22-9", "target_clause": "п. 4", "action": "restate", "affected_terms": ["obligations"],
         "new_value_summary": "Новый порядок приёмки.", "effective_from": None,
         "quote": "Изложить пункт 1 Дополнительного соглашения № П-118/22-1 в следующей редакции", "confidence": 0.85})


def main() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    extractions, ocr, gold = {}, {}, {}
    for d in DOCS:
        path = DOCS_DIR / d["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if d["kind"] == "scan":
            write_scan_pdf(path, d["paragraphs"])
            # 6. рукописная (неуверенно прочитанная) дата на скане договора аренды
            uncertain = ["«01» сентября 2026 г."] if "аренда" in d["path"] else []
            ocr[d["path"]] = {"1": {"text": "\n".join(d["paragraphs"]), "legibility": "medium" if uncertain else "high",
                                    "uncertain_fragments": uncertain}}
        else:
            write_text_pdf(path, d["paragraphs"])
        extractions[d["path"]] = to_extraction(d["extraction"])
        gold[d["path"]] = gold_record(d)
    for rel in UNSUPPORTED:
        (DOCS_DIR / rel).write_bytes(b"PK\x03\x04 synthetic placeholder, not a real workbook")

    cassette = copy.deepcopy(extractions)
    inject_errors(cassette)
    CASSETTE.mkdir(parents=True, exist_ok=True)
    (CASSETTE / "extractions.json").write_text(json.dumps(cassette, ensure_ascii=False, indent=1), encoding="utf-8")
    (CASSETTE / "ocr.json").write_text(json.dumps(ocr, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "gold.json").write_text(json.dumps(
        {"our_party": OUR_PARTY, "as_of": AS_OF, "documents": gold, "chains": CHAINS, "invalidated": INVALIDATED},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"docs: {len(DOCS)} PDF + {len(UNSUPPORTED)} unsupported -> {DOCS_DIR}")


if __name__ == "__main__":
    main()
