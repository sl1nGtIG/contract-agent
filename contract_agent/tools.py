"""Инструменты агента.

Каждый инструмент:
* описан JSON-схемой (её видит модель) и docstring-описанием «когда использовать»;
* принимает контекст ``ToolContext`` + аргументы, возвращает dict;
* обёрнут в ``ToolRegistry.execute``: исключения превращаются в ``{"status": "error"}``
  с подсказкой, что делать дальше — агент видит ошибку как данные, а не падает.

Статусы результата: ok — есть данные; empty — выполнено, но данных нет (это не ошибка);
error — не выполнено, в ``error`` причина, в ``hint`` — рекомендуемое действие.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Annotated, Any, Callable, Literal

from pydantic import ConfigDict, Field, ValidationError, validate_call

from .analysis import (
    CONTRACT_DIMENSIONS,
    DOCUMENT_DIMENSIONS,
    TERM_RU,
    cluster_items,
    compare_contracts,
    consolidate,
    contract_features,
    document_features,
    link_chains,
)
from .config import Settings
from .extraction import extract_card
from .ingest import FORMATS, PDF_OR_IMAGE_FORMATS, PLAIN_TEXT_FORMATS, DocumentInfo, DocumentLoader, ParsedDocument, ParseError
from .knowledge_base import KnowledgeBase
from .providers import LLMBackend  # noqa: F401  (интерфейс для документации и IDE)
from .reflection import check_plan
from .report import FULL_REPORT_NAME, render_report
from .schemas import DocumentCard, Severity, TermKey


class ToolError(Exception):
    """Ожидаемая ошибка инструмента с подсказкой для агента."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


@dataclass
class ToolContext:
    settings: Settings
    kb: KnowledgeBase
    loader: DocumentLoader
    llm: Any  # LLMBackend | None; Any — потому что pydantic не строит isinstance-проверку для Protocol
    parsed_cache: dict[str, ParsedDocument] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]
    read_only: bool = False

    def __post_init__(self):
        # аргументы от модели приходят как JSON: проверяем и приводим типы по аннотациям обработчика
        # (строка вместо списка, "5" вместо 5, лишние поля) — ошибка возвращается агенту, а не роняет инструмент
        self.handler = validate_call(self.handler, config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))

    def to_api(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def api_definitions(self) -> list[dict]:
        return [t.to_api() for t in self.tools.values()]

    def execute(self, ctx: ToolContext, name: str, args: dict) -> tuple[dict, bool]:
        """Возвращает (результат, is_error). Никогда не бросает исключений наружу."""
        tool = self.tools.get(name)
        if tool is None:
            return {"status": "error", "error": f"неизвестный инструмент {name}", "hint": f"доступны: {sorted(self.tools)}"}, True
        try:
            result = tool.handler(ctx, **(args or {}))
        except ToolError as exc:
            return {"status": "error", "error": str(exc), "hint": exc.hint}, True
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors(include_url=False))
            return {"status": "error", "error": f"неверные аргументы: {problems}", "hint": "сверьтесь со схемой инструмента"}, True
        except TypeError as exc:
            return {"status": "error", "error": f"неверные аргументы: {exc}", "hint": "сверьтесь со схемой инструмента"}, True
        except Exception as exc:  # непредвиденная ошибка — агент должен узнать о ней и решить, что делать
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "hint": "повторите один раз или пропустите шаг"}, True
        result.setdefault("status", "ok")
        return result, result["status"] == "error"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _info(ctx: ToolContext, doc_id: str) -> DocumentInfo:
    if not ctx.kb.documents:
        _refresh_inventory(ctx)
    info = ctx.kb.documents.get(doc_id)
    if info is None:
        card = ctx.kb.find_card(doc_id)
        if card:
            return ctx.kb.documents[card.doc_id]
        raise ToolError(f"документ {doc_id} не найден", "вызовите list_documents и используйте doc_id из списка")
    return info


def _refresh_inventory(ctx: ToolContext) -> list[DocumentInfo]:
    docs = ctx.loader.inventory()
    with ctx.lock:
        ctx.kb.documents = {d.doc_id: d for d in docs}
        ctx.kb.save()
    return docs


def _parsed(ctx: ToolContext, info: DocumentInfo, method: str = "auto") -> ParsedDocument:
    key = f"{info.doc_id}:{method}"
    if key not in ctx.parsed_cache:
        ctx.parsed_cache[key] = ctx.loader.parse(info, method=method)
    return ctx.parsed_cache[key]


def _best_parsed(ctx: ToolContext, info: DocumentInfo) -> ParsedDocument:
    """Текст тем методом, которым документ был успешно прочитан последним parse_document."""
    status = ctx.kb.parse_status.get(info.doc_id, {})
    method = status.get("method") or ("ocr" if status.get("force_ocr") else "auto")
    return _parsed(ctx, info, method=method)


def _short(text: str | None, n: int = 220) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def card_brief(card: DocumentCard) -> dict:
    return {
        "doc_id": card.doc_id,
        "file": card.file,
        "status": card.status,
        "doc_type": card.doc_type.value,
        "title": card.title,
        "number": card.number.value,
        "date": card.date.value,
        "parent_number": card.parent_number.value,
        "parent_date": card.parent_date.value,
        "effective_from": card.effective_from.value,
        "subject_category": card.subject_category.value,
        "subject": _short(card.subject_summary, 300),
        "key_terms": [f"{k.term.value}: {_short(k.summary, 160)}" for k in card.key_terms],
        "changes": [f"{c.action.value} {c.target_clause}"
                    + (f" [в {c.target_document}]" if c.target_document else "") + f": {_short(c.new_value_summary, 200)}"
                    for c in card.changes],
        "invalidates": [f"{r.number} от {r.date}" for r in card.invalidates],
        "related_contracts": [r.number for r in card.related_contracts],
        "rejected": [f"{i.field}: {i.reason} (кандидат: {_short(str(i.raw_value), 80)})" for i in card.rejected],
        "warnings": [f"{i.field}: {i.reason}" for i in card.issues if i.severity == Severity.WARNING],
        "notes": card.extraction_notes,
    }


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------
def list_documents(ctx: ToolContext) -> dict:
    docs = _refresh_inventory(ctx)
    if not docs:
        return {"status": "empty", "message": f"в {ctx.settings.input_dir} нет файлов"}
    formats = sorted({d.format for d in docs})
    return {
        "total": len(docs),
        "by_format": {f: sum(d.format == f for d in docs) for f in formats},
        "how_to_read": {f: FORMATS[f] for f in formats},
        "documents": [
            {"doc_id": d.doc_id, "path": d.path, "format": d.format, "pages": d.pages,
             "text_layer": f"{d.text_pages}/{d.pages} стр.", "already_extracted": d.doc_id in ctx.kb.cards}
            for d in docs
        ],
    }


def parse_document(ctx: ToolContext, doc_id: str, method: Literal["auto", "text_layer", "ocr"] = "auto") -> dict:
    info = _info(ctx, doc_id)
    try:
        parsed = _parsed(ctx, info, method)
    except ParseError as exc:
        # неподдерживаемый/повреждённый файл или неприменимый метод: фиксируем как нераспознанный документ
        ctx.kb.set_parse_status(info.doc_id, {"recognized": False, "reason": str(exc), "quality": {"source": info.format},
                                              "warnings": [], "method": method})
        scan_like = info.format in PDF_OR_IMAGE_FORMATS
        hint = ("выберите другой метод: " + FORMATS[info.format]) if scan_like and method != "ocr" and ctx.llm else \
            "документ помечен нераспознанным — продолжайте с остальными и отразите это в отчёте"
        raise ToolError(f"документ не прочитан: {exc}", hint) from exc
    ok, reason = parsed.is_recognized()
    quality = parsed.quality
    ctx.kb.set_parse_status(info.doc_id, {"recognized": ok, "reason": reason, "quality": quality,
                                          "warnings": parsed.warnings, "method": method})
    result = {
        "doc_id": info.doc_id,
        "file": info.file,
        "pages": len(parsed.pages),
        "quality": quality,
        "recognized": ok,
        "warnings": parsed.warnings,
        "preview": _short(parsed.text, 400),
    }
    if not ok:
        can_ocr = method != "ocr" and ctx.llm is not None and info.format not in PLAIN_TEXT_FORMATS
        result.update(status="error", error=f"документ не распознан: {reason}",
                      hint="повторите с method='ocr'" if can_ocr else "документ нужно отметить как нераспознанный и продолжить")
    return result


def extract_document_card(ctx: ToolContext, doc_id: str, force: bool = False) -> dict:
    info = _info(ctx, doc_id)
    if info.doc_id in ctx.kb.cards and not force:
        return {"cached": True, "card": card_brief(ctx.kb.cards[info.doc_id])}
    if ctx.llm is None:
        raise ToolError("извлечение требует LLM", "задайте ANTHROPIC_API_KEY")
    status = ctx.kb.parse_status.get(info.doc_id)
    if status is None:
        parse_res = parse_document(ctx, info.doc_id)
        status = ctx.kb.parse_status[info.doc_id]
        if parse_res.get("status") == "error":
            raise ToolError(parse_res["error"], parse_res.get("hint"))
    if not status.get("recognized"):
        raise ToolError(f"документ не распознан: {status.get('reason')}", "сначала parse_document с method='ocr'")
    parsed = _best_parsed(ctx, info)
    card = extract_card(info, parsed, ctx.llm, ctx.settings)
    ctx.kb.put_card(card)
    result = {"cached": False, "card": card_brief(card)}
    if card.status == "unrecognized":
        result.update(status="error", error="из документа не удалось извлечь ни номер, ни дату, ни условия",
                      hint="проверьте текст через search_documents или отметьте документ как нераспознанный")
    return result


def get_document_card(ctx: ToolContext, doc: str) -> dict:
    card = ctx.kb.find_card(doc)
    if card is None:
        return {"status": "empty", "message": f"карточка {doc} не найдена", "known": [c.number.value or c.doc_id for c in ctx.kb.cards.values()]}
    data = card.model_dump(mode="json", exclude={"issues"})
    data["contract_id"] = ctx.kb.contract_of(card.doc_id)
    data["rejected"] = [i.model_dump(mode="json") for i in card.rejected]
    data["warnings"] = [i.reason for i in card.issues if i.severity == Severity.WARNING]
    return data


def search_documents(ctx: ToolContext, query: str, doc_ids: list[str] | None = None,
                     max_hits: Annotated[int, Field(ge=1, le=20)] = 8) -> dict:
    words = [w.lower().replace("ё", "е") for w in re.findall(r"[\wА-Яа-яЁё.\-]+", query) if len(w) > 2]
    if not words:
        raise ToolError("пустой запрос", "передайте ключевые слова, номер документа или пункта")
    # грубый стемминг для русского: отбрасываем окончание (арендная/арендную → арендн, плата/плату → плат)
    stems = [w[: max(4, len(w) - 2)] if w.isalpha() and len(w) > 4 else w for w in words]
    unknown = [d for d in (doc_ids or []) if d not in ctx.kb.documents]
    if unknown:
        raise ToolError(f"неизвестные doc_id: {unknown}", "используйте doc_id из list_documents")
    # ищем только по уже прочитанным документам: поиск не должен запускать OCR или падать на нераспознанных файлах
    targets = [ctx.kb.documents[d] for d in (doc_ids or ctx.kb.documents)
               if ctx.kb.parse_status.get(d, {}).get("recognized")]
    if not targets:
        return {"status": "empty", "message": "нет прочитанных документов для поиска — сначала parse_document"}
    hits = []
    for info in targets:
        parsed = _best_parsed(ctx, info)
        for page in parsed.pages:
            for para in re.split(r"\n(?=\d+(?:\.\d+)*\.?\s)|\n\n", page.text):
                low = para.lower().replace("ё", "е")
                score = sum(1 for s in stems if s in low)
                if score:
                    hits.append((score, info, page.page, para))
    if not hits:
        return {"status": "empty", "message": f"по запросу «{query}» ничего не найдено"}
    hits.sort(key=lambda h: -h[0])
    return {
        "hits": [
            {"doc_id": info.doc_id, "file": info.file, "number": (ctx.kb.cards.get(info.doc_id).number.value if info.doc_id in ctx.kb.cards else None),
             "page": page, "score": score, "text": _short(para, 700)}
            for score, info, page, para in hits[:max_hits]
        ]
    }


def link_contract_chains(ctx: ToolContext) -> dict:
    if not ctx.kb.cards:
        raise ToolError("нет карточек документов", "сначала extract_document_card для документов")
    contracts, issues = link_chains(ctx.kb.cards.values())
    with ctx.lock:
        ctx.kb.contracts, ctx.kb.link_issues = contracts, issues
        ctx.kb.consolidated, ctx.kb.clusters = {}, {}
        ctx.kb.touch()
        ctx.kb.save()
    unrecognized = [c.file for c in ctx.kb.cards.values() if c.status == "unrecognized"]
    return {
        "contracts": [
            {
                "contract_id": c.contract_id,
                "title": c.title,
                "date": c.date.value,
                "main_document_in_package": c.main_doc_id is not None,
                "status": c.status,
                "timeline": [f"{t.date or '—'} {t.number or t.doc_id} ({t.doc_type.value})" + ("" if t.valid else f" — НЕДЕЙСТВИТЕЛЕН: {t.invalid_reason}")
                             for t in c.timeline],
                "related_contracts": c.related_contract_ids,
                "issues": [f"[{i.severity.value}] {i.reason}" for i in c.issues],
            }
            for c in contracts.values()
        ],
        "global_issues": [i.reason for i in issues],
        "not_linked_unrecognized": unrecognized,
    }


def consolidate_contract(ctx: ToolContext, contract_id: str) -> dict:
    if not ctx.kb.contracts:
        raise ToolError("цепочки договоров не построены", "сначала link_contract_chains")
    contract = ctx.kb.find_contract(contract_id)
    if contract is None:
        raise ToolError(f"договор {contract_id} не найден", f"известные договоры: {sorted(ctx.kb.contracts)}")
    cons = consolidate(contract, ctx.kb.cards, today=ctx.settings.as_of)
    with ctx.lock:
        ctx.kb.consolidated[contract.contract_id] = cons
        ctx.kb.touch()
        ctx.kb.save()
    terms = {
        TERM_RU[TermKey(k)]: {
            "active": [f"{_short(v.summary, 220)} [{v.source_number}, {v.clause_ref or '—'}]" for v in st.active],
            "versions": len(st.history),
            "changed_by": sorted({v.source_number for v in st.history if v.action != "base" and v.source_number}),
        }
        for k, st in cons.terms.items()
    }
    result = {"contract_id": contract.contract_id, "contract_status": cons.status, "as_of": cons.as_of, "terms": terms,
              "applied_docs": len(cons.applied_docs), "skipped_docs": cons.skipped_docs, "notes": cons.notes}
    if not cons.terms:
        result["status"] = "empty"
        result["message"] = "ни одно условие не подтверждено проверками — свертка пустая"
    return result


def cluster_contracts(ctx: ToolContext, level: Literal["contract", "document"] = "contract",
                      dimensions: list[str] | None = None) -> dict:
    if level not in ("contract", "document"):
        raise ToolError("level должен быть contract или document")
    allowed = CONTRACT_DIMENSIONS if level == "contract" else DOCUMENT_DIMENSIONS
    dims = dimensions or ["subject_category"]
    bad = [d for d in dims if d not in allowed]
    if bad:
        raise ToolError(f"неизвестные измерения {bad}", f"допустимые для {level}: {list(allowed)}")
    if level == "contract":
        if not ctx.kb.contracts:
            raise ToolError("цепочки договоров не построены", "сначала link_contract_chains")
        features = {cid: contract_features(c) for cid, c in ctx.kb.contracts.items()}
    else:
        if not ctx.kb.cards:
            raise ToolError("нет карточек документов", "сначала extract_document_card")
        features = {d: document_features(c) for d, c in ctx.kb.cards.items() if c.status != "unrecognized"}
    clusters = cluster_items(features, dims, level)
    with ctx.lock:
        ctx.kb.clusters[f"{level}:{','.join(dims)}"] = clusters
        ctx.kb.touch()
        ctx.kb.save()
    return {
        "level": level,
        "dimensions": dims,
        "clusters": [
            {"cluster_id": c.cluster_id, "key": c.key, "size": len(c.members),
             "members": c.members if level == "contract" else [ctx.kb.cards[m].number.value or ctx.kb.cards[m].file for m in c.members],
             "differing": c.differing}
            for c in clusters
        ],
    }


def compare_contracts_tool(ctx: ToolContext, contract_ids: list[str] | None = None, cluster_id: str | None = None) -> dict:
    ids = list(contract_ids or [])
    if cluster_id:
        found = [c for cls in ctx.kb.clusters.values() for c in cls if c.cluster_id == cluster_id and c.level == "contract"]
        if not found:
            raise ToolError(f"кластер {cluster_id} не найден", "сначала cluster_contracts(level='contract')")
        ids += found[0].members
    contracts = []
    for cid in dict.fromkeys(ids):
        c = ctx.kb.find_contract(cid)
        if c is None:
            raise ToolError(f"договор {cid} не найден", f"известные договоры: {sorted(ctx.kb.contracts)}")
        contracts.append(c)
    if len(contracts) < 2:
        raise ToolError("для сравнения нужно минимум два договора")
    result = compare_contracts(contracts, ctx.kb.consolidated)
    if result["not_consolidated"]:
        result["hint"] = f"для {result['not_consolidated']} нет свертки — вызовите consolidate_contract, чтобы сравнить условия"
    return result


def list_contracts(ctx: ToolContext) -> dict:
    if not ctx.kb.contracts:
        return {"status": "empty", "message": "база знаний пуста: сначала запустите анализ (python -m contract_agent analyze)"}
    return {
        "contracts": [
            {"contract_id": c.contract_id, "title": c.title, "date": c.date.value, "status": c.status,
             "subject_category": c.subject_category.value, "counterparty_form": c.counterparty_legal_form.value,
             "city": c.city, "documents": [t.number or t.doc_id for t in c.timeline],
             "main_document_in_package": c.main_doc_id is not None}
            for c in ctx.kb.contracts.values()
        ],
        "clusters": {k: [{"cluster_id": c.cluster_id, "key": c.key, "members": c.members} for c in v]
                     for k, v in ctx.kb.clusters.items()},
        "executive_summary": (ctx.kb.notes.get("executive_summary") or {}).get("text"),
    }


def get_contract_summary(ctx: ToolContext, contract_id: str) -> dict:
    c = ctx.kb.find_contract(contract_id)
    if c is None:
        return {"status": "empty", "message": f"договор {contract_id} не найден", "known": sorted(ctx.kb.contracts)}
    cons = ctx.kb.consolidated.get(c.contract_id)
    return {
        "contract": c.model_dump(mode="json", exclude={"timeline"}),
        "timeline": [t.model_dump(mode="json") for t in c.timeline],
        "current_terms": {
            TERM_RU[TermKey(k)]: [v.model_dump(mode="json") for v in st.active] for k, st in (cons.terms.items() if cons else [])
        },
        "history": {
            TERM_RU[TermKey(k)]: [v.model_dump(mode="json") for v in st.history] for k, st in (cons.terms.items() if cons else [])
        },
        "consolidation_notes": cons.notes if cons else ["свертка не выполнена"],
        "skipped_docs": cons.skipped_docs if cons else [],
        "analyst_note": (ctx.kb.notes.get("contracts") or {}).get(c.contract_id),
    }


def generate_report(ctx: ToolContext, executive_summary: str, cluster_notes: dict[str, str] | None = None,
                    contract_notes: dict[str, str] | None = None) -> dict:
    if not ctx.kb.contracts:
        raise ToolError("нет данных для отчёта", "сначала link_contract_chains и consolidate_contract")
    with ctx.lock:
        ctx.kb.notes = {
            "executive_summary": {"text": executive_summary.strip()},
            "clusters": {**(ctx.kb.notes.get("clusters") or {}), **(cluster_notes or {})},
            "contracts": {**(ctx.kb.notes.get("contracts") or {}), **(contract_notes or {})},
        }
        unknown = [k for k in (contract_notes or {}) if k not in ctx.kb.contracts]
        text = write_reports(ctx)
        ctx.kb.report_generated_at = datetime.now().isoformat(timespec="microseconds")
        ctx.kb.save()
    return {
        "path": str(ctx.settings.report_path),
        "full_report": str(ctx.settings.report_path.with_name(FULL_REPORT_NAME)),
        "chars": len(text),
        "sections": [ln for ln in text.splitlines() if ln.startswith("## ")],
        "warnings": [f"заметки для неизвестных договоров проигнорированы: {unknown}"] if unknown else [],
        "next": "вызовите check_plan_completion",
    }


def write_reports(ctx: ToolContext) -> str:
    """Пишет краткий report.md и детальный report_full.md; возвращает текст краткого."""
    if ctx.llm is not None and ctx.llm.usage.get("calls"):
        ctx.kb.usage = dict(ctx.llm.usage)
    ctx.settings.report_path.parent.mkdir(parents=True, exist_ok=True)
    brief = render_report(ctx.kb, ctx.settings.model, brief=True)
    ctx.settings.report_path.write_text(brief, encoding="utf-8")
    ctx.settings.report_path.with_name(FULL_REPORT_NAME).write_text(
        render_report(ctx.kb, ctx.settings.model, brief=False), encoding="utf-8")
    return brief


def check_plan_completion(ctx: ToolContext) -> dict:
    result = check_plan(ctx.kb, ctx.settings.report_path)
    with ctx.lock:
        ctx.kb.reflection = result
        ctx.kb.save()
        if ctx.settings.report_path.exists():  # добавляем чек-лист и метрики качества в отчёты
            write_reports(ctx)
    return result


# ---------------------------------------------------------------------------
# регистрация
# ---------------------------------------------------------------------------
def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


DOC_ID = {"type": "string", "description": "doc_id из list_documents (или номер документа вида D123456789-01)"}
CONTRACT_ID = {"type": "string", "description": "Номер договора (contract_id) или номер любого его соглашения"}

ALL_TOOLS: list[Tool] = [
    Tool("list_documents",
         "Инвентаризация входного пакета: все файлы с doc_id, форматом (pdf_text, pdf_scan, pdf_mixed, docx, text, image, "
         "unsupported, corrupted), числом страниц и подсказкой, каким методом читать каждый формат. Вызывай первым.",
         _schema(), list_documents, read_only=True),
    Tool("parse_document",
         "Читает документ выбранным методом. method: auto — текстовый слой, а страницы-сканы через OCR (по умолчанию); "
         "text_layer — только текстовый слой (быстро, без LLM); ocr — все страницы через OCR vision-моделью (для сканов, "
         "изображений и PDF с битым текстовым слоем). Метод выбирай по формату из list_documents. Возвращает метрики качества "
         "и recognized=true/false; если не распознан — status=error и подсказка, какой метод попробовать.",
         _schema({"doc_id": DOC_ID, "method": {"type": "string", "enum": ["auto", "text_layer", "ocr"]}}, ["doc_id"]),
         parse_document),
    Tool("extract_document_card",
         "Извлекает карточку документа (тип, номер, дата, родительский договор, предмет, существенные условия, "
         "изменения, признание соглашений недействительными) и проверяет каждое значение по тексту. "
         "Непроверенные значения не записываются и возвращаются в rejected с причиной. Карточки кешируются: "
         "force=true — извлечь заново.",
         _schema({"doc_id": DOC_ID, "force": {"type": "boolean"}}, ["doc_id"]),
         extract_document_card),
    Tool("get_document_card",
         "Полная карточка документа из базы знаний: все поля с цитатами, изменения, отклонённые значения.",
         _schema({"doc": DOC_ID}, ["doc"]), get_document_card, read_only=True),
    Tool("search_documents",
         "Полнотекстовый поиск по распознанным документам: возвращает фрагменты с номером страницы. Используй, "
         "чтобы проверить спорное значение или найти условие, которого нет в карточке.",
         _schema({"query": {"type": "string"}, "doc_ids": {"type": "array", "items": {"type": "string"}},
                  "max_hits": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query"]),
         search_documents, read_only=True),
    Tool("link_contract_chains",
         "Связывает карточки в цепочки «договор → соглашения» по номерам (не по папкам): ДС к ДС, "
         "недействительные соглашения, отсутствующие основные договоры, противоречия дат. Перезапускай после "
         "изменения карточек.",
         _schema(), link_contract_chains),
    Tool("consolidate_contract",
         "Свертка договора: применяет действующие соглашения в хронологическом порядке к базовой редакции и "
         "возвращает актуальные условия и историю их изменений.",
         _schema({"contract_id": CONTRACT_ID}, ["contract_id"]), consolidate_contract),
    Tool("cluster_contracts",
         "Кластеризация. level=contract — договоры (цепочки), измерения: " + ", ".join(CONTRACT_DIMENSIONS)
         + ". level=document — отдельные документы, измерения: " + ", ".join(DOCUMENT_DIMENSIONS)
         + ". По умолчанию — договоры по предмету. Возвращает кластеры и различающиеся признаки внутри них.",
         _schema({"level": {"type": "string", "enum": ["contract", "document"]},
                  "dimensions": {"type": "array", "items": {"type": "string"}}}),
         cluster_contracts),
    Tool("compare_contracts",
         "Сравнивает договоры (список или кластер): метаданные и актуальные условия после свертки — что совпадает, "
         "что различается, какие условия есть только у части договоров.",
         _schema({"contract_ids": {"type": "array", "items": {"type": "string"}}, "cluster_id": {"type": "string"}}),
         compare_contracts_tool, read_only=True),
    Tool("list_contracts",
         "Обзор базы знаний: договоры, их документы, статус, кластеры и резюме аналитика.",
         _schema(), list_contracts, read_only=True),
    Tool("get_contract_summary",
         "Всё о договоре: метаданные, цепочка документов, актуальные условия, история изменений каждого условия, "
         "замечания и комментарий аналитика.",
         _schema({"contract_id": CONTRACT_ID}, ["contract_id"]), get_contract_summary, read_only=True),
    Tool("generate_report",
         "Формирует Markdown-отчёт из базы знаний и добавляет твои аналитические заметки. "
         "cluster_notes: {cluster_id: текст}, contract_notes: {contract_id: текст}.",
         _schema({"executive_summary": {"type": "string"},
                  "cluster_notes": {"type": "object", "additionalProperties": {"type": "string"}},
                  "contract_notes": {"type": "object", "additionalProperties": {"type": "string"}}},
                 ["executive_summary"]),
         generate_report),
    Tool("check_plan_completion",
         "Reflection: формальная проверка, что все пункты плана выполнены (документы распознаны, карточки есть, "
         "цепочки, свертки, кластеры, заметки, отчёт). Возвращает complete и список недоделанного (todo).",
         _schema(), check_plan_completion),
]

CHAT_TOOL_NAMES = ("list_contracts", "get_contract_summary", "get_document_card", "search_documents", "compare_contracts")


def analyst_registry() -> ToolRegistry:
    return ToolRegistry([t for t in ALL_TOOLS if t.name not in ("list_contracts",)])


def chat_registry() -> ToolRegistry:
    return ToolRegistry([t for t in ALL_TOOLS if t.name in CHAT_TOOL_NAMES])


def dumps(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
