"""Reflection: проверка результата на соответствие цели.

Чек-лист формальный и вычисляется кодом по базе знаний и файлу отчёта — агент не может
«убедить себя», что всё сделано. Результат возвращается агенту; если пункты не выполнены,
агент обязан их доделать (это же проверяет цикл агента перед завершением).
"""
from __future__ import annotations

from pathlib import Path

from .knowledge_base import KnowledgeBase
from .report import REQUIRED_SECTIONS


def check_plan(kb: KnowledgeBase, report_path: Path) -> dict:
    items: list[dict] = []

    def add(key: str, title: str, ok: bool, detail: str = "") -> None:
        items.append({"key": key, "title": title, "ok": ok, "detail": detail})

    docs = kb.documents
    add("inventory", "Пакет документов проинвентаризирован", bool(docs), f"{len(docs)} файлов" if docs else "документы не найдены")

    not_parsed = [d.file for i, d in docs.items() if i not in kb.parse_status]
    add("parsed", "Все документы прочитаны (текстовый слой или OCR)", not not_parsed,
        f"не обработаны: {not_parsed}" if not_parsed else "")

    unrecognized = [docs[i].file for i, s in kb.parse_status.items() if not s.get("recognized") and i in docs]
    unrecognized += [c.file for c in kb.cards.values() if c.status == "unrecognized"]
    missing_cards = [docs[i].file for i, s in kb.parse_status.items() if s.get("recognized") and i not in kb.cards and i in docs]
    add("cards", "По каждому распознанному документу есть проверенная карточка", not missing_cards,
        (f"нет карточек: {missing_cards}. " if missing_cards else "") + (f"явно помечены как нераспознанные: {unrecognized}" if unrecognized else ""))

    linked_docs = {d for c in kb.contracts.values() for d in ([c.main_doc_id] if c.main_doc_id else []) + c.supplement_doc_ids}
    unlinked = [c.file for c in kb.cards.values() if c.status != "unrecognized" and c.doc_id not in linked_docs]
    add("chains", "Документы связаны в цепочки договоров", bool(kb.contracts) and not unlinked,
        f"вне цепочек: {unlinked}" if unlinked else f"{len(kb.contracts)} договоров")

    not_cons = sorted(set(kb.contracts) - set(kb.consolidated))
    add("consolidated", "Для каждого договора выполнена свертка актуальных условий", bool(kb.contracts) and not not_cons,
        f"без свертки: {not_cons}" if not_cons else "")

    subject_keys = [k for k in kb.clusters if k.startswith("contract:") and "subject_category" in k]
    add("clusters", "Договоры кластеризованы по предмету", bool(subject_keys))

    exec_ok = bool((kb.notes.get("executive_summary") or {}).get("text"))
    multi = [cl.cluster_id for k in subject_keys for cl in kb.clusters[k] if len(cl.members) > 1]
    notes_clusters = kb.notes.get("clusters") or {}
    missing_cluster_notes = [cid for cid in multi if cid not in notes_clusters]
    notes_contracts = kb.notes.get("contracts") or {}
    missing_contract_notes = sorted(set(kb.contracts) - set(notes_contracts))
    add("diff_notes", "Описаны различия договоров внутри кластеров", bool(subject_keys) and not missing_cluster_notes,
        f"нет комментария для кластеров: {missing_cluster_notes}" if missing_cluster_notes else "")
    add("evolution_notes", "Описана эволюция условий каждого договора", bool(kb.contracts) and not missing_contract_notes,
        f"нет комментария для: {missing_contract_notes}" if missing_contract_notes else "")
    add("summary", "Есть резюме для руководителя", exec_ok)

    # отчёт должен быть построен ПОСЛЕ последнего изменения данных (карточек, цепочек, сверток, кластеров),
    # иначе в нём устаревшие выводы
    report_ok, detail = False, "отчёт не сформирован (generate_report)"
    if report_path.exists() and kb.report_generated_at:
        stale = bool(kb.changed_at and kb.report_generated_at < kb.changed_at)
        missing = [s for s in REQUIRED_SECTIONS if s not in report_path.read_text(encoding="utf-8")]
        report_ok = not stale and not missing
        detail = ("отчёт устарел: данные менялись после генерации — вызовите generate_report ещё раз" if stale
                  else f"нет разделов: {missing}" if missing else str(report_path))
    add("report", "Отчёт сформирован по актуальным данным", report_ok, detail)

    # пустая свертка допустима (например, все соглашения истекли или отменены), но агент обязан объяснить её
    empty = [cid for cid, cons in kb.consolidated.items() if not any(st.active for st in cons.terms.values())]
    unexplained = [cid for cid in empty if not notes_contracts.get(cid)]
    add("consolidation_content", "У каждого договора есть действующие условия или объяснение, почему их нет",
        bool(kb.consolidated) and not unexplained,
        f"пустая свертка без комментария аналитика: {unexplained}" if unexplained
        else (f"пустая свертка объяснена в комментарии: {empty}" if empty else ""))

    complete = all(i["ok"] for i in items)
    return {"complete": complete, "items": items, "quality": quality_metrics(kb),
            "todo": [f"{i['title']}: {i['detail']}" for i in items if not i["ok"]]}


def quality_metrics(kb: KnowledgeBase) -> dict[str, str]:
    """Метрики качества работы агента: сколько распознано, сколько значений подтверждено, сколько отклонено."""
    docs = len(kb.documents)
    recognized = sum(1 for s in kb.parse_status.values() if s.get("recognized"))
    ocr = sum(1 for s in kb.parse_status.values() if s.get("quality", {}).get("source") in ("ocr", "mixed"))
    low_pages = sum(len(s.get("quality", {}).get("low_legibility_pages", [])) for s in kb.parse_status.values())
    cards = list(kb.cards.values())
    by_status = {st: sum(1 for c in cards if c.status == st) for st in ("ok", "partial", "unrecognized")}

    scalar = ("number", "date", "parent_number", "parent_date", "counterparty", "effective_from", "valid_until")
    confidences = [getattr(c, f).confidence for c in cards for f in scalar if getattr(c, f).value]
    confidences += [k.confidence for c in cards for k in c.key_terms] + [ch.confidence for c in cards for ch in c.changes]
    verified = len(confidences)
    rejected = sum(len(c.rejected) for c in cards)
    warnings = sum(1 for c in kb.contracts.values() for i in c.issues if i.severity.value == "warning")
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "—"  # noqa: E731
    return {
        "Документов распознано": pct(recognized, docs),
        "Из них через OCR (страниц с низкой разборчивостью)": f"{ocr} ({low_pages})",
        "Карточки: ok / partial / unrecognized": f"{by_status['ok']} / {by_status['partial']} / {by_status['unrecognized']}",
        "Значений подтверждено проверками": pct(verified, verified + rejected),
        "Отказов от записи": str(rejected),
        "Средняя уверенность записанных значений": f"{sum(confidences) / len(confidences):.2f}" if confidences else "—",
        "Договоров без основного документа в пакете": str(sum(1 for c in kb.contracts.values() if not c.main_doc_id)),
        "Противоречий и предупреждений по договорам": str(warnings),
    }
