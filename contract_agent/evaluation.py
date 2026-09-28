"""Оценка качества на размеченном наборе.

Эталон (gold.json) генерируется вместе с синтетическим пакетом из той же спецификации — до любого
прогона модели, поэтому подогнать его под ответы нельзя. Сравниваются:

* поля карточек (тип, номер, даты, родительский договор, форма контрагента, отменённые ДС). Исходы:
  correct — совпало со значением; empty — пусто и в эталоне, и у агента (считается отдельно, это не «попадание»);
  wrong — записано неверное значение (опасная ошибка); extra — записано значение, которого нет в эталоне;
  refused — не записано, потому что не прошло проверки (безопасный отказ); missed — не найдено;
* изменения из доп. соглашений (изменяемый документ + пункт + действие): точность и полнота;
* цепочки: к какому договору отнесён каждый документ, есть ли основной договор, какие ДС отменены.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .analysis import same_clause
from .knowledge_base import KnowledgeBase
from .validators import normalize_number, parse_date

FIELDS = ("doc_type", "number", "date", "parent_number", "parent_date", "effective_from", "valid_until",
          "counterparty_legal_form", "invalidates")
DATE_FIELDS = {"date", "parent_date", "effective_from", "valid_until"}
NUMBER_FIELDS = {"number", "parent_number"}
OUTCOMES = ("correct", "empty", "wrong", "extra", "refused", "missed")


def _norm(field: str, value):
    if value in (None, "", [], "unknown"):
        return None
    if field in DATE_FIELDS:
        d = parse_date(value)
        return d.isoformat() if d else value
    if field in NUMBER_FIELDS:
        return normalize_number(value)
    if field == "invalidates":
        return tuple(sorted(normalize_number(v) for v in value))
    return value


def _live(card, field: str):
    if field == "invalidates":
        return [r.number for r in card.invalidates]
    v = getattr(card, field)
    return v.value if hasattr(v, "value") else v


def _has_refs(clause: str) -> bool:
    return bool(re.search(r"(п\.|пункт|раздел|приложени)\s*№?\s*\d", clause, re.IGNORECASE))


def _match_changes(gold: list[dict], live: list) -> tuple[int, int, int]:
    """(совпало, всего в эталоне, всего у агента). Совпадение: тот же изменяемый документ, действие и пункт;
    если пункт в эталоне назван словами, без номера («дополнительные услуги»), — документ и действие.
    Признание ДС недействительным считается не изменением, а полем invalidates."""
    live = [c for c in live if c.action.value != "invalidate_agreement"]
    used, hits = set(), 0
    for g in gold:
        for i, c in enumerate(live):
            if i in used or c.action.value != g["action"]:
                continue
            if normalize_number(c.target_document) != normalize_number(g["target_document"]):
                continue
            if (same_clause(c.target_clause, g["target_clause"]) or c.target_clause.strip() == g["target_clause"].strip()
                    or not _has_refs(g["target_clause"])):
                used.add(i)
                hits += 1
                break
    return hits, len(gold), len(live)


def evaluate(kb: KnowledgeBase, gold: dict) -> dict:
    docs_gold = gold["documents"]
    by_path = {info.path: doc_id for doc_id, info in kb.documents.items()}
    totals = dict.fromkeys(OUTCOMES, 0)
    per_field = {f: dict.fromkeys(OUTCOMES, 0) for f in FIELDS}
    rows, missing_docs = [], []
    ch_hit = ch_gold = ch_live = 0

    for path, g in sorted(docs_gold.items()):
        card = kb.cards.get(by_path.get(path, ""))
        if card is None:
            missing_docs.append(path)
            continue
        rejected_fields = {i.field for i in card.rejected}
        for field in FIELDS:
            gv, lv = _norm(field, g.get(field)), _norm(field, _live(card, field))
            if lv == gv:
                outcome = "empty" if gv is None else "correct"
            elif lv is None:
                outcome = "refused" if field in rejected_fields else "missed"
            elif gv is None:
                outcome = "extra"
            else:
                outcome = "wrong"
            totals[outcome] += 1
            per_field[field][outcome] += 1
            if outcome not in ("correct", "empty"):
                rows.append({"file": Path(path).name, "field": field, "gold": gv, "live": lv, "outcome": outcome})
        h, ng, nl = _match_changes(g.get("changes", []), card.changes)
        ch_hit, ch_gold, ch_live = ch_hit + h, ch_gold + ng, ch_live + nl

    filled = totals["correct"] + totals["wrong"] + totals["extra"]
    return {
        "documents": len(docs_gold) - len(missing_docs),
        "missing_documents": missing_docs,
        "totals": totals,
        "precision": round(totals["correct"] / filled, 3) if filled else None,  # доля верных среди записанных значений
        "per_field": per_field,
        "mismatches": rows,
        "changes": {"matched": ch_hit, "gold": ch_gold, "extracted": ch_live,
                    "recall": round(ch_hit / ch_gold, 3) if ch_gold else None,
                    "precision": round(ch_hit / ch_live, 3) if ch_live else None},
        "structure": _compare_structure(kb, gold),
    }


def _compare_structure(kb: KnowledgeBase, gold: dict) -> dict:
    number_of = {d: c.number.value for d, c in kb.cards.items()}
    live_chains = {cid: {number_of.get(d) for d in ([c.main_doc_id] if c.main_doc_id else []) + c.supplement_doc_ids}
                   for cid, c in kb.contracts.items()}
    wrong = {}
    for cid, spec in gold.get("chains", {}).items():
        live = live_chains.get(normalize_number(cid))
        expected = {normalize_number(n) for n in spec["docs"]}
        if live != expected:
            wrong[cid] = {"gold": sorted(expected), "live": sorted(x for x in (live or set()) if x)}
        elif bool(kb.contracts[normalize_number(cid)].main_doc_id) != spec["main"]:
            wrong[cid] = {"gold": "основной договор " + ("есть" if spec["main"] else "отсутствует"), "live": "иначе"}
    invalid_live = sorted({t.number for c in kb.contracts.values() for t in c.timeline if not t.valid and t.number})
    return {
        "chains_correct": f"{len(gold.get('chains', {})) - len(wrong)}/{len(gold.get('chains', {}))}",
        "wrong_chains": wrong,
        "extra_contracts": sorted(set(kb.contracts) - {normalize_number(c) for c in gold.get("chains", {})}),
        "invalidated_gold": sorted(normalize_number(n) for n in gold.get("invalidated", [])),
        "invalidated_live": invalid_live,
    }


def render_markdown(result: dict, kb_path: str, gold_path: str) -> str:
    t, ch, s = result["totals"], result["changes"], result["structure"]
    pct = lambda x: f"{x:.1%}" if x is not None else "—"  # noqa: E731
    out = [
        "# Оценка качества на размеченном наборе\n",
        f"_База знаний: `{kb_path}`; эталон: `{gold_path}` (генерируется вместе с пакетом, до прогона модели)._\n",
        "## Поля карточек\n",
        "| Исход | Кол-во | Что значит |",
        "|---|---|---|",
        f"| correct | {t['correct']} | значение записано и совпало с эталоном |",
        f"| empty | {t['empty']} | пусто и в эталоне, и у агента (не считается попаданием) |",
        f"| wrong | {t['wrong']} | **записано неверное значение** |",
        f"| extra | {t['extra']} | записано значение, которого нет в эталоне |",
        f"| refused | {t['refused']} | не записано: не прошло проверки (безопасный отказ) |",
        f"| missed | {t['missed']} | значение не найдено |",
        "",
        f"Доля верных среди записанных значений: **{pct(result['precision'])}** "
        f"({t['correct']} из {t['correct'] + t['wrong'] + t['extra']}); документов: {result['documents']}.\n",
        "## Изменения из доп. соглашений\n",
        f"Найдено {ch['matched']} из {ch['gold']} изменений эталона (полнота {pct(ch['recall'])}); "
        f"из {ch['extracted']} записанных агентом совпадают с эталоном {ch['matched']} (точность {pct(ch['precision'])}). "
        "Совпадение — тот же изменяемый документ, пункт и действие; содержание новой редакции проверяется валидаторами, "
        "а не этой метрикой.\n",
        "## Цепочки\n",
        f"- Цепочки собраны верно: {s['chains_correct']}",
        f"- Лишние договоры: {s['extra_contracts'] or 'нет'}",
        f"- Отменённые ДС: эталон {s['invalidated_gold']}, агент {s['invalidated_live']}",
    ]
    for cid, v in s["wrong_chains"].items():
        out.append(f"  - {cid}: эталон {v['gold']}, агент {v['live']}")
    out += ["", "## По полям\n", "| Поле | " + " | ".join(OUTCOMES) + " |", "|---" * (len(OUTCOMES) + 1) + "|"]
    for f, c in result["per_field"].items():
        out.append(f"| {f} | " + " | ".join(str(c[o]) for o in OUTCOMES) + " |")
    out += ["", "## Расхождения\n", "| Документ | Поле | Эталон | Агент | Исход |", "|---|---|---|---|---|"]
    for r in result["mismatches"]:
        out.append(f"| {r['file']} | {r['field']} | {r['gold']} | {r['live']} | {r['outcome']} |")
    if not result["mismatches"]:
        out.append("| — | — | — | — | расхождений нет |")
    return "\n".join(out) + "\n"


def run(kb_path: Path, gold_path: Path, out_path: Path | None) -> dict:
    kb = KnowledgeBase.load(kb_path)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    result = evaluate(kb, gold)
    if out_path:
        out_path.write_text(render_markdown(result, kb_path.as_posix(), gold_path.as_posix()), encoding="utf-8")
    return result
