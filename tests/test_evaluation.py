"""Проверка пролонгации, перепроверка карточек и оценка качества по эталону."""
import json
from pathlib import Path

from contract_agent.evaluation import evaluate
from contract_agent.ingest import DocumentInfo, PageText, ParsedDocument
from contract_agent.knowledge_base import KnowledgeBase
from contract_agent.schemas import DocType, EvidencedValue, Severity
from contract_agent.validators import build_card, revalidate_card

from .test_validators import extraction

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"


def _orion_like():
    text = ("ДОГОВОР № ПД-2022/045 г. Томск «12» мая 2022г. 9.1. Договор вступает в силу после его подписания "
            "последней из Сторон и действует до «31» декабря 2022 года. Срок действия автоматически продлевается на каждый "
            "последующий календарный год в случае, если ни одна из Сторон не уведомит другую Сторону о его расторжении.")
    parsed = ParsedDocument("doc_o", "o.pdf", [PageText(1, text, "text_layer")])
    info = DocumentInfo("doc_o", "o.pdf", "f", "o.pdf", 1, 1, 1, False)
    ex = extraction(doc_type=DocType.MAIN_CONTRACT,
                    number=EvidencedValue(value="ПД-2022/045", quote="ДОГОВОР № ПД-2022/045", confidence=0.95),
                    valid_until=EvidencedValue(value="31.12.2022", quote="действует до «31» декабря 2022 года", confidence=0.9))
    return info, parsed, ex


def test_end_date_with_auto_prolongation_is_refused():
    info, parsed, ex = _orion_like()
    card = build_card(info, parsed, ex, 0.7, 0.6)
    assert card.valid_until.value is None
    assert any(i.field == "valid_until" and i.severity == Severity.REJECTED and "пролонгации" in i.reason for i in card.issues)


def test_revalidation_is_idempotent_and_keeps_previous_refusals():
    info, parsed, ex = _orion_like()
    card = build_card(info, parsed, ex, 0.7, 0.6)
    again = revalidate_card(card, info, parsed, 0.7, 0.6)
    assert again.number.value == card.number.value == "ПД-2022/045"
    assert [i.reason for i in again.rejected] == [i.reason for i in card.rejected]  # отказ не потерян и не задвоен


def test_demo_pipeline_matches_gold(tmp_path):
    """Офлайн-демо на синтетическом пакете против эталона, сгенерированного вместе с пакетом."""
    from contract_agent.config import Settings
    from contract_agent.demo import run_demo

    package = ROOT / "examples" / "synthetic_package"
    gold = json.loads((package / "gold.json").read_text(encoding="utf-8"))
    settings = Settings(input_dir=package / "docs", work_dir=tmp_path, as_of=gold["as_of"], our_party=gold["our_party"])
    run_demo(settings, DEMO / "cassette", verbose=False)
    result = evaluate(KnowledgeBase.load(settings.kb_path), gold)

    assert result["documents"] == 12 and result["totals"]["wrong"] == 0 and result["totals"]["extra"] == 0
    # в кассету заложены ошибки модели; все они должны стать отказами, а не записанными значениями.
    # Среди полей карточек эталона остаётся только неуверенно прочитанная дата на скане (и производная от неё)
    assert sorted((r["file"], r["field"], r["outcome"]) for r in result["mismatches"]) == [
        ("ДС3 аренда скан.pdf", "date", "refused"), ("ДС3 аренда скан.pdf", "effective_from", "refused")]
    assert result["changes"]["matched"] == result["changes"]["gold"] == 12
    assert result["structure"]["chains_correct"] == "4/4"
    assert result["structure"]["invalidated_live"] == result["structure"]["invalidated_gold"] == ["П-118/22-2"]
