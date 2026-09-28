"""Цикл агента и инструменты без обращения к API: модель заменена сценарием (ScriptedLLM)."""
import json
import shutil
from types import SimpleNamespace

import pytest

from contract_agent.agent import Agent
from contract_agent.ingest import DocumentLoader
from contract_agent.knowledge_base import KnowledgeBase
from contract_agent.llm import LLMError
from contract_agent.prompts import ANALYST_SYSTEM
from contract_agent.schemas import DocType, DocumentExtraction, EvidencedValue, LegalForm, SubjectCategory
from contract_agent.tools import ToolContext, analyst_registry
from contract_agent.tracing import Tracer



def text(t):
    return SimpleNamespace(type="text", text=t)


def call(name, args, id_):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


class ScriptedLLM:
    """Та же поверхность, что у LLMClient: agent_turn / structured / ocr_pages / usage."""

    def __init__(self, settings, turns, extraction=None):
        self.settings = settings
        self.turns = list(turns)
        self.extraction = extraction
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "calls": 0}
        self.seen_messages = []

    def agent_turn(self, system, messages, tools):
        self.seen_messages.append(list(messages))
        self.usage["calls"] += 1
        step = self.turns.pop(0)
        if isinstance(step, Exception):
            raise step
        content = step(messages) if callable(step) else step
        return SimpleNamespace(content=content, stop_reason="tool_use" if any(b.type == "tool_use" for b in content) else "end_turn")

    def structured(self, system, content, schema):
        if self.extraction is None:
            raise LLMError("нет сценария извлечения")
        return self.extraction

    def ocr_pages(self, pages, doc_path=None):
        raise LLMError("OCR недоступен в тесте")


@pytest.fixture
def small_input(tmp_path, sample_dir):
    src = tmp_path / "input" / "Партнёр"
    src.mkdir(parents=True)
    for name in ("ДС_консервация.pdf", "Партнёрский_договор_скан.pdf"):
        shutil.copy(sample_dir / "Партнёр" / name, src / name)
    return tmp_path / "input"


def make_ctx(settings, llm):
    return ToolContext(settings=settings, kb=KnowledgeBase(settings.kb_path), loader=DocumentLoader(settings, llm), llm=llm)


def read_log(tracer):
    return [json.loads(line) for line in tracer.path.read_text(encoding="utf-8").splitlines()]


def test_tool_errors_are_returned_as_data(settings, small_input):
    settings = settings.with_overrides(input_dir=small_input)
    ctx = make_ctx(settings, None)
    reg = analyst_registry()

    res, err = reg.execute(ctx, "no_such_tool", {})
    assert err and "неизвестный инструмент" in res["error"]

    res, err = reg.execute(ctx, "parse_document", {"doc_idd": "x"})
    assert err and "неверные аргументы" in res["error"]

    res, err = reg.execute(ctx, "parse_document", {"doc_id": "doc_missing"})
    assert err and "list_documents" in res["hint"]

    res, err = reg.execute(ctx, "consolidate_contract", {"contract_id": "D1"})
    assert err and "link_contract_chains" in res["hint"]

    res, err = reg.execute(ctx, "list_documents", {})
    assert not err and res["total"] == 2 and res["by_format"] == {"pdf_scan": 1, "pdf_text": 1}

    scan = next(d["doc_id"] for d in res["documents"] if d["format"] == "pdf_scan")
    res, err = reg.execute(ctx, "parse_document", {"doc_id": scan})
    assert err and "нужен LLM" in res["error"]  # скан без LLM прочитать нельзя — явная ошибка, а не пустой текст
    res, err = reg.execute(ctx, "parse_document", {"doc_id": scan, "method": "text_layer"})
    assert err and res["recognized"] is False  # текстовый слой пустой — документ не распознан
    assert ctx.kb.parse_status[scan]["recognized"] is False

    res, err = reg.execute(ctx, "search_documents", {"query": "Совместного предложения"})
    assert res["status"] == "empty" and "parse_document" in res["message"]  # поиск только по прочитанным документам
    text_doc = next(d for d in ctx.kb.documents.values() if d.format == "pdf_text").doc_id
    reg.execute(ctx, "parse_document", {"doc_id": text_doc})

    res, err = reg.execute(ctx, "search_documents", {"query": "аренда нежилого помещения склада"})
    assert not err and res["status"] == "empty"

    res, err = reg.execute(ctx, "search_documents", {"query": "Совместного предложения"})
    assert res["status"] == "ok" and res["hits"][0]["page"] == 1


def test_agent_loop_logs_thoughts_handles_errors_and_reflects(settings, small_input):
    settings = settings.with_overrides(input_dir=small_input, max_reflection_rounds=1)

    def parse_all(messages):
        listing = json.loads(messages[-1]["content"][0]["content"])
        ids = [d["doc_id"] for d in listing["documents"]]
        return [text("Мысль: два документа, один скан.\nПлан: прочитать оба.\nСледующий шаг: parse_document параллельно.")] + [
            call("parse_document", {"doc_id": i}, f"p{n}") for n, i in enumerate(ids)
        ]

    turns = [
        [text("Мысль: начинаю с инвентаризации."), call("list_documents", {}, "t1")],
        parse_all,
        [text("Готово.")],                          # преждевременное завершение → reflection вернёт к работе
        [text("Больше ничего сделать не могу.")],   # второй раз — лимит раундов, завершение с пометкой
    ]
    llm = ScriptedLLM(settings, turns)
    ctx = make_ctx(settings, llm)
    tracer = Tracer(settings.logs_dir, "test", verbose=False)
    result = Agent(ctx, analyst_registry(), ANALYST_SYSTEM, tracer, reflection_gate=True).run("проанализируй")

    assert result.completed and result.plan_complete is False  # analyze вернёт код 1
    assert "Не все пункты плана выполнены" in result.text

    # результаты параллельных вызовов вернулись в одном сообщении, ошибка помечена is_error
    tool_msg = llm.seen_messages[2][-1]["content"]
    assert [r["tool_use_id"] for r in tool_msg] == ["p0", "p1"]
    assert sum(1 for r in tool_msg if r.get("is_error")) == 1

    # reflection-промпт отправлен агенту
    assert "Reflection" in llm.seen_messages[3][-1]["content"]

    events = [e["event"] for e in read_log(tracer)]
    assert events.count("thought") >= 3
    assert "tool_call" in events and "tool_result" in events
    assert events.count("reflection") == 2
    assert events[-1] == "final"


def test_agent_stops_gracefully_on_llm_error(settings, small_input):
    settings = settings.with_overrides(input_dir=small_input)
    llm = ScriptedLLM(settings, [LLMError("Нет доступа к Anthropic API")])
    tracer = Tracer(settings.logs_dir, "test", verbose=False)
    result = Agent(make_ctx(settings, llm), analyst_registry(), ANALYST_SYSTEM, tracer, reflection_gate=True).run("x")
    assert not result.completed and "Нет доступа" in result.text


def test_extraction_tool_validates_llm_output(settings, small_input):
    settings = settings.with_overrides(input_dir=small_input)
    ev = lambda v, q, c=0.95: EvidencedValue(value=v, quote=q, confidence=c)  # noqa: E731
    fake = DocumentExtraction(
        doc_type=DocType.SUPPLEMENTARY, title="Дополнительное соглашение",
        number=ev("ПД-2022/045-ДС6", "Дополнительное соглашение № ПД-2022/045-ДС6"),
        date=ev("15.04.2024", "подписано 15 апреля"),  # цитаты нет в тексте → отказ
        city="г. Томск",
        parent_number=ev("ПД-2022/045", "К Договору № ПД-2022/045"), parent_date=ev("12.05.2022", "от «12» мая 2022 г."),
        parent_title="Договор", counterparty=ev(None, None, 0), counterparty_legal_form=LegalForm.UNKNOWN, signatory_branch=None,
        subject_category=SubjectCategory.JOINT_PROMOTION, subject_summary="прекращение совместного продвижения",
        effective_from=ev(None, None, 0), valid_until=ev(None, None, 0), key_terms=[], changes=[], invalidates=[], related_contracts=[],
        anonymized_fields=["подписанты"], extraction_notes=[],
    )
    ctx = make_ctx(settings, ScriptedLLM(settings, [], extraction=fake))
    reg = analyst_registry()
    docs, _ = reg.execute(ctx, "list_documents", {})
    doc = next(d["doc_id"] for d in docs["documents"] if "консервация" in d["path"])
    res, err = reg.execute(ctx, "extract_document_card", {"doc_id": doc})
    assert not err
    card = res["card"]
    assert card["number"] == "ПД-2022/045-ДС6" and card["date"] is None
    assert any(r.startswith("date:") for r in card["rejected"])

    # повторный вызов берёт карточку из базы знаний, а база знаний переживает перезагрузку
    res, _ = reg.execute(ctx, "extract_document_card", {"doc_id": doc})
    assert res["cached"] is True
    assert KnowledgeBase.load(settings.kb_path).cards[doc].number.value == "ПД-2022/045-ДС6"
