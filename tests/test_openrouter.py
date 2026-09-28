"""Бэкенд OpenRouter без сети: подменяем клиент openai фейком и проверяем перевод форматов."""
import json
from types import SimpleNamespace

import pytest

from contract_agent.config import Settings
from contract_agent.llm import LLMError
from contract_agent.llm_openrouter import Block, OpenRouterClient, to_openai_messages
from contract_agent.schemas import OcrResult


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message, finish, cost = self.replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=None, cost=cost, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)


def fake_client(replies):
    comp = FakeCompletions(replies)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def msg(content=None, tool_calls=None, reasoning=None, details=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=reasoning, reasoning_details=details, model_extra={})


def test_history_translation_keeps_tool_calls_results_and_reasoning():
    history = [
        {"role": "user", "content": "задача"},
        {"role": "assistant", "content": [Block(type="thinking", thinking="…", details=[{"type": "reasoning.text", "signature": "s"}]),
                                          Block(type="text", text="Мысль: начинаю"),
                                          Block(type="tool_use", id="c1", name="list_documents", input={})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": '{"status":"error"}', "is_error": True}]},
        {"role": "user", "content": "Reflection: доделай"},
    ]
    out = to_openai_messages("SYS", history)
    assert out[0]["role"] == "system"
    assert out[2]["tool_calls"][0]["function"]["name"] == "list_documents"
    assert out[2]["reasoning_details"][0]["signature"] == "s"
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": '[ОШИБКА ИНСТРУМЕНТА] {"status":"error"}'}
    assert out[4] == {"role": "user", "content": "Reflection: доделай"}


def test_agent_turn_parses_tool_calls_and_counts_cost():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="parse_document", arguments='{"doc_id": "doc_1"}'))
    client, comp = fake_client([(msg("Мысль: читаю", [call], reasoning="думаю"), "tool_calls", 0.012)])
    llm = OpenRouterClient(Settings(provider="openrouter", model="anthropic/claude-opus-5"), client=client)
    resp = llm.agent_turn("SYS", [{"role": "user", "content": "x"}],
                          [{"name": "parse_document", "description": "d", "input_schema": {"type": "object", "properties": {}}}])
    assert [b.type for b in resp.content] == ["thinking", "text", "tool_use"]
    assert resp.content[2].input == {"doc_id": "doc_1"} and resp.stop_reason == "tool_use"
    req = comp.requests[0]
    assert req["tools"][0]["function"]["name"] == "parse_document"
    assert req["extra_body"]["reasoning"] == {"effort": "high"} and req["extra_body"]["usage"] == {"include": True}
    assert llm.usage["cost_usd"] == 0.012


def test_structured_repairs_invalid_json_once_and_sends_images():
    bad = msg('{"pages": [{"page": 1}]}')
    good = msg(json.dumps({"pages": [{"page": 1, "text": "ДОГОВОР", "legibility": "high", "uncertain_fragments": []}]}))
    client, comp = fake_client([(bad, "stop", 0.01), (good, "stop", 0.01)])
    llm = OpenRouterClient(Settings(provider="openrouter"), client=client)
    res = llm.ocr_pages([(1, b"png")], doc_path="a.pdf")
    assert isinstance(res, OcrResult) and res.pages[0].text == "ДОГОВОР"
    first = comp.requests[0]
    assert first["response_format"]["type"] == "json_schema"
    assert first["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "не соответствует схеме" in comp.requests[1]["messages"][-1]["content"]


def test_cost_cap_stops_calls():
    client, _ = fake_client([(msg("ok"), "stop", 5.0)])
    llm = OpenRouterClient(Settings(provider="openrouter", max_cost_usd=1.0), client=client)
    llm.agent_turn("SYS", [{"role": "user", "content": "x"}], [])
    with pytest.raises(LLMError, match="лимит расходов"):
        llm.agent_turn("SYS", [{"role": "user", "content": "x"}], [])
