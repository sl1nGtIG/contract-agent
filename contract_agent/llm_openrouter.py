"""Бэкенд OpenRouter (OpenAI-совместимый Chat Completions API).

Интерфейс тот же, что у ``llm.LLMClient``: ``agent_turn`` / ``structured`` / ``ocr_pages`` / ``usage``.
Цикл агента хранит историю в формате Anthropic Messages (блоки text / tool_use / tool_result),
а этот модуль переводит её в формат Chat Completions и обратно — поэтому agent.py, tools.py
и остальной код не знают, через какого провайдера идут вызовы.
"""
from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .llm import LLMError
from .prompts import OCR_SYSTEM
from .schemas import OcrResult

if TYPE_CHECKING:
    from .config import Settings

T = TypeVar("T", bound=BaseModel)

BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Block:
    """Блок ответа в форме, которую понимает цикл агента (как у Anthropic SDK)."""
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    thinking: str | None = None
    details: list | None = None   # reasoning_details OpenRouter: возвращаются модели на следующем шаге


@dataclass
class Response:
    content: list[Block]
    stop_reason: str


def _get(block: Any, key: str, default=None):
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """История в формате Anthropic → формат Chat Completions."""
    out: list[dict] = [{"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}]
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "\n".join(_get(b, "text") or "" for b in content if _get(b, "type") == "text").strip()
            calls = [
                {"id": _get(b, "id"), "type": "function",
                 "function": {"name": _get(b, "name"), "arguments": json.dumps(_get(b, "input") or {}, ensure_ascii=False)}}
                for b in content if _get(b, "type") == "tool_use"
            ]
            item: dict = {"role": "assistant", "content": text or None}
            if calls:
                item["tool_calls"] = calls
            details = [d for b in content if _get(b, "type") == "thinking" for d in (_get(b, "details") or [])]
            if details:  # рассуждения с подписью нужно вернуть, иначе модель теряет их между вызовами инструментов
                item["reasoning_details"] = details
            out.append(item)
            continue
        # user: результаты инструментов и/или текст
        texts = []
        for b in content:
            if _get(b, "type") == "tool_result":
                body = _get(b, "content")
                if _get(b, "is_error"):
                    body = f"[ОШИБКА ИНСТРУМЕНТА] {body}"
                out.append({"role": "tool", "tool_call_id": _get(b, "tool_use_id"), "content": body})
            elif _get(b, "type") == "text":
                texts.append(_get(b, "text"))
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]


class OpenRouterClient:
    def __init__(self, settings: "Settings", client: openai.OpenAI | None = None):
        self.settings = settings
        if client is None:
            import os

            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise LLMError("Не задан OPENROUTER_API_KEY (см. .env.example)")
            client = openai.OpenAI(base_url=BASE_URL, api_key=key, max_retries=4, timeout=600)
        self.client = client
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "calls": 0, "cost_usd": 0.0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ low level
    def _call(self, **kwargs):
        if self.usage["cost_usd"] >= self.settings.max_cost_usd:
            raise LLMError(f"Достигнут лимит расходов ${self.settings.max_cost_usd:.2f} (CONTRACT_AGENT_MAX_COST_USD)")
        extra = kwargs.pop("extra_body", {})
        extra["usage"] = {"include": True}  # OpenRouter вернёт фактическую стоимость вызова
        try:
            resp = self.client.chat.completions.create(extra_body=extra, **kwargs)
        except openai.AuthenticationError as exc:
            raise LLMError("OpenRouter отклонил ключ: проверьте OPENROUTER_API_KEY") from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError(f"OpenRouter: нет доступа ({exc.message})") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenRouter: превышен лимит запросов (после повторных попыток)") from exc
        except openai.BadRequestError as exc:
            raise LLMError(f"OpenRouter: некорректный запрос: {exc.message}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 402:
                raise LLMError("OpenRouter: недостаточно средств на балансе/лимите ключа") from exc
            raise LLMError(f"OpenRouter: ошибка API {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Нет соединения с OpenRouter") from exc
        if not getattr(resp, "choices", None):
            err = getattr(resp, "error", None)
            raise LLMError(f"OpenRouter вернул пустой ответ{': ' + str(err) if err else ''}")
        self._account(resp)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("Ответ модели обрезан по max_tokens")
        if choice.finish_reason == "content_filter":
            raise LLMError("Модель отказалась обрабатывать запрос (content_filter)")
        return choice

    def _account(self, resp) -> None:
        u = resp.usage
        if u is None:
            return
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        cost = getattr(u, "cost", None)
        if cost is None and getattr(u, "model_extra", None):
            cost = u.model_extra.get("cost")
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += (u.prompt_tokens or 0) - cached
            self.usage["cache_read_input_tokens"] += cached
            self.usage["output_tokens"] += u.completion_tokens or 0
            self.usage["cost_usd"] = round(self.usage["cost_usd"] + float(cost or 0), 6)

    # ------------------------------------------------------------------ public API
    def agent_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Response:
        extra = {}
        if self.settings.openrouter_reasoning:
            extra["reasoning"] = {"effort": self.settings.agent_effort}
        choice = self._call(
            model=self.settings.model,
            max_tokens=self.settings.agent_max_tokens,
            messages=to_openai_messages(system, messages),
            tools=to_openai_tools(tools),
            extra_body=extra,
        )
        msg = choice.message
        blocks: list[Block] = []
        extra_fields = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning", None) or extra_fields.get("reasoning")
        details = getattr(msg, "reasoning_details", None) or extra_fields.get("reasoning_details")
        if reasoning or details:
            blocks.append(Block(type="thinking", thinking=reasoning or "", details=details))
        if msg.content:
            blocks.append(Block(type="text", text=msg.content))
        for call in msg.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": call.function.arguments}
            blocks.append(Block(type="tool_use", id=call.id, name=call.function.name, input=args))
        return Response(content=blocks, stop_reason="tool_use" if msg.tool_calls else "end_turn")

    def structured(self, system: str, content: list[dict] | str, schema: type[T], max_tokens: int | None = None) -> T:
        """Structured output через response_format=json_schema; ответ дополнительно валидируется pydantic.
        При невалидном ответе — одна попытка исправления с текстом ошибки."""
        user_content = content if isinstance(content, str) else [self._to_openai_part(p) for p in content]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        response_format = {"type": "json_schema",
                           "json_schema": {"name": schema.__name__, "strict": True, "schema": _strict_schema(schema)}}
        last_error = None
        for attempt in range(2):
            choice = self._call(model=self.settings.extraction_model, max_tokens=max_tokens or self.settings.extraction_max_tokens,
                                messages=messages, response_format=response_format)
            text = (choice.message.content or "").strip()
            try:
                return schema.model_validate_json(_strip_fences(text))
            except ValidationError as exc:
                last_error = exc
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Ответ не соответствует схеме ({exc.error_count()} ошибок): {str(exc)[:1500]}. Верни исправленный JSON целиком."}]
        raise LLMError(f"Ответ модели не прошёл валидацию схемы {schema.__name__}: {last_error.error_count()} ошибок")

    def ocr_pages(self, pages: list[tuple[int, bytes]], doc_path: str | None = None) -> OcrResult:
        content: list[dict] = []
        for num, png in pages:
            content.append({"type": "text", "text": f"Страница {num}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.standard_b64encode(png).decode()}})
        where = f" документа {doc_path}" if doc_path else ""
        content.append({"type": "text", "text": f"Расшифруй страницы {[n for n, _ in pages]}{where} по правилам."})
        return self.structured(OCR_SYSTEM, content, OcrResult)

    @staticmethod
    def _to_openai_part(part: dict) -> dict:
        if part.get("type") == "image":
            src = part["source"]
            return {"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}}
        return part


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text


def _strict_schema(schema: type[BaseModel]) -> dict:
    """JSON Schema из pydantic в «строгом» виде: все поля обязательны, additionalProperties=false,
    без неподдерживаемых ограничений. Используем тот же трансформер, что и Anthropic SDK."""
    from anthropic.lib._parse._transform import transform_schema

    return transform_schema(schema)
