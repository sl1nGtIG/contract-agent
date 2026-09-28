"""Тонкая обёртка над Anthropic SDK: один вход для всех вызовов модели.

Зачем обёртка, а не прямые вызовы:
* единая обработка stop_reason (refusal / max_tokens) и ошибок API;
* server-side fallbacks включаются в одном месте;
* подсчёт токенов для отчёта о стоимости;
* подмена на ScriptedLLM в тестах (тот же интерфейс).
"""
from __future__ import annotations

import base64
import threading
from typing import TYPE_CHECKING, Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from .prompts import OCR_SYSTEM
from .schemas import OcrResult

if TYPE_CHECKING:
    from .config import Settings

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """Ошибка вызова модели, которую агент должен увидеть как результат инструмента."""


class LLMClient:
    def __init__(self, settings: "Settings", client: anthropic.Anthropic | None = None):
        self.settings = settings
        try:
            self.client = client or anthropic.Anthropic(max_retries=4)
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Не удалось создать клиент Anthropic: {exc}. Задайте ANTHROPIC_API_KEY") from exc
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "calls": 0}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ low level
    def _stream(self, **kwargs: Any):
        if self.settings.use_fallbacks:
            kwargs.setdefault("betas", []).append(FALLBACK_BETA)
            kwargs["fallbacks"] = "default"
        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("Нет доступа к Anthropic API: проверьте ANTHROPIC_API_KEY") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Некорректный запрос к модели: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Превышен лимит запросов к API (после повторных попыток)") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Ошибка API {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Нет соединения с Anthropic API") from exc

        self._account(message)
        if message.stop_reason == "refusal":
            raise LLMError("Модель отказалась обрабатывать запрос (stop_reason=refusal)")
        if message.stop_reason == "max_tokens":
            raise LLMError("Ответ модели обрезан по max_tokens")
        return message

    def _account(self, message) -> None:
        u = message.usage
        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
            self.usage["output_tokens"] += u.output_tokens or 0
            self.usage["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0

    # ------------------------------------------------------------------ public API
    def agent_turn(self, system: str, messages: list[dict], tools: list[dict]):
        """Один шаг агента: модель думает (adaptive thinking) и либо вызывает tools, либо отвечает."""
        return self._stream(
            model=self.settings.model,
            max_tokens=self.settings.agent_max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            cache_control={"type": "ephemeral"},  # история диалога растёт — кешируем префикс между шагами
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.settings.agent_effort},
        )

    def structured(self, system: str, content: list[dict] | str, schema: type[T], max_tokens: int | None = None) -> T:
        """Structured output: ответ модели гарантированно соответствует JSON-схеме ``schema``."""
        message = self._stream(
            model=self.settings.extraction_model,
            max_tokens=max_tokens or self.settings.extraction_max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
            output_config={"effort": self.settings.extraction_effort},
        )
        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            raise LLMError("Модель вернула пустой ответ")
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"Ответ модели не прошёл валидацию схемы {schema.__name__}: {exc.error_count()} ошибок") from exc

    def ocr_pages(self, pages: list[tuple[int, bytes]], doc_path: str | None = None) -> OcrResult:
        content: list[dict] = []
        for num, png in pages:
            content.append({"type": "text", "text": f"Страница {num}:"})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode()},
                }
            )
        where = f" документа {doc_path}" if doc_path else ""
        content.append({"type": "text", "text": f"Расшифруй страницы {[n for n, _ in pages]}{where} по правилам."})
        return self.structured(OCR_SYSTEM, content, OcrResult)
