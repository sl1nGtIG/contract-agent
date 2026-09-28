"""Выбор бэкенда LLM по настройкам и общий интерфейс бэкендов."""
from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import Settings
from .llm import LLMClient, LLMError
from .schemas import OcrResult

T = TypeVar("T", bound=BaseModel)


class LLMBackend(Protocol):
    """Всё, что остальной код знает о модели. Реализации: LLMClient (Anthropic), OpenRouterClient,
    ReplayLLM (демо-кассета) и сценарные модели в тестах."""

    settings: Settings
    usage: dict

    def agent_turn(self, system: str, messages: list[dict], tools: list[dict]) -> Any:
        """Один шаг агента: ответ с блоками text / thinking / tool_use и stop_reason."""

    def structured(self, system: str, content: list[dict] | str, schema: type[T], max_tokens: int | None = None) -> T:
        """Ответ строго по JSON-схеме pydantic-модели."""

    def ocr_pages(self, pages: list[tuple[int, bytes]], doc_path: str | None = None) -> OcrResult:
        """Расшифровка страниц-изображений."""


def make_llm(settings: Settings) -> LLMBackend:
    if settings.provider == "anthropic":
        return LLMClient(settings)
    if settings.provider == "openrouter":
        from .llm_openrouter import OpenRouterClient

        return OpenRouterClient(settings)
    raise LLMError(f"Неизвестный провайдер {settings.provider!r}: допустимо anthropic | openrouter")
