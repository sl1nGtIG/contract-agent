"""Извлечение карточки документа: LLM (structured output) + детерминированная проверка."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .prompts import EXTRACTION_SYSTEM
from .schemas import DocumentCard, DocumentExtraction
from .validators import build_card

if TYPE_CHECKING:
    from .config import Settings
    from .ingest import DocumentInfo, ParsedDocument
    from .llm import LLMClient


def build_extraction_prompt(info: "DocumentInfo", parsed: "ParsedDocument", max_chars: int,
                            our_party: str | None = None) -> str:
    text = parsed.text
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    header = [
        f"Файл: {info.path}",
        "(имя файла и папки — подсказка, а не источник фактов: реквизиты бери из текста)",
        f"Источник текста: {parsed.text_source}; страниц: {len(parsed.pages)}",
    ]
    if our_party:
        header.append(f"Наша сторона (контрагент — другая сторона договора): {our_party}")
    uncertain = [f for p in parsed.pages for f in p.uncertain_fragments]
    if uncertain:
        header.append("Фрагменты, неуверенно распознанные OCR: " + "; ".join(uncertain[:30]))
    if truncated:
        header.append(f"ВНИМАНИЕ: документ обрезан до первых {max_chars} символов; учти это в extraction_notes.")
    return "\n".join(header) + "\n\n<document>\n" + text + "\n</document>\n\nИзвлеки данные по схеме."


def extract_card(info: "DocumentInfo", parsed: "ParsedDocument", llm: "LLMClient", settings: "Settings") -> DocumentCard:
    prompt = build_extraction_prompt(info, parsed, settings.max_doc_chars, settings.our_party)
    extraction = llm.structured(EXTRACTION_SYSTEM, prompt, DocumentExtraction)
    return build_card(info, parsed, extraction, settings.min_confidence, settings.min_quote_match,
                      our_party=settings.our_party)
