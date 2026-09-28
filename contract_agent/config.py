"""Настройки агента. Всё читается из переменных окружения (и .env), чтобы не править код."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- LLM ---
    model: str = "claude-opus-5"             # модель, которая рассуждает и вызывает tools
    extraction_model: str = "claude-opus-5"  # модель для извлечения карточек и OCR
    agent_effort: str = "high"               # low | medium | high | xhigh | max
    extraction_effort: str = "medium"
    use_fallbacks: bool = True               # server-side fallbacks на случай refusal
    provider: str = "anthropic"              # anthropic | openrouter
    openrouter_reasoning: bool = True        # передавать reasoning (effort) в OpenRouter
    max_cost_usd: float = 12.0               # стоп-кран по фактической стоимости (OpenRouter сообщает cost)
    agent_max_tokens: int = 16000
    extraction_max_tokens: int = 32000

    # --- Пути ---
    input_dir: Path = Path("data")
    work_dir: Path = Path("workdir")

    # --- Пороговые значения проверок ---
    min_confidence: float = 0.7        # ниже — значение не записывается в базу знаний
    min_quote_match: float = 0.6       # доля совпавших 4-грамм цитаты с исходным текстом
    min_chars_per_page: int = 50       # меньше символов в текстовом слое -> страница считается сканом
    ocr_dpi: int = 150
    ocr_pages_per_request: int = 5
    max_doc_chars: int = 600_000       # защита от гигантских документов (явное предупреждение, не молча)
    as_of: str | None = None           # дата (ISO), на которую строится свертка; None — сегодня
    our_party: str | None = None       # «наша» сторона договоров: контрагентом считается другая сторона
    number_patterns: tuple[str, ...] = ()  # регулярные выражения номеров документов (пусто — «№ <номер>»)

    # --- Цикл агента ---
    max_agent_steps: int = 60
    max_reflection_rounds: int = 2
    max_parallel_tools: int = 6

    extra: dict = field(default_factory=dict)

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.work_dir / "logs"

    @property
    def kb_path(self) -> Path:
        return self.work_dir / "knowledge_base.json"

    @property
    def report_path(self) -> Path:
        return self.work_dir / "report.md"

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


OPENROUTER_DEFAULT_MODEL = "anthropic/claude-opus-5"


def _provider() -> str:
    explicit = os.getenv("CONTRACT_AGENT_PROVIDER")
    if explicit:
        return explicit.strip().lower()
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        return "openrouter"
    return "anthropic"


def load_settings(**overrides) -> Settings:
    provider = _provider()
    default_model = OPENROUTER_DEFAULT_MODEL if provider == "openrouter" else Settings.model
    model = os.getenv("CONTRACT_AGENT_MODEL", default_model)
    s = Settings(
        provider=provider,
        openrouter_reasoning=_env_bool("CONTRACT_AGENT_OPENROUTER_REASONING", True),
        max_cost_usd=float(os.getenv("CONTRACT_AGENT_MAX_COST_USD", Settings.max_cost_usd)),
        model=model,
        extraction_model=os.getenv("CONTRACT_AGENT_EXTRACTION_MODEL", model),
        agent_effort=os.getenv("CONTRACT_AGENT_EFFORT", Settings.agent_effort),
        extraction_effort=os.getenv("CONTRACT_AGENT_EXTRACTION_EFFORT", Settings.extraction_effort),
        use_fallbacks=_env_bool("CONTRACT_AGENT_FALLBACKS", Settings.use_fallbacks),
        input_dir=Path(os.getenv("CONTRACT_AGENT_INPUT", str(Settings.input_dir))),
        work_dir=Path(os.getenv("CONTRACT_AGENT_WORKDIR", str(Settings.work_dir))),
        min_confidence=float(os.getenv("CONTRACT_AGENT_MIN_CONFIDENCE", Settings.min_confidence)),
        as_of=os.getenv("CONTRACT_AGENT_AS_OF") or None,
        our_party=os.getenv("CONTRACT_AGENT_OUR_PARTY") or None,
        number_patterns=tuple(p for p in os.getenv("CONTRACT_AGENT_NUMBER_PATTERNS", "").split(";;") if p.strip()),
    )
    s = s.with_overrides(**overrides)
    from .validators import configure_number_patterns

    configure_number_patterns(s.number_patterns)
    return s
