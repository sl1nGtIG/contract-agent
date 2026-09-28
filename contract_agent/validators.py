"""Детерминированные проверки извлечённых значений.

Принцип: LLM предлагает значение + дословную цитату + уверенность, а код проверяет
его по исходному тексту. Значение попадает в базу знаний только если:
  1) цитата действительно есть в тексте документа (нечёткое сравнение по 4-граммам слов);
  2) значение связано с цитатой: номер и дата встречаются в тексте, слова названия контрагента —
     в цитате, каждое число и дата из описания условия/изменения — в тексте рядом с цитатой;
  3) уверенность не ниже порога;
  4) не попадает в неуверенно распознанный OCR-фрагмент.
Иначе значение НЕ записывается (value=None), а в карточке остаётся ValidationIssue
с severity=rejected, исходным значением и причиной отказа.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import PurePosixPath

from .ingest import DocumentInfo, ParsedDocument
from .schemas import (
    Change,
    DocType,
    DocumentCard,
    DocumentExtraction,
    EvidencedValue,
    KeyTerm,
    LegalForm,
    RelatedContract,
    Severity,
    ValidationIssue,
    VerifiedValue,
)

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_MONTH_RE = r"(январ[ья]|феврал[ья]|марта?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|августа?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])"
NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s?\.\s?(\d{1,2})\s?\.\s?(\d{4})(?!\d)")
VERBAL_DATE_RE = re.compile(r"[«\"]?\s?(\d{1,2})\s?[»\"]?\s+" + _MONTH_RE + r"\s+(\d{4})", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
PROLONGATION_RE = re.compile(r"(автоматически\s+(продлевается|пролонгируется)|пролонгиру\w*\s+на\s+каждый|продлевается\s+на\s+каждый)", re.IGNORECASE)
# Шаблоны номеров для поиска в именах файлов/папок и в тексте. По умолчанию — «№ <номер>»;
# свои шаблоны (например, корпоративный формат) задаются через CONTRACT_AGENT_NUMBER_PATTERNS.
DEFAULT_NUMBER_PATTERNS = (r"№\s*([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9/._\-–]*\d[A-Za-zА-Яа-яЁё0-9/._\-–]*)",)
_number_patterns: list[re.Pattern] = [re.compile(p) for p in DEFAULT_NUMBER_PATTERNS]


def configure_number_patterns(patterns: list[str] | tuple[str, ...]) -> None:
    """Задать шаблоны номеров (группа 1 — сам номер; без группы — всё совпадение)."""
    global _number_patterns
    _number_patterns = [re.compile(p) for p in (patterns or DEFAULT_NUMBER_PATTERNS)]


# ---------------------------------------------------------------------------
# Нормализация
# ---------------------------------------------------------------------------
_DASHES = str.maketrans({"–": "-", "—": "-", "_": "-", "\u2011": "-"})


def normalize_number(raw: str | None) -> str | None:
    """Номер документа в каноническом виде: без «№», пробелов и вариантов тире, в верхнем регистре.
    «№ АР-2021/015», «ар–2021/015» → «АР-2021/015»; «ab 123_45» → «AB123-45»."""
    if not raw:
        return None
    s = re.sub(r"^\s*(№|N°|No\.?)\s*", "", raw.strip())
    s = re.sub(r"\s+", "", s).translate(_DASHES).strip(".,;:«»\"'()").upper()
    return s or None


def _compact(text: str) -> str:
    """Текст в той же нормализации, что и номера: для проверки «номер встречается в тексте»."""
    return re.sub(r"\s+", "", text).translate(_DASHES).upper()


def _month_num(word: str) -> int | None:
    w = word.lower()
    for stem, num in MONTHS.items():
        if w.startswith(stem) and (stem != "ма" or w in {"мая", "май"}):
            return num
    return None


def parse_date(raw: str | None) -> date | None:
    """Понимает ДД.ММ.ГГГГ, «ДД» месяца ГГГГ и ISO. Возвращает None, если не дата."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        if m := ISO_DATE_RE.match(raw):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m := NUMERIC_DATE_RE.search(raw):
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if m := VERBAL_DATE_RE.search(raw):
            month = _month_num(m.group(2))
            if month:
                return date(int(m.group(3)), month, int(m.group(1)))
    except ValueError:
        return None
    return None


def dates_in_text(text: str) -> set[date]:
    found: set[date] = set()
    for m in NUMERIC_DATE_RE.finditer(text):
        try:
            found.add(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    for m in VERBAL_DATE_RE.finditer(text):
        month = _month_num(m.group(2))
        if month:
            try:
                found.add(date(int(m.group(3)), month, int(m.group(1))))
            except ValueError:
                pass
    return found


def numbers_in_text(text: str) -> set[str]:
    out: set[str] = set()
    for pattern in _number_patterns:
        for m in pattern.finditer(text):
            n = normalize_number(m.group(1) if m.groups() else m.group(0))
            if n:
                out.add(n)
    return out


_TOKEN_RE = re.compile(r"[0-9a-zа-я]+", re.IGNORECASE)

# Ссылки на пункты/приложения и номера документов в описании условия — это не «значения», их не сверяем
_REF_RE = re.compile(
    r"(п\.|пп\.|пункт\w*|раздел\w*|приложени\w*|табл\w*\.?|стать\w*|ст\.|глав\w*)\s*№?\s*\d+(?:\.\d+)*"
    r"(?:\s*[–—-]\s*\d+(?:\.\d+)*)?|№\s*\S+", re.IGNORECASE)
# \u0447\u0438\u0441\u043b\u043e \u2014 \u043e\u0442\u0434\u0435\u043b\u044c\u043d\u044b\u0439 \u0442\u043e\u043a\u0435\u043d: \u0446\u0438\u0444\u0440\u044b \u0432\u043d\u0443\u0442\u0440\u0438 \u043a\u043e\u0434\u043e\u0432 \u0438 \u043d\u043e\u043c\u0435\u0440\u043e\u0432 (\u00ab\u0414\u04215\u00bb, \u00ab2022/045\u00bb, \u00abA-12\u00bb) \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f\u043c\u0438 \u043d\u0435 \u0441\u0447\u0438\u0442\u0430\u044e\u0442\u0441\u044f
_NUM_RE = re.compile(r"(?<![\w/\-])(?:\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\w/\-]|[.,]\d)")


def _numbers(text: str) -> set[str]:
    """Числа без дат: «450 000» → 450000, «0,05» → 0.05. Даты извлекаются отдельно."""
    text = NUMERIC_DATE_RE.sub(" ", VERBAL_DATE_RE.sub(" ", text))
    out = set()
    for m in _NUM_RE.finditer(text):
        v = re.sub(r"[ \u00a0]", "", m.group(0)).replace(",", ".")
        v = v.rstrip("0").rstrip(".") if "." in v else v.lstrip("0") or "0"
        out.add(v)
    return out


def facts_of(summary: str, known_numbers: set[str] = frozenset()) -> tuple[set[str], set[date]]:
    """Проверяемые факты описания: числа и даты (без ссылок на пункты и номеров документов)."""
    cleaned = _REF_RE.sub(" ", summary)
    for n in known_numbers:
        cleaned = cleaned.replace(n, " ")
    return _numbers(cleaned), dates_in_text(cleaned)


def same_party(a: str, b: str) -> bool:
    """Одна ли это организация: совпадают значимые слова названия (без организационно-правовой формы)."""
    words = lambda s: {t for t in _tokens(s) if len(t) >= 3 and t not in LEGAL_WORDS}  # noqa: E731
    wa, wb = words(a), words(b)
    return bool(wa) and wa == wb


LEGAL_WORDS = {"ооо", "ао", "пао", "зао", "оао", "нао", "ип", "общество", "обществом", "общества", "ограниченной",
               "ответственностью", "акционерное", "акционерным", "публичное", "непубличное", "закрытое", "открытое",
               "индивидуальный", "предприниматель", "компания", "фирма", "группа", "llc", "ltd", "jsc"}


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower().replace("ё", "е"))


def _shingles(tokens: list[str], n: int = 4) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}


class TextIndex:
    """Индекс исходного текста для быстрой проверки цитат, дат и номеров."""

    def __init__(self, text: str):
        self.text = text
        # «ё» → «е» ДО разбиения на слова, как и в _tokens(): иначе «платёж» в тексте рвётся на «плат» + «ж»
        # и цитаты с «ё» ложно не находятся (замена посимвольная — позиции слов не сдвигаются)
        norm = text.replace("ё", "е").replace("Ё", "Е")
        spans = [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(norm)]
        self.tokens = [t for t, _, _ in spans]
        self.spans = [(a, b) for _, a, b in spans]
        self.joined = " ".join(self.tokens)
        self.shingles = _shingles(self.tokens)
        self.first_pos: dict[tuple, int] = {}
        for i in range(max(0, len(self.tokens) - 3)):
            self.first_pos.setdefault(tuple(self.tokens[i : i + 4]), i)
        self.dates = dates_in_text(text)
        self.compact = _compact(text)

    def has_number(self, number: str) -> bool:
        return bool(number) and _compact(number) in self.compact

    def window(self, quote: str | None, radius: int = 2500) -> str:
        """Фрагмент текста вокруг цитаты (±radius символов). Если цитату не удалось привязать — весь текст."""
        qt = _tokens(quote or "")
        positions = [self.first_pos[s] for s in (tuple(qt[i : i + 4]) for i in range(max(0, len(qt) - 3))) if s in self.first_pos]
        if not positions:
            return self.text
        start = self.spans[min(positions)][0]
        end = self.spans[min(max(positions) + 3, len(self.spans) - 1)][1]
        return self.text[max(0, start - radius) : end + radius]

    def quote_match(self, quote: str | None) -> float:
        """Доля 4-грамм цитаты, найденных в тексте (0..1). Короткие цитаты — точное вхождение."""
        if not quote:
            return 0.0
        qt = _tokens(quote)
        if not qt:
            return 0.0
        if len(qt) < 4:
            return 1.0 if " ".join(qt) in self.joined else 0.0
        qs = _shingles(qt)
        return len(qs & self.shingles) / len(qs)


# ---------------------------------------------------------------------------
# Проверки
# ---------------------------------------------------------------------------
class CardValidator:
    def __init__(self, parsed: ParsedDocument, min_confidence: float, min_quote_match: float, today: date | None = None,
                 ocr_penalty: bool = True):
        self.parsed = parsed
        self.apply_ocr_penalty = ocr_penalty
        self.index = TextIndex(parsed.text)
        self.min_conf = min_confidence
        self.min_match = min_quote_match
        self.today = today or date.today()
        self.issues: list[ValidationIssue] = []
        self.uncertain = [f for p in parsed.pages for f in p.uncertain_fragments if f.strip()]
        self.is_ocr = parsed.text_source in {"ocr", "mixed"}

    # -- helpers ---------------------------------------------------------------
    def _issue(self, field: str, severity: Severity, reason: str, raw=None) -> None:
        self.issues.append(ValidationIssue(field=field, severity=severity, reason=reason, raw_value=raw, doc_id=self.parsed.doc_id))

    def _ocr_penalty(self, *texts: str | None) -> tuple[float, str | None]:
        """Если значение/цитата пересекаются с неуверенно распознанным OCR-фрагментом — штраф к уверенности."""
        if not self.is_ocr:
            return 1.0, None
        hay = " ".join(_tokens(" ".join(t for t in texts if t)))
        hay_tokens = set(hay.split())
        for frag in self.uncertain:
            ft = " ".join(_tokens(frag))
            significant = [t for t in ft.split() if len(t) > 3 or (t.isdigit() and len(t) >= 2)]
            hits = sum(t in hay_tokens for t in significant)
            # фрагмент считается задействованным, если он входит целиком или совпадает большая часть его значимых слов
            if ft and (ft in hay or (hits >= 2 and hits / len(significant) >= 0.6)):
                return 0.6, frag
        if "неразборчиво" in hay:
            return 0.5, "[неразборчиво]"
        return 1.0, None

    def _base_checks(self, field: str, ev: EvidencedValue) -> float | None:
        """Общие проверки: цитата в тексте, уверенность, OCR. Возвращает итоговую уверенность или None (отказ)."""
        match = self.index.quote_match(ev.quote)
        if ev.quote and match < self.min_match:
            self._issue(field, Severity.REJECTED, f"цитата не найдена в тексте документа (совпадение {match:.0%})", ev.value)
            return None
        penalty, frag = self._ocr_penalty(ev.value, ev.quote) if self.apply_ocr_penalty else (1.0, None)
        conf = round(ev.confidence * penalty, 2)
        if frag:
            self._issue(field, Severity.WARNING, f"значение пересекается с неуверенно распознанным фрагментом OCR: «{frag}»", ev.value)
        if conf < self.min_conf:
            self._issue(field, Severity.REJECTED, f"низкая уверенность распознавания ({conf} < {self.min_conf})", ev.value)
            return None
        return conf

    # -- typed checks ----------------------------------------------------------
    def check_number(self, field: str, ev: EvidencedValue) -> VerifiedValue:
        if not ev.value:
            return VerifiedValue()
        conf = self._base_checks(field, ev)
        if conf is None:
            return VerifiedValue()
        num = normalize_number(ev.value)
        if not self.index.has_number(num):
            self._issue(field, Severity.REJECTED, f"номер {num} не встречается в тексте документа", ev.value)
            return VerifiedValue()
        return VerifiedValue(value=num, quote=ev.quote, confidence=conf)

    def check_facts(self, field: str, summary: str, quote: str | None, raw) -> bool:
        """Каждое число и дата из описания должны встречаться в тексте рядом с цитатой (±2500 символов).
        Так нельзя приписать цитате чужое значение: «вознаграждение 95%» при цитате из преамбулы не пройдёт."""
        numbers, dates = facts_of(summary)
        if not numbers and not dates:
            return True
        window_numbers, window_dates = facts_of(self.index.window(quote))  # в окне тоже не считаем ссылки и номера
        missing = sorted(numbers - window_numbers) + [f"{d:%d.%m.%Y}" for d in sorted(dates - window_dates)]
        if missing:
            self._issue(field, Severity.REJECTED,
                        f"в описании есть значения, которых нет в тексте рядом с цитатой: {', '.join(missing[:5])}", raw)
            return False
        return True

    def check_date(self, field: str, ev: EvidencedValue, allow_phrase: tuple[str, ...] = (),
                   phrase_date: str | None = None) -> VerifiedValue:
        if not ev.value:
            return VerifiedValue()
        parsed = parse_date(ev.value)
        if parsed is None:
            self._issue(field, Severity.REJECTED, "значение не распознаётся как дата", ev.value)
            return VerifiedValue()
        if not (1990 <= parsed.year <= self.today.year + 2):
            self._issue(field, Severity.REJECTED, f"неправдоподобный год: {parsed.year}", ev.value)
            return VerifiedValue()
        conf = self._base_checks(field, ev)
        if conf is None:
            return VerifiedValue()
        in_text = parsed in self.index.dates
        # «вступает в силу с момента подписания»: даты в тексте может не быть, но тогда она обязана
        # совпасть с датой подписания самого документа
        phrase = bool(ev.quote) and any(p in (ev.quote or "").lower() for p in allow_phrase)
        if not in_text and not (phrase and phrase_date == parsed.isoformat()):
            reason = (f"дата {parsed:%d.%m.%Y} выведена из «с момента подписания», но не совпадает с датой документа "
                      f"({phrase_date or 'не установлена'})" if phrase else f"дата {parsed:%d.%m.%Y} не встречается в тексте документа")
            self._issue(field, Severity.REJECTED, reason, ev.value)
            return VerifiedValue()
        return VerifiedValue(value=parsed.isoformat(), quote=ev.quote, confidence=conf)

    def check_party(self, field: str, ev: EvidencedValue) -> VerifiedValue:
        """Наименование стороны: значимые слова названия (без «ООО», «общество» и т.п.) должны быть в цитате."""
        if not ev.value:
            return VerifiedValue()
        if not ev.quote:
            self._issue(field, Severity.REJECTED, "нет подтверждающей цитаты", ev.value)
            return VerifiedValue()
        conf = self._base_checks(field, ev)
        if conf is None:
            return VerifiedValue()
        words = [t for t in _tokens(ev.value) if len(t) >= 3 and t not in LEGAL_WORDS] or _tokens(ev.value)
        quote_tokens = _tokens(ev.quote)
        # сравнение по основе слова: «Ромашка» / «Ромашки» / «Ромашкой»
        absent = [w for w in words if not any(q[:5] == w[:5] and abs(len(q) - len(w)) <= 3 for q in quote_tokens)]
        if absent:
            self._issue(field, Severity.REJECTED, f"слова названия отсутствуют в цитате: {', '.join(absent)}", ev.value)
            return VerifiedValue()
        return VerifiedValue(value=ev.value.strip(), quote=ev.quote, confidence=conf)

    def check_quoted_item(self, field: str, quote: str | None, confidence: float, raw) -> float | None:
        if not quote:
            self._issue(field, Severity.REJECTED, "нет подтверждающей цитаты", raw)
            return None
        return self._base_checks(field, EvidencedValue(value=str(raw), quote=quote, confidence=confidence))


def _iso(raw: str | None) -> str | None:
    d = parse_date(raw)
    return d.isoformat() if d else None


def build_card(
    info: DocumentInfo,
    parsed: ParsedDocument,
    ex: DocumentExtraction,
    min_confidence: float,
    min_quote_match: float,
    today: date | None = None,
    ocr_penalty: bool = True,
    our_party: str | None = None,
) -> DocumentCard:
    """Превращает сырой ответ LLM в карточку, записывая только проверенные значения."""
    v = CardValidator(parsed, min_confidence, min_quote_match, today, ocr_penalty)

    number = v.check_number("number", ex.number)
    doc_date = v.check_date("date", ex.date)
    parent_number = v.check_number("parent_number", ex.parent_number)
    parent_date = v.check_date("parent_date", ex.parent_date)
    counterparty = v.check_party("counterparty", ex.counterparty)
    legal_form = ex.counterparty_legal_form
    if counterparty.value and our_party and same_party(counterparty.value, our_party):
        # в цитате действительно есть название, но это «наша» сторона, а не контрагент
        v._issue("counterparty", Severity.REJECTED, f"указана «наша» сторона ({our_party}), а не контрагент", counterparty.value)
        v._issue("counterparty_legal_form", Severity.REJECTED, "форма относится к «нашей» стороне, а не к контрагенту",
                 legal_form.value)
        counterparty, legal_form = VerifiedValue(), LegalForm.UNKNOWN
    effective_from = v.check_date("effective_from", ex.effective_from, allow_phrase=("подписан",), phrase_date=doc_date.value)
    valid_until = v.check_date("valid_until", ex.valid_until)
    if valid_until.value and effective_from.value and valid_until.value < effective_from.value:
        v._issue("valid_until", Severity.REJECTED, "дата окончания раньше даты начала действия", valid_until.value)
        valid_until = VerifiedValue()
    if valid_until.value and PROLONGATION_RE.search(parsed.text):
        # «действует до 31.12.2022, автоматически продлевается на каждый год» — это не дата прекращения
        v._issue("valid_until", Severity.REJECTED,
                 "в документе есть условие об автоматической пролонгации — дата окончания не окончательная", valid_until.value)
        valid_until = VerifiedValue()

    # внутренние противоречия
    if number.value and parent_number.value and number.value == parent_number.value and ex.doc_type == DocType.SUPPLEMENTARY:
        v._issue("parent_number", Severity.REJECTED, "номер родительского договора совпадает с номером самого соглашения", parent_number.value)
        parent_number = VerifiedValue()
    if ex.doc_type == DocType.SUPPLEMENTARY and not parent_number.value:
        v._issue("parent_number", Severity.WARNING, "доп. соглашение без подтверждённой ссылки на основной договор")
    if doc_date.value and effective_from.value and effective_from.value < doc_date.value:
        v._issue("effective_from", Severity.INFO, f"действие распространено на период до подписания (с {effective_from.value})")

    key_terms: list[KeyTerm] = []
    for i, kt in enumerate(ex.key_terms):
        field = f"key_terms[{i}].{kt.term.value}"
        conf = v.check_quoted_item(field, kt.quote, kt.confidence, kt.summary)
        if conf is not None and v.check_facts(field, kt.summary, kt.quote, kt.summary):
            key_terms.append(KeyTerm(term=kt.term, summary=kt.summary, clause_ref=kt.clause_ref, quote=kt.quote, confidence=conf))

    changes: list[Change] = []
    for i, ch in enumerate(ex.changes):
        field = f"changes[{i}] {ch.target_clause}"
        conf = v.check_quoted_item(field, ch.quote, ch.confidence, ch.new_value_summary)
        if conf is None or not v.check_facts(field, ch.new_value_summary, ch.quote, ch.new_value_summary):
            continue
        target = normalize_number(ch.target_document)
        if target and not v.index.has_number(target):
            v._issue(field, Severity.REJECTED,
                     f"изменяемый документ {target} не упомянут в тексте — изменение не записано", ch.new_value_summary)
            continue
        change_from = _iso(ch.effective_from)
        if change_from and parse_date(change_from) not in v.index.dates and change_from != effective_from.value:
            v._issue(f"{field}.effective_from", Severity.REJECTED,
                     f"дата начала действия изменения {change_from} не встречается в тексте — взята дата документа", change_from)
            change_from = None
        changes.append(
            Change(
                target_document=target,
                target_clause=ch.target_clause,
                action=ch.action,
                affected_terms=ch.affected_terms,
                new_value_summary=ch.new_value_summary,
                effective_from=change_from or effective_from.value,
                quote=ch.quote,
                confidence=conf,
            )
        )

    def _related(items, field) -> list[RelatedContract]:
        out = []
        for r in items:
            num = normalize_number(r.number)
            if num and not v.index.has_number(num):
                v._issue(field, Severity.REJECTED, f"документ {num} не упомянут в тексте", r.number)
                continue
            out.append(RelatedContract(number=num or r.number, date=_iso(r.date), title=r.title, relation=r.relation))
        return out

    invalidates = _related(ex.invalidates, "invalidates")
    related = _related(ex.related_contracts, "related_contracts")

    # имя файла vs содержимое: только предупреждение, имя файла не авторитетно
    file_numbers = numbers_in_text(PurePosixPath(info.file).stem)
    known = {n for n in (number.value, parent_number.value) if n}
    if file_numbers and known and not (file_numbers & known):
        v._issue("file", Severity.WARNING, f"номер в имени файла {sorted(file_numbers)} не совпадает с номерами в документе {sorted(known)}")
    if "проект" in info.file.lower():
        v._issue("file", Severity.WARNING, "файл назван «Проект»: подписанная редакция может отличаться")

    for w in parsed.warnings:
        v._issue("parsing", Severity.WARNING, w)

    rejected = [i for i in v.issues if i.severity == Severity.REJECTED]
    if not number.value and not doc_date.value and not key_terms and not changes:
        status = "unrecognized"
    elif rejected:
        status = "partial"
    else:
        status = "ok"

    return DocumentCard(
        doc_id=info.doc_id,
        file=info.file,
        folder=info.folder,
        text_source=parsed.text_source,
        doc_type=ex.doc_type,
        title=ex.title,
        number=number,
        date=doc_date,
        city=ex.city,
        parent_number=parent_number,
        parent_date=parent_date,
        parent_title=ex.parent_title,
        counterparty=counterparty,
        counterparty_legal_form=legal_form,
        signatory_branch=ex.signatory_branch,
        subject_category=ex.subject_category,
        subject_summary=ex.subject_summary,
        effective_from=effective_from,
        valid_until=valid_until,
        key_terms=key_terms,
        changes=changes,
        invalidates=invalidates,
        related_contracts=related,
        anonymized_fields=ex.anonymized_fields,
        extraction_notes=ex.extraction_notes,
        issues=v.issues,
        status=status,
    )


def card_to_extraction(card: DocumentCard) -> DocumentExtraction:
    """Обратное преобразование: сохранённая карточка → «ответ модели» (значение + цитата + уверенность)."""
    from .schemas import ChangeExtraction, KeyTermExtraction, RelatedContractExtraction

    ev = lambda v: EvidencedValue(value=v.value, quote=v.quote, confidence=v.confidence)  # noqa: E731
    rel = lambda r: RelatedContractExtraction(number=r.number, date=r.date, title=r.title, relation=r.relation)  # noqa: E731
    return DocumentExtraction(
        doc_type=card.doc_type, title=card.title, number=ev(card.number), date=ev(card.date), city=card.city,
        parent_number=ev(card.parent_number), parent_date=ev(card.parent_date), parent_title=card.parent_title,
        counterparty=ev(card.counterparty), counterparty_legal_form=card.counterparty_legal_form, signatory_branch=card.signatory_branch,
        subject_category=card.subject_category, subject_summary=card.subject_summary,
        effective_from=ev(card.effective_from), valid_until=ev(card.valid_until),
        key_terms=[KeyTermExtraction(term=k.term, summary=k.summary, clause_ref=k.clause_ref, quote=k.quote,
                                     confidence=k.confidence) for k in card.key_terms],
        changes=[ChangeExtraction(target_document=c.target_document, target_clause=c.target_clause, action=c.action,
                                  affected_terms=c.affected_terms, new_value_summary=c.new_value_summary,
                                  effective_from=c.effective_from, quote=c.quote, confidence=c.confidence) for c in card.changes],
        invalidates=[rel(r) for r in card.invalidates], related_contracts=[rel(r) for r in card.related_contracts],
        anonymized_fields=card.anonymized_fields, extraction_notes=card.extraction_notes,
    )


def revalidate_card(card: DocumentCard, info: DocumentInfo, parsed: ParsedDocument,
                    min_confidence: float, min_quote_match: float, our_party: str | None = None) -> DocumentCard:
    """Повторная проверка сохранённой карточки текущими правилами (без обращения к модели).
    Штраф OCR второй раз не применяется; ранее отклонённые значения остаются в списке отказов."""
    fresh = build_card(info, parsed, card_to_extraction(card), min_confidence, min_quote_match, ocr_penalty=False,
                       our_party=our_party)
    known = {(i.field, i.reason) for i in fresh.issues}
    fresh.issues += [i for i in card.issues if i.severity == Severity.REJECTED and (i.field, i.reason) not in known]
    if fresh.rejected and fresh.status == "ok":
        fresh.status = "partial"
    return fresh
