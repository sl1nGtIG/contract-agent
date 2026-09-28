"""Детерминированная аналитика над карточками: цепочки, свертка, кластеризация, сравнение.

Здесь нет LLM: всё, что можно посчитать кодом, считается кодом — это воспроизводимо
и проверяемо тестами. LLM-агент решает, КОГДА и С КАКИМИ параметрами вызывать эти
функции, и интерпретирует результат.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from typing import Iterable

from .schemas import (
    ChangeAction,
    Cluster,
    ConsolidatedTerms,
    Contract,
    DocType,
    DocumentCard,
    LegalForm,
    Severity,
    SubjectCategory,
    TermKey,
    TermState,
    TermVersion,
    TimelineEntry,
    ValidationIssue,
    VerifiedValue,
)
from .validators import numbers_in_text

TITLE_CATEGORY_HINTS: list[tuple[str, SubjectCategory]] = [
    ("коммерческого представительства", SubjectCategory.COMMERCIAL_REPRESENTATION),
    ("агентск", SubjectCategory.COMMERCIAL_REPRESENTATION),
    ("партнерск", SubjectCategory.JOINT_PROMOTION),
    ("партнёрск", SubjectCategory.JOINT_PROMOTION),
    ("аренд", SubjectCategory.LEASE),
    ("купли-продажи", SubjectCategory.SALE),
    ("поставк", SubjectCategory.SALE),
    ("оказания услуг", SubjectCategory.SERVICES),
    ("возмездного оказания", SubjectCategory.SERVICES),
]

CATEGORY_RU = {
    SubjectCategory.COMMERCIAL_REPRESENTATION: "Коммерческое представительство",
    SubjectCategory.JOINT_PROMOTION: "Партнёрство / совместное продвижение",
    SubjectCategory.AGENCY_ORDER: "Поручение",
    SubjectCategory.INCENTIVE_PROGRAM: "Мотивация / планы продаж",
    SubjectCategory.SERVICES: "Оказание услуг",
    SubjectCategory.LEASE: "Аренда",
    SubjectCategory.SALE: "Купля-продажа / поставка",
    SubjectCategory.OTHER: "Прочее",
}

TERM_RU = {
    TermKey.SUBJECT: "Предмет",
    TermKey.TERM: "Срок действия",
    TermKey.REMUNERATION: "Вознаграждение / цена",
    TermKey.PAYMENT_PROCEDURE: "Порядок расчётов",
    TermKey.SALES_PLANS: "Плановые показатели",
    TermKey.OBLIGATIONS: "Обязанности сторон",
    TermKey.LIABILITY: "Ответственность",
    TermKey.TERMINATION: "Прекращение / расторжение",
    TermKey.TERRITORY: "Территория / точки продаж",
    TermKey.REPORTING: "Отчётность",
    TermKey.DEFINITIONS: "Определения",
    TermKey.CONTACTS: "Реквизиты / уведомления",
    TermKey.PERSONAL_DATA: "Персональные данные",
    TermKey.OTHER: "Прочее",
}


def normalize_city(city: str | None) -> str | None:
    """«г. Омск», «город Омск», «Омск» → «Омск» (иначе кластеры по городу дробятся)."""
    if not city:
        return None
    city = re.sub(r"^\s*(г\.|г\s|город\s)\s*", "", city.strip(), flags=re.IGNORECASE).strip(" .,")
    return city or None


def category_from_title(title: str | None) -> SubjectCategory | None:
    t = (title or "").lower()
    for hint, cat in TITLE_CATEGORY_HINTS:
        if hint in t:
            return cat
    return None


def _sort_key(card: DocumentCard) -> tuple[int, str, str]:
    """Хронология: основной документ первым, далее по дате подписания, иначе — по дате начала действия."""
    first = 0 if card.doc_type in (DocType.MAIN_CONTRACT, DocType.STANDALONE_AGREEMENT) else 1
    change_dates = [c.effective_from for c in card.changes if c.effective_from]
    when = card.date.value or card.effective_from.value or (min(change_dates) if change_dates else None) or "9999-99-99"
    return (first, when, card.number.value or card.doc_id)


def _issue(field: str, severity: Severity, reason: str, raw=None, doc_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(field=field, severity=severity, reason=reason, raw_value=raw, doc_id=doc_id)


# ---------------------------------------------------------------------------
# 1. Связывание документов в цепочки «договор → доп. соглашения»
# ---------------------------------------------------------------------------
def link_chains(cards: Iterable[DocumentCard]) -> tuple[dict[str, Contract], list[ValidationIssue]]:
    cards = [c for c in cards if c.status != "unrecognized"]
    global_issues: list[ValidationIssue] = []

    by_number: dict[str, DocumentCard] = {}
    for c in cards:
        if not c.number.value:
            continue
        if c.number.value in by_number:
            global_issues.append(
                _issue("number", Severity.WARNING, f"номер {c.number.value} встречается в двух документах: {by_number[c.number.value].file} и {c.file}", c.number.value, c.doc_id)
            )
            continue
        by_number[c.number.value] = c

    roots = [c for c in cards if c.doc_type in (DocType.MAIN_CONTRACT, DocType.STANDALONE_AGREEMENT)]
    supplements = [c for c in cards if c.doc_type not in (DocType.MAIN_CONTRACT, DocType.STANDALONE_AGREEMENT)]

    contracts: dict[str, Contract] = {}
    doc_to_contract: dict[str, str] = {}
    for r in roots:
        cid = r.number.value or f"UNNUMBERED-{r.doc_id}"
        if cid in contracts:
            global_issues.append(_issue("number", Severity.WARNING, f"два основных документа с номером {cid}", cid, r.doc_id))
            continue
        contracts[cid] = Contract(
            contract_id=cid,
            title=r.title,
            date=r.date,
            main_doc_id=r.doc_id,
            counterparty=r.counterparty.value,
            counterparty_legal_form=r.counterparty_legal_form,
            city=normalize_city(r.city),
            subject_category=r.subject_category,
            subject_summary=r.subject_summary,
            folders=[r.folder],
        )
        doc_to_contract[r.doc_id] = cid

    def resolve_root(number: str, seen: set[str]) -> str | None:
        """Номер родителя -> id договора. Поддерживает ДС к ДС (идём вверх по цепочке)."""
        if number in contracts:
            return number
        parent_card = by_number.get(number)
        if parent_card is None or number in seen:
            return None
        seen.add(number)
        if parent_card.doc_id in doc_to_contract:
            return doc_to_contract[parent_card.doc_id]
        if parent_card.parent_number.value:
            return resolve_root(parent_card.parent_number.value, seen)
        return None

    pending = sorted(supplements, key=_sort_key)
    for _ in range(3):  # несколько проходов: ДС к ДС может встретиться раньше своего родителя
        rest = []
        for s in pending:
            parent = s.parent_number.value
            root = resolve_root(parent, set()) if parent else None
            if root:
                doc_to_contract[s.doc_id] = root
            else:
                rest.append(s)
        pending = rest

    for s in pending:  # родитель не найден в пакете
        parent = s.parent_number.value
        issue_doc = s.doc_id
        if parent:
            cid = parent
            reason = f"основной договор {parent} отсутствует в пакете: цепочка восстановлена по ссылкам доп. соглашений"
        else:
            folder_numbers = sorted(numbers_in_text(s.folder.replace("_", "-")))
            if folder_numbers:
                cid = folder_numbers[0]
                reason = f"у соглашения нет подтверждённой ссылки на договор; привязано к {cid} только по имени папки"
            else:
                cid = f"UNLINKED-{s.doc_id}"
                reason = "соглашение не удалось привязать ни к одному договору"
        if cid not in contracts:
            contracts[cid] = Contract(contract_id=cid, main_doc_id=None)
            contracts[cid].issues.append(_issue("main_doc", Severity.WARNING, reason, cid, issue_doc))
        elif not parent:
            contracts[cid].issues.append(_issue("link", Severity.WARNING, reason, cid, issue_doc))
        doc_to_contract[s.doc_id] = cid

    card_by_id = {c.doc_id: c for c in cards}
    for cid, contract in contracts.items():
        members = [card_by_id[d] for d, c in doc_to_contract.items() if c == cid]
        supps = sorted([m for m in members if m.doc_id != contract.main_doc_id], key=_sort_key)
        contract.supplement_doc_ids = [m.doc_id for m in supps]
        contract.folders = sorted({m.folder for m in members})
        if len(contract.folders) > 1:
            contract.issues.append(_issue("folders", Severity.INFO, f"документы договора лежат в разных папках: {contract.folders}"))
        _fill_contract_meta(contract, members, supps)
        _check_parent_dates(contract, supps)
        _apply_invalidations(contract, members, by_number)
        _check_foreign_targets(contract, supps, by_number)
        contract.related_contract_ids = sorted(
            {r.number for m in members for r in m.related_contracts if r.number and r.number != cid}
            - {m.number.value for m in members}
        )
        _set_status(contract, supps)
    return contracts, global_issues


def _fill_contract_meta(contract: Contract, members: list[DocumentCard], supps: list[DocumentCard]) -> None:
    if contract.main_doc_id is None:
        titles = [s.parent_title for s in supps if s.parent_title]
        contract.title = titles[0] if titles else None
        contract.subject_category = (
            category_from_title(contract.title)
            or Counter(s.subject_category for s in supps).most_common(1)[0][0]
            if supps
            else SubjectCategory.OTHER
        )
        contract.subject_summary = f"Восстановлено по доп. соглашениям: {contract.title or 'вид договора не указан'}"
    elif contract.subject_category == SubjectCategory.OTHER:
        contract.subject_category = category_from_title(contract.title) or SubjectCategory.OTHER
    if not contract.counterparty:
        contract.counterparty = next((m.counterparty.value for m in members if m.counterparty.value), None)
    if contract.counterparty_legal_form in (LegalForm.UNKNOWN, LegalForm.OTHER):
        forms = [m.counterparty_legal_form for m in members if m.counterparty_legal_form not in (LegalForm.UNKNOWN, LegalForm.OTHER)]
        if forms:
            contract.counterparty_legal_form = Counter(forms).most_common(1)[0][0]
    if not contract.city:
        contract.city = next((normalize_city(m.city) for m in members if m.city), None)


def _check_parent_dates(contract: Contract, supps: list[DocumentCard]) -> None:
    """Сверяем дату договора с тем, как на неё ссылаются доп. соглашения."""
    refs: dict[str, list[str]] = defaultdict(list)
    for s in supps:
        if s.parent_date.value and s.parent_number.value == contract.contract_id:
            refs[s.parent_date.value].append(s.number.value or s.doc_id)
    if not refs:
        return
    if contract.date.value:
        for d, docs in refs.items():
            if d != contract.date.value:
                contract.issues.append(
                    _issue(
                        "date",
                        Severity.WARNING,
                        f"противоречие: в договоре дата {contract.date.value}, а в соглашениях {', '.join(docs)} договор датирован {d}. "
                        "Дата договора взята из самого договора, ссылки соглашений не использованы",
                        {"contract": contract.date.value, "references": {k: v for k, v in refs.items()}},
                    )
                )
        return
    if len(refs) == 1:
        d, docs = next(iter(refs.items()))
        contract.date = VerifiedValue(value=d, quote=None, confidence=0.8, source="cross_document")
        contract.issues.append(_issue("date", Severity.INFO, f"дата договора восстановлена по ссылкам соглашений {', '.join(docs)}", d))
    else:
        contract.issues.append(
            _issue("date", Severity.REJECTED, "дата договора не записана: соглашения ссылаются на разные даты", dict(refs))
        )


def _apply_invalidations(contract: Contract, members: list[DocumentCard], by_number: dict[str, DocumentCard]) -> None:
    invalid: dict[str, str] = {}
    for m in members:
        for inv in m.invalidates:
            target = by_number.get(inv.number)
            by = m.number.value or m.doc_id
            if target is None:
                contract.issues.append(
                    _issue("invalidates", Severity.INFO, f"{by} признаёт недействительным {inv.number}, которого нет в пакете", inv.number, m.doc_id)
                )
                continue
            invalid[target.doc_id] = f"признано недействительным соглашением {by}"
            if inv.date and target.date.value and inv.date != target.date.value:
                contract.issues.append(
                    _issue(
                        "invalidates",
                        Severity.WARNING,
                        f"{by} ссылается на {inv.number} от {inv.date}, но в самом {inv.number} дата {target.date.value}",
                        {"reference": inv.date, "document": target.date.value},
                        m.doc_id,
                    )
                )
            elif inv.date and not target.date.value:
                contract.issues.append(
                    _issue("invalidates", Severity.INFO, f"дата {inv.number} известна только из ссылки в {by}: {inv.date}", inv.date, m.doc_id)
                )

    timeline = []
    for m in sorted(members, key=_sort_key):
        timeline.append(
            TimelineEntry(
                doc_id=m.doc_id,
                number=m.number.value,
                date=m.date.value,
                effective_from=m.effective_from.value,
                valid_until=m.valid_until.value,
                doc_type=m.doc_type,
                summary=m.subject_summary,
                valid=m.doc_id not in invalid,
                invalid_reason=invalid.get(m.doc_id),
            )
        )
    contract.timeline = timeline


def _check_foreign_targets(contract: Contract, supps: list[DocumentCard], by_number: dict[str, DocumentCard]) -> None:
    for s in supps:
        for t in dict.fromkeys(ch.target_document for ch in s.changes):
            if t and t != contract.contract_id and t not in by_number:
                contract.issues.append(
                    _issue(
                        "changes",
                        Severity.WARNING,
                        f"{s.number.value or s.file} изменяет соглашение {t}, которого нет в пакете: исходная редакция неизвестна",
                        t,
                        s.doc_id,
                    )
                )


def _set_status(contract: Contract, supps: list[DocumentCard]) -> None:
    valid_ids = {t.doc_id for t in contract.timeline if t.valid}
    actions = [ch for s in supps if s.doc_id in valid_ids for ch in s.changes]
    if any(ch.action == ChangeAction.TERMINATE and "расторг" in ch.new_value_summary.lower() and "договор" in ch.new_value_summary.lower() for ch in actions):
        contract.status = "terminated"
    elif any(ch.action == ChangeAction.TERMINATE for ch in actions):
        contract.status = "partially_terminated"
    elif any(ch.action == ChangeAction.SUSPEND for ch in actions):
        contract.status = "suspended"


# ---------------------------------------------------------------------------
# 2. Свертка: актуальная редакция условий с учётом всех действующих соглашений
# ---------------------------------------------------------------------------
_APPENDIX_RE = re.compile(r"приложени\w*\s*№?\s*(\d+)", re.IGNORECASE)
_POINT_RE = re.compile(r"(?:п\.|пункт\w*|раздел\w*)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
# срок действия и «прочие условия» самого ДС описывают соглашение, а не договор
SUPPLEMENT_OWN_TERMS = {TermKey.TERM, TermKey.OTHER}
WHOLE_DOC_MARKERS = ("целиком", "в целом", "полностью", "весь договор", "всего договора", "договор в следующей редакции")


# диапазоны пунктов/разделов: «п. 1.1–1.37», «разделы 1–8», «п. 1 – п. 14»
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)*)\s*[–—-]\s*(?:п\.\s*)?(\d+(?:\.\d+)*)")


def _num_key(point: str) -> tuple[int, ...]:
    return tuple(int(x) for x in point.split(".") if x.isdigit())


def _clause_parts(ref: str | None) -> tuple[str | None, set[str]]:
    """Номер приложения и пункты всей ссылки (для проверки «частичного» изменения)."""
    if not ref:
        return None, set()
    app = _APPENDIX_RE.search(ref)
    return (app.group(1) if app else None), set(_POINT_RE.findall(ref))


def _segments(ref: str) -> list[tuple[str | None, set[str], list[tuple[tuple, tuple]]]]:
    """«раздел 8; Приложение № 5, п. 9.1» → [(None, {8}, []), ('5', {9.1}, [])]; «п. 1.1–1.37» → диапазон."""
    out = []
    for seg in re.split(r";", ref):
        app = _APPENDIX_RE.search(seg)
        points = set(_POINT_RE.findall(seg))
        ranges = [(_num_key(a), _num_key(b)) for a, b in _RANGE_RE.findall(seg)]
        out.append((app.group(1) if app else None, points, ranges))
    return out


def _points_overlap(pa: set[str], ra: list, pb: set[str], rb: list) -> bool:
    if pa & pb or any(x.startswith(y + ".") or y.startswith(x + ".") for x in pa for y in pb):
        return True
    for points, ranges in ((pa, rb), (pb, ra)):
        for pt in points:
            k = _num_key(pt)
            if any(lo <= k <= hi or (len(k) < len(lo) and lo[: len(k)] == k) for lo, hi in ranges):
                return True
    return False


def same_clause(a: str | None, b: str | None) -> bool:
    """Одно ли место договора: сравнение по сегментам ссылки (приложение + пункты/диапазоны пунктов)."""
    if not a or not b:
        return False
    if re.sub(r"\s+", "", a.lower()) == re.sub(r"\s+", "", b.lower()):
        return True
    for app_a, pts_a, rng_a in _segments(a):
        for app_b, pts_b, rng_b in _segments(b):
            if app_a != app_b:
                continue
            if app_a is not None and (not (pts_a or rng_a) or not (pts_b or rng_b)):
                return True
            if _points_overlap(pts_a, rng_a, pts_b, rng_b):
                return True
    return False


def is_partial(old_ref: str | None, new_ref: str | None) -> bool:
    """Новое изменение затрагивает только часть пунктов старого положения (п. 1.25 из «п. 1.25, п. 1.26»)."""
    app_o, pts_o = _clause_parts(old_ref)
    app_n, pts_n = _clause_parts(new_ref)
    has_range = bool(_RANGE_RE.search(old_ref or ""))
    return app_o == app_n and bool(pts_n) and (pts_n < pts_o or (has_range and not pts_n >= pts_o))


def _is_whole_document(clause: str) -> bool:
    c = clause.lower()
    return any(m in c for m in WHOLE_DOC_MARKERS)


def consolidate(contract: Contract, cards: dict[str, DocumentCard], today: str | None = None) -> ConsolidatedTerms:
    """Свертка. ``today`` (ISO) — дата, на которую определяются действующие условия (по умолчанию сегодня)."""
    today = today or date.today().isoformat()
    result = ConsolidatedTerms(contract_id=contract.contract_id, status=contract.status)
    expired: dict[str, str] = {}   # doc_id -> дата окончания действия
    unmatched: dict[str, list[str]] = {}  # номер ДС -> пункты, которых не было среди положений предыдущей редакции
    pending: set[int] = set()      # id() версий, которые вступят в силу позже даты среза
    states: dict[TermKey, TermState] = {}

    def state(term: TermKey) -> TermState:
        return states.setdefault(term, TermState(term=term))

    def push(term: TermKey, version: TermVersion, replace: bool, remove_only: bool = False, own_contract: bool = True) -> None:
        st = state(term)
        if replace or remove_only:
            kept, matched = [], False
            by = version.source_number or version.source_doc_id
            for old in st.active:
                if not same_clause(old.clause_ref, version.clause_ref):
                    kept.append(old)
                    continue
                matched = True
                if replace and is_partial(old.clause_ref, version.clause_ref):
                    # меняется только часть пунктов старого положения: оно остаётся, но с пометкой
                    old.amended_by.append(f"{by}: {version.clause_ref}")
                    kept.append(old)
                else:
                    old.replaced_by = by
            comparable = [o for o in st.active if o.source_doc_id not in expired]
            if replace and not matched and comparable and own_contract:
                unmatched.setdefault(by, [])
                if version.clause_ref not in unmatched[by]:
                    unmatched[by].append(version.clause_ref)
            st.active = kept
        st.history.append(version)
        if not remove_only and not history_only:
            st.active.append(version)

    history_only = False  # переключается для изменений чужих соглашений (см. ниже)
    chain_numbers = {cards[d].number.value for d in ([contract.main_doc_id] if contract.main_doc_id else [])
                     + contract.supplement_doc_ids if d in cards and cards[d].number.value}
    foreign: dict[str, list[str]] = {}  # соглашения вне цепочки -> кто их менял

    main = cards.get(contract.main_doc_id) if contract.main_doc_id else None
    if main:
        for kt in main.key_terms:
            push(kt.term, TermVersion(summary=kt.summary, source_doc_id=main.doc_id, source_number=main.number.value,
                                      clause_ref=kt.clause_ref, action="base", effective_from=main.date.value or contract.date.value), replace=False)
        result.applied_docs.append(main.doc_id)
    else:
        result.notes.append("Основной договор отсутствует в пакете: базовая редакция неизвестна, показаны только изменения из соглашений")

    valid = {t.doc_id: t for t in contract.timeline}
    for doc_id in contract.supplement_doc_ids:
        card = cards[doc_id]
        entry = valid.get(doc_id)
        if entry is not None and not entry.valid:
            result.skipped_docs.append({"doc_id": doc_id, "number": card.number.value, "reason": entry.invalid_reason})
            continue
        num = card.number.value or card.file
        touched: set[TermKey] = set()
        is_expired = bool(card.valid_until.value and card.valid_until.value < today)
        if is_expired:
            expired[doc_id] = card.valid_until.value

        whole = [ch for ch in card.changes if ch.action == ChangeAction.RESTATE and _is_whole_document(ch.target_clause)
                 and (ch.target_document in (None, contract.contract_id))]
        if whole:
            # «Изложить Договор в следующей редакции»: всё, что было до этого, заменяется
            for st in states.values():
                for old in st.active:
                    old.replaced_by = num
                st.active = []
            for kt in card.key_terms:
                push(kt.term, TermVersion(summary=kt.summary, source_doc_id=doc_id, source_number=num, clause_ref=kt.clause_ref,
                                          action="restate_whole", effective_from=card.effective_from.value or card.date.value), replace=False)
                touched.add(kt.term)
            result.notes.append(f"{num} излагает договор в новой редакции целиком: базой для последующих изменений служит она")

        for ch in card.changes:
            if ch in whole or ch.action == ChangeAction.INVALIDATE_AGREEMENT:
                continue
            clause = ch.target_clause
            own = ch.target_document in (None, contract.contract_id) or ch.target_document in chain_numbers
            if ch.target_document and ch.target_document != contract.contract_id:
                clause = f"{clause} (соглашения {ch.target_document})"
            if not own:
                foreign.setdefault(ch.target_document, [])
                if num not in foreign[ch.target_document]:
                    foreign[ch.target_document].append(num)
            version = TermVersion(summary=ch.new_value_summary, source_doc_id=doc_id, source_number=num, clause_ref=clause,
                                  action=ch.action.value, effective_from=ch.effective_from or card.effective_from.value or card.date.value)
            is_future = bool(version.effective_from and version.effective_from > today)
            terms = ch.affected_terms or [TermKey.OTHER]
            # одно изменение — одно положение: в основное условие (где есть заменяемый пункт, иначе первое
            # из перечисленных), в остальных условиях — только запись в истории со ссылкой на основное
            primary = next((t for t in terms if any(same_clause(o.clause_ref, clause) for o in state(t).active)), terms[0])
            for term in terms:
                # истёкшее или ещё не вступившее изменение не должно вытеснять текущую редакцию: его положения
                # только добавляются, а после всех шагов снимаются из действующих (в истории они остаются);
                # изменения соглашений вне цепочки договора в действующую редакцию не попадают вовсе
                v = version.model_copy(deep=True)
                if term != primary:
                    v.clause_ref = f"{clause} → см. «{TERM_RU[primary]}»"
                if is_future:
                    pending.add(id(v))
                history_only = term != primary or not own
                push(term, v, replace=ch.action == ChangeAction.RESTATE and not (is_expired or is_future or history_only),
                     remove_only=ch.action == ChangeAction.DELETE and not (is_expired or is_future or history_only),
                     own_contract=own)
                history_only = False
                touched.add(term)

        # условия, которые соглашение устанавливает самостоятельно (напр. новое поручение), но не описаны как изменения.
        # Срок действия и «прочие условия» самого ДС («действует в пределах срока Договора», число экземпляров)
        # описывают соглашение, а не договор: они видны в цепочке документов и в свертку не попадают.
        for kt in card.key_terms:
            if kt.term not in touched and not whole and kt.term not in SUPPLEMENT_OWN_TERMS:
                push(kt.term, TermVersion(summary=kt.summary, source_doc_id=doc_id, source_number=num, clause_ref=kt.clause_ref,
                                          action="add", effective_from=card.effective_from.value or card.date.value), replace=False)
        result.applied_docs.append(doc_id)

    for st in states.values():
        for v in st.active:
            if v.source_doc_id in expired:
                v.replaced_by = f"срок действия истёк {expired[v.source_doc_id]}"
            elif id(v) in pending:
                v.replaced_by = f"вступит в силу {v.effective_from}"
        st.active = [v for v in st.active if v.source_doc_id not in expired and id(v) not in pending]
    for n in sorted({v.source_number for st in states.values() for v in st.history if id(v) in pending}):
        result.notes.append(f"{n}: часть изменений вступает в силу после {today} — они показаны в истории, но не в действующей редакции")
    for doc_id, until in expired.items():
        result.notes.append(f"{cards[doc_id].number.value or cards[doc_id].file}: срок действия истёк {until} — "
                            "его условия показаны в истории, но не входят в действующую редакцию")

    for target, nums in foreign.items():
        result.notes.append(f"{', '.join(nums)} изменяет соглашение {target}, которого нет в цепочке договора: изменения "
                            "показаны в истории, в действующую редакцию договора не включены")
    for by, clauses in unmatched.items():
        result.notes.append(
            f"{by}: пункты {', '.join(f'«{c}»' for c in clauses)} не найдены среди положений предыдущей редакции "
            "(в карточке договора — только существенные условия); изменения добавлены отдельными положениями")

    result.terms = {k.value: v for k, v in sorted(states.items(), key=lambda kv: list(TermKey).index(kv[0]))}
    result.as_of = today
    return result


# ---------------------------------------------------------------------------
# 3. Кластеризация и сравнение
# ---------------------------------------------------------------------------
CONTRACT_DIMENSIONS = ("subject_category", "counterparty_legal_form", "city", "status", "chain_complete", "amendments")
DOCUMENT_DIMENSIONS = ("doc_type", "subject_category", "main_change_area", "text_source", "status")


def contract_features(c: Contract) -> dict[str, str]:
    n = len(c.supplement_doc_ids)
    return {
        "subject_category": c.subject_category.value,
        "counterparty_legal_form": c.counterparty_legal_form.value,
        "city": normalize_city(c.city) or "не указан",
        "status": c.status,
        "chain_complete": "да" if c.main_doc_id else "нет основного договора",
        "amendments": "нет" if n == 0 else ("1–2" if n <= 2 else "3+"),
    }


def document_features(card: DocumentCard) -> dict[str, str]:
    areas = Counter(t.value for ch in card.changes for t in ch.affected_terms)
    return {
        "doc_type": card.doc_type.value,
        "subject_category": card.subject_category.value,
        "main_change_area": areas.most_common(1)[0][0] if areas else "—",
        "text_source": card.text_source,
        "status": card.status,
    }


DIM_ABBR = {
    "subject_category": "P", "counterparty_legal_form": "F", "city": "G", "status": "S", "chain_complete": "M",
    "amendments": "A", "doc_type": "T", "main_change_area": "Z", "text_source": "X",
}


def cluster_items(features: dict[str, dict[str, str]], dimensions: list[str], level: str) -> list[Cluster]:
    """Группировка по совпадению значений выбранных измерений. id вида C-P-1 (договоры, по предмету, №1)."""
    prefix = f"{level[0].upper()}-{''.join(DIM_ABBR.get(d, d[:1].upper()) for d in dimensions)}"
    groups: dict[tuple, list[str]] = defaultdict(list)
    for item_id, feats in features.items():
        groups[tuple(feats.get(d, "—") for d in dimensions)].append(item_id)

    clusters = []
    for i, (key, members) in enumerate(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])), start=1):
        all_dims = set().union(*(features[m].keys() for m in members))
        common, differing = {}, {}
        for d in sorted(all_dims):
            values = {m: features[m].get(d, "—") for m in members}
            if len(set(values.values())) == 1:
                common[d] = next(iter(values.values()))
            else:
                differing[d] = values
        clusters.append(
            Cluster(cluster_id=f"{prefix}-{i}", level=level, key=dict(zip(dimensions, key)),
                    members=sorted(members), common=common, differing=differing)
        )
    return clusters


def compare_contracts(contracts: list[Contract], consolidated: dict[str, ConsolidatedTerms]) -> dict:
    """Сравнение метаданных и актуальных условий (после свертки) нескольких договоров."""
    ids = [c.contract_id for c in contracts]
    meta = {c.contract_id: {**contract_features(c), "date": c.date.value or "—", "counterparty": c.counterparty or "обезличен/не указан",
                            "supplements": len(c.supplement_doc_ids)} for c in contracts}
    meta_diff = {}
    for field in next(iter(meta.values()), {}):
        vals = {cid: meta[cid][field] for cid in ids}
        meta_diff[field] = {"same": len(set(map(str, vals.values()))) == 1, "values": vals}

    terms: dict[str, dict] = {}
    all_terms = sorted({t for cid in ids if cid in consolidated for t in consolidated[cid].terms},
                       key=lambda t: list(TermKey).index(TermKey(t)))
    for t in all_terms:
        per = {}
        for cid in ids:
            st = consolidated.get(cid, ConsolidatedTerms(contract_id=cid)).terms.get(t)
            per[cid] = [f"{v.summary} [{v.source_number}, {v.clause_ref or '—'}]" for v in st.active] if st else []
        present = [cid for cid in ids if per[cid]]
        terms[TERM_RU[TermKey(t)]] = {
            "present_in": present,
            "missing_in": [cid for cid in ids if not per[cid]],
            "values": per,
        }
    missing_consolidation = [cid for cid in ids if cid not in consolidated]
    return {"contracts": ids, "metadata": meta_diff, "terms": terms, "not_consolidated": missing_consolidation}
