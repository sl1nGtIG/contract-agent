"""Сборка Markdown-отчёта из базы знаний.

Факты (номера, даты, условия, противоречия) рендерятся кодом из проверенных данных —
так в отчёт не попадёт ничего, чего нет в базе знаний. Модель добавляет только
аналитические комментарии (notes), они явно помечены как выводы аналитика.
"""
from __future__ import annotations

from datetime import datetime

from .analysis import CATEGORY_RU, TERM_RU
from .knowledge_base import KnowledgeBase
from .schemas import DocType, Severity, SubjectCategory, TermKey

DOC_TYPE_RU = {
    DocType.MAIN_CONTRACT: "Договор",
    DocType.SUPPLEMENTARY: "Доп. соглашение",
    DocType.STANDALONE_AGREEMENT: "Соглашение",
    DocType.OTHER: "Прочее",
}
STATUS_RU = {
    "active": "действует",
    "partially_terminated": "обязательства частично прекращены",
    "terminated": "расторгнут",
    "suspended": "приостановлен",
}
DIM_RU = {
    "subject_category": "предмет",
    "counterparty_legal_form": "форма контрагента",
    "city": "город",
    "status": "статус",
    "chain_complete": "основной договор в пакете",
    "amendments": "кол-во соглашений",
    "doc_type": "тип документа",
    "main_change_area": "область изменений",
    "text_source": "источник текста",
}
ACTION_RU = {
    "base": "исходная редакция", "restate": "новая редакция", "restate_whole": "новая редакция договора целиком", "add": "дополнено", "delete": "исключено",
    "new_obligation": "новые обязательства", "terminate": "прекращение", "suspend": "приостановление", "other": "изменено",
}
REQUIRED_SECTIONS = (
    "## 1. Резюме",
    "## 2. Состав пакета",
    "## 3. Кластеры",
    "## 4. Договоры",
    "## 5. Связи",
    "## 6. Отказы",
    "## 7. Проверка",
)


def _c(value) -> str:
    """Безопасное значение для ячейки таблицы."""
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _d(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y")
    except ValueError:
        return iso


def _feature_ru(dim: str, value: str) -> str:
    if dim == "subject_category":
        try:
            return CATEGORY_RU[SubjectCategory(value)]
        except ValueError:
            return value
    if dim == "status":
        return STATUS_RU.get(value, value)
    if dim == "doc_type":
        try:
            return DOC_TYPE_RU[DocType(value)]
        except ValueError:
            return value
    if dim == "main_change_area" and value != "—":
        try:
            return TERM_RU[TermKey(value)]
        except ValueError:
            return value
    return value


def _short(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


FULL_REPORT_NAME = "report_full.md"


def _contract_brief(w, kb: KnowledgeBase, n: int, cid: str, c) -> None:
    """Краткий блок договора: паспорт, цепочка, что менялось, что действует сейчас, проблемы."""
    date_src = " (восстановлена по ссылкам ДС)" if c.date.source == "cross_document" else ""
    w(f"### 4.{n}. {cid} — {_c(c.title or 'вид договора не установлен')}\n")
    w(f"{CATEGORY_RU[c.subject_category]} · {_c(c.counterparty or 'контрагент обезличен')} ({c.counterparty_legal_form.value}) · "
      f"{_c(c.city)} · от {_d(c.date.value)}{date_src} · **{STATUS_RU.get(c.status, c.status)}**"
      + ("" if c.main_doc_id else " · **основного договора нет в пакете**") + "\n")

    w("| # | Документ | Дата | Действует с | Содержание | В свертке |")
    w("|---|---|---|---|---|---|")
    for i, t in enumerate(c.timeline, start=1):
        card = kb.cards.get(t.doc_id)
        until = f" по {_d(t.valid_until)}" if t.valid_until else ""
        used = "да" if t.valid else f"**нет**: {t.invalid_reason}"
        w(f"| {i} | {_c(t.number or (card.file if card else t.doc_id))} | {_d(t.date)} | {_d(t.effective_from)}{until} | "
          f"{_c(_short(t.summary, 160))} | {_c(used)} |")
    w("")

    cons = kb.consolidated.get(cid)
    if cons is None:
        w("_Свертка не выполнена._\n")
    else:
        changes = sorted(
            [(v, st.term) for st in cons.terms.values() for v in st.history if v.action != "base"],
            key=lambda x: (x[0].effective_from or "9999", x[0].source_number or ""),
        )
        seen: set[tuple] = set()
        whole: dict[str, list[str]] = {}
        for v, term in changes:
            if v.action == "restate_whole":
                whole.setdefault(v.source_number, []).append(TERM_RU[term].lower())
        w("**Эволюция условий** (изменения, внесённые соглашениями)\n")
        if not changes:
            w("_Соглашения не меняли существенные условия (или изменения не прошли проверку)._")
        for v, _term in changes:
            if v.action == "restate_whole":
                # изложение договора целиком — одной строкой; содержание новой редакции — в таблице ниже и в полном отчёте
                if ("whole", v.source_number) not in seen:
                    seen.add(("whole", v.source_number))
                    w(f"- {_d(v.effective_from)} · **{v.source_number}** · договор изложен в новой редакции целиком; "
                      f"обновлены: {', '.join(dict.fromkeys(whole[v.source_number]))}")
                continue
            key = (v.source_doc_id, (v.clause_ref or "").split(" → см.")[0], v.summary)  # ссылки «→ см.» — то же изменение
            if key in seen:  # одно изменение может затрагивать несколько условий — показываем один раз
                continue
            seen.add(key)
            status = f" _({v.replaced_by})_" if v.replaced_by and v.replaced_by.startswith(("срок", "вступит")) else ""
            w(f"- {_d(v.effective_from)} · **{v.source_number}** · {ACTION_RU.get(v.action, v.action)} · "
              f"{_c(_short(key[1], 50))}: {_short(v.summary, 180)}{status}")
        w("")
        w(f"**Актуальная редакция условий** на {_d(cons.as_of)}\n")
        w("| Условие | Действует сейчас | Источник |")
        w("|---|---|---|")
        for st in cons.terms.values():
            if not st.active or st.term == TermKey.OTHER:
                continue
            # сверху — базовая (или изложенная целиком) редакция условия, ниже — дополнения соглашений по датам
            ordered = sorted(st.active, key=lambda v: (v.action not in ("base", "restate_whole"), v.effective_from or ""))
            head, rest = ordered[0], ordered[1:]
            text = _c(_short(head.summary, 200)) + (" _(частично изменено)_" if head.amended_by else "")
            for v in rest[:3]:
                text += f"<br>+ {_d(v.effective_from)} {_c(v.source_number)}: {_c(_short(v.summary, 110))}"
            if len(rest) > 3:
                text += f"<br>_…ещё {len(rest) - 3} — в полном отчёте_"
            w(f"| {TERM_RU[st.term]} | {text} | {_c(head.source_number)}, {_c(_short(head.clause_ref, 40))} |")
        w("")
        extra = (["не учтены: " + "; ".join(f"{s['number'] or s['doc_id']} ({s['reason']})" for s in cons.skipped_docs)]
                 if cons.skipped_docs else [])
        for note in extra + cons.notes:
            w(f"- {note}")
        if extra or cons.notes:
            w("")

    issues = [i for i in c.issues if i.severity != Severity.INFO]
    if issues:
        w("**Замечания и противоречия**\n")
        for i in issues:
            w(f"- {'⛔' if i.severity == Severity.REJECTED else '⚠️'} {i.reason}")
        w("")
    note = (kb.notes.get("contracts") or {}).get(cid)
    if note:
        w(f"_Комментарий аналитика:_ {note}\n")


def render_report(kb: KnowledgeBase, model: str, usage: dict | None = None, brief: bool = True) -> str:
    """brief=True — краткий отчёт (report.md), brief=False — детальный (report_full.md) с полной историей версий."""
    out: list[str] = []
    w = out.append
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    recognized = sum(1 for s in kb.parse_status.values() if s.get("recognized"))
    usage = usage or kb.usage

    w("# Анализ пакета договоров: кластеры, цепочки и актуальные условия" + ("" if brief else " (детальная версия)") + "\n")
    w(f"_Сформировано {now}; модель: `{model}`; документов: {len(kb.documents)}, распознано: {recognized}, "
      f"договоров (цепочек): {len(kb.contracts)}._\n")
    w("> Факты в таблицах собраны кодом из проверенных карточек документов. Блоки «Комментарий аналитика» — "
      "выводы LLM-агента на основе этих данных. Значения, не прошедшие проверки, в таблицы не попадают — "
      "они перечислены в разделе 6.\n")
    if brief:
        w(f"> Это краткая версия. Полная история версий каждого условия, кластеризация документов и все предупреждения — "
          f"в `{FULL_REPORT_NAME}`.\n")

    # 1 -------------------------------------------------------------------------
    w("## 1. Резюме\n")
    w((kb.notes.get("executive_summary") or {}).get("text") or "_Резюме не предоставлено агентом._")
    w("")

    # 2 -------------------------------------------------------------------------
    w("## 2. Состав пакета и качество распознавания\n")
    w("| Документ | Папка | Тип | Номер | Дата | Текст | Статус карточки | Отклонено |")
    w("|---|---|---|---|---|---|---|---|")
    for doc_id, info in sorted(kb.documents.items(), key=lambda kv: kv[1].path):
        card = kb.cards.get(doc_id)
        ps = kb.parse_status.get(doc_id, {})
        if card is None:
            status = "не распознан" if ps and not ps.get("recognized") else "не обработан"
            reason = ps.get("reason") or ps.get("error") or ""
            w(f"| {_c(info.file)} | {_c(info.folder)} | — | — | — | {_c(ps.get('quality', {}).get('source'))} | "
              f"**{status}** {_c(reason)} | — |")
            continue
        w(f"| {_c(info.file)} | {_c(info.folder)} | {DOC_TYPE_RU[card.doc_type]} | {_c(card.number.value)} | "
          f"{_d(card.date.value)} | {card.text_source} | {card.status} | {len(card.rejected)} |")
    w("")

    # 3 -------------------------------------------------------------------------
    w("## 3. Кластеры договоров\n")
    if not kb.clusters:
        w("_Кластеризация не выполнялась._\n")
    for key, clusters in kb.clusters.items():
        level, dims = key.split(":", 1)
        if brief and level != "contract":
            continue
        dims_list = dims.split(",")
        level_ru = "договоры" if level == "contract" else "документы"
        w(f"### Кластеризация ({level_ru}) по: {', '.join(DIM_RU.get(d, d) for d in dims_list)}\n")
        w("| Кластер | Признаки | Состав |")
        w("|---|---|---|")
        for cl in clusters:
            feats = "; ".join(f"{DIM_RU.get(k, k)}: {_feature_ru(k, v)}" for k, v in cl.key.items())
            members = ", ".join(cl.members) if level == "contract" else ", ".join(
                (kb.cards[m].number.value or kb.cards[m].file) if m in kb.cards else m for m in cl.members)
            w(f"| {cl.cluster_id} | {_c(feats)} | {_c(members)} |")
        w("")
        label = (lambda m: m) if level == "contract" else (
            lambda m: (kb.cards[m].number.value or kb.cards[m].file) if m in kb.cards else m)
        for cl in clusters:
            note = (kb.notes.get("clusters") or {}).get(cl.cluster_id)
            if len(cl.members) < 2 and not note:
                continue
            w(f"**{cl.cluster_id} — сходства и различия.**")
            if cl.common:
                w("Общее: " + "; ".join(f"{DIM_RU.get(k, k)} — {_feature_ru(k, v)}" for k, v in cl.common.items()) + ".")
            if cl.differing:
                w("Различается: " + "; ".join(
                    f"{DIM_RU.get(k, k)} ({', '.join(f'{label(m)}: {_feature_ru(k, v)}' for m, v in vals.items())})"
                    for k, vals in cl.differing.items()) + ".")
            if note:
                w(f"\n_Комментарий аналитика:_ {note}")
            w("")

    # 4 -------------------------------------------------------------------------
    w("## 4. Договоры: цепочки, эволюция условий и актуальная редакция\n")
    if brief:
        for n, (cid, c) in enumerate(sorted(kb.contracts.items()), start=1):
            _contract_brief(w, kb, n, cid, c)
    for n, (cid, c) in enumerate(sorted(kb.contracts.items()) if not brief else [], start=1):
        w(f"### 4.{n}. {cid} — {_c(c.title or 'вид договора не установлен')}\n")
        w("| Параметр | Значение |")
        w("|---|---|")
        date_src = " (восстановлена по ссылкам соглашений)" if c.date.source == "cross_document" else ""
        w(f"| Дата договора | {_d(c.date.value)}{date_src} |")
        w(f"| Предмет | {CATEGORY_RU[c.subject_category]}: {_c(c.subject_summary)} |")
        w(f"| Контрагент | {_c(c.counterparty or 'обезличен')} ({c.counterparty_legal_form.value}) |")
        w(f"| Город | {_c(c.city)} |")
        w(f"| Статус | {STATUS_RU.get(c.status, c.status)} |")
        w(f"| Основной договор в пакете | {'да' if c.main_doc_id else '**нет**'} |")
        w(f"| Папки | {_c(', '.join(c.folders))} |")
        if c.related_contract_ids:
            w(f"| Связанные договоры | {_c(', '.join(c.related_contract_ids))} |")
        w("")

        w("**Цепочка документов**\n")
        w("| # | Документ | Тип | Дата | Действует с | Содержание | Учтён в свертке |")
        w("|---|---|---|---|---|---|---|")
        for i, t in enumerate(c.timeline, start=1):
            card = kb.cards.get(t.doc_id)
            used = "да" if t.valid else f"**нет** — {t.invalid_reason}"
            w(f"| {i} | {_c(t.number or (card.file if card else t.doc_id))} | {DOC_TYPE_RU[t.doc_type]} | {_d(t.date)} | "
              f"{_d(t.effective_from)} | {_c(t.summary)} | {_c(used)} |")
        w("")

        cons = kb.consolidated.get(cid)
        if cons is None:
            w("_Свертка не выполнена._\n")
        else:
            w("**Эволюция условий**\n")
            evolved = [st for st in cons.terms.values() if len(st.history) > 1 or any(v.action != "base" for v in st.history)]
            if not evolved:
                w("_Соглашения не меняли существенные условия (или изменения не прошли проверку)._\n")
            for st in evolved:
                w(f"- **{TERM_RU[st.term]}**")
                for v in st.history:
                    mark = (f" → заменено {v.replaced_by}" if v.replaced_by and not v.replaced_by.startswith("срок")
                            else f" → {v.replaced_by}" if v.replaced_by else "")
                    w(f"  - {_d(v.effective_from)} · {v.source_number or v.source_doc_id} · {ACTION_RU.get(v.action, v.action)} · "
                      f"{v.clause_ref or '—'}: {v.summary}{mark}")
            w("")
            w("**Актуальная редакция условий (свертка)**" + (f" — по состоянию на {_d(cons.as_of)}" if cons.as_of else "") + "\n")
            w("| Условие | Действующая редакция | Источник |")
            w("|---|---|---|")
            for st in cons.terms.values():
                for v in st.active:
                    amended = f" _(частично изменено: {'; '.join(v.amended_by)})_" if v.amended_by else ""
                    w(f"| {TERM_RU[st.term]} | {_c(v.summary)}{_c(amended) if amended else ''} | {_c(v.source_number)}, {_c(v.clause_ref)} |")
            w("")
            if cons.skipped_docs:
                w("Не учтены в свертке: " + "; ".join(f"{s['number'] or s['doc_id']} ({s['reason']})" for s in cons.skipped_docs) + ".\n")
            for note in cons.notes:
                w(f"- {note}")
            if cons.notes:
                w("")

        issues = [i for i in c.issues if i.severity != Severity.INFO] + [i for i in c.issues if i.severity == Severity.INFO]
        if issues:
            w("**Замечания и противоречия**\n")
            for i in issues:
                icon = {"rejected": "⛔", "warning": "⚠️", "info": "ℹ️"}[i.severity.value]
                w(f"- {icon} {i.reason}")
            w("")
        note = (kb.notes.get("contracts") or {}).get(cid)
        if note:
            w(f"_Комментарий аналитика:_ {note}\n")

    # 5 -------------------------------------------------------------------------
    w("## 5. Связи между договорами\n")
    links = [(cid, r) for cid, c in sorted(kb.contracts.items()) for r in c.related_contract_ids]
    if not links:
        w("_Перекрёстных ссылок между договорами не найдено._\n")
    else:
        w("| Договор | Ссылается на | Есть в пакете |")
        w("|---|---|---|")
        for cid, r in links:
            w(f"| {cid} | {r} | {'да' if r in kb.contracts else 'нет'} |")
        w("")
    for i in kb.link_issues:
        w(f"- ⚠️ {i.reason}")

    # 6 -------------------------------------------------------------------------
    w("## 6. Отказы от записи значений\n")
    w("Эти значения предложила модель, но они не прошли встроенные проверки и **не записаны** в базу знаний.\n")
    rejected = [(kb.cards[i.doc_id].file if i.doc_id in kb.cards else (i.doc_id or "—"), i) for i in kb.all_rejected()]
    if not rejected:
        w("_Отказов нет._\n")
    else:
        w("| Документ | Поле | Значение-кандидат | Причина отказа |")
        w("|---|---|---|---|")
        for file, i in rejected:
            raw = i.raw_value if not isinstance(i.raw_value, str) else i.raw_value[:120]
            w(f"| {_c(file)} | {_c(i.field)} | {_c(raw)} | {_c(i.reason)} |")
        w("")
    warnings = [(c.file, i) for c in kb.cards.values() for i in c.issues if i.severity == Severity.WARNING]
    if warnings and not brief:
        w("<details><summary>Предупреждения по документам (значения записаны, но требуют внимания)</summary>\n")
        for file, i in warnings:
            w(f"- {file}: {i.field} — {i.reason}")
        w("\n</details>\n")

    # 7 -------------------------------------------------------------------------
    w("## 7. Проверка выполнения плана (reflection)\n")
    if kb.reflection:
        for item in kb.reflection["items"]:
            w(f"- [{'x' if item['ok'] else ' '}] {item['title']}" + (f" — {item['detail']}" if item.get("detail") else ""))
        quality = kb.reflection.get("quality")
        if quality:
            w("\n**Метрики качества работы агента**\n")
            w("| Метрика | Значение |")
            w("|---|---|")
            for name, value in quality.items():
                w(f"| {name} | {value} |")
    else:
        w("_Проверка будет выполнена после генерации отчёта (check_plan_completion)._")
    if usage:
        w(f"\n_Расход токенов: вход {usage.get('input_tokens', 0):,}, выход {usage.get('output_tokens', 0):,}, "
          f"из кеша {usage.get('cache_read_input_tokens', 0):,}; вызовов модели: {usage.get('calls', 0)}".replace(",", " ")
          + (f"; стоимость ${usage['cost_usd']:.2f}" if usage.get("cost_usd") else "") + "._")
    return "\n".join(out) + "\n"
