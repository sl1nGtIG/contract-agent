from contract_agent.analysis import (
    cluster_items,
    compare_contracts,
    consolidate,
    contract_features,
    link_chains,
    same_clause,
)
from contract_agent.schemas import ChangeAction, DocType, Severity, SubjectCategory, TermKey

from .conftest import make_card


def test_chain_links_by_number_and_applies_invalidations(orion_cards):
    contracts, issues = link_chains(orion_cards)
    assert list(contracts) == ["ПД-2022/045"]
    c = contracts["ПД-2022/045"]
    assert c.main_doc_id == "m"
    assert set(c.supplement_doc_ids) == {"ds2", "ds4", "stop1", "stop2"}
    invalid = {t.number: t.invalid_reason for t in c.timeline if not t.valid}
    assert set(invalid) == {"ПД-2022/045-ДС2", "ПД-2022/045-ДС5"}
    assert "ПД-2022/045-ДС4" in invalid["ПД-2022/045-ДС2"]
    assert c.status == "partially_terminated"


def test_consolidation_skips_invalid_and_replaces_clauses(orion_cards):
    contracts, _ = link_chains(orion_cards)
    c = contracts["ПД-2022/045"]
    cards = {x.doc_id: x for x in orion_cards}
    cons = consolidate(c, cards)

    remuneration = cons.terms[TermKey.REMUNERATION.value]
    assert [v.summary for v in remuneration.active] == ["55% от начислений (новая версия)"]
    assert remuneration.history[0].summary == "45% от оплаченных начислений"
    assert remuneration.history[0].replaced_by == "ПД-2022/045-ДС4"
    # недействительный ДС не применён
    assert "ПД-2022/045-ДС2" not in {v.source_number for v in remuneration.history}
    assert {s["number"] for s in cons.skipped_docs} == {"ПД-2022/045-ДС2", "ПД-2022/045-ДС5"}

    contacts = cons.terms[TermKey.CONTACTS.value]
    assert contacts.active[0].summary == "e-mail new@partner.example"
    assert cons.terms[TermKey.TERMINATION.value].active[0].source_number == "ПД-2022/045-ДС6"


def test_missing_main_contract_is_reconstructed_from_supplements():
    ds = make_card("a", "КП-2015/009-ДС12", parent="КП-2015/009", parent_date="2016-02-10", date="2026-06-01",
                   parent_title="Агентский договор", category=SubjectCategory.INCENTIVE_PROGRAM,
                   changes=[("КП-2015/009", "Таблица 1 п.1.1 Приложения №7", ChangeAction.RESTATE, [TermKey.REMUNERATION], "группа Б 675 → 1200 руб.")])
    ocr = make_card("b", "КП-2015/009-ДС3", parent="КП-2015/009", parent_date="2016-02-10", date="2018-04-01",
                    category=SubjectCategory.AGENCY_ORDER,
                    key_terms=[(TermKey.REMUNERATION, "1200 руб. за точку", "п. 4.1")])
    contracts, _ = link_chains([ds, ocr])
    c = contracts["КП-2015/009"]
    assert c.main_doc_id is None
    assert c.subject_category == SubjectCategory.COMMERCIAL_REPRESENTATION  # по виду договора из ссылки ДС
    assert c.date.value == "2016-02-10" and c.date.source == "cross_document"
    assert any("отсутствует в пакете" in i.reason for i in c.issues)

    cons = consolidate(c, {"a": ds, "b": ocr})
    rem = cons.terms[TermKey.REMUNERATION.value]
    assert len(rem.active) == 2  # поручение (2017) + ставки Приложения 7 (2026)
    assert any("отсутствует" in n for n in cons.notes)


def test_date_conflict_between_contract_and_references_is_reported():
    main = make_card("m", "КП-2021/001", DocType.MAIN_CONTRACT, date="2021-02-18")
    ds3 = make_card("d3", "КП-2021/001-ДС3", parent="КП-2021/001", parent_date="2021-02-15", date="2021-11-09")
    contracts, _ = link_chains([main, ds3])
    c = contracts["КП-2021/001"]
    assert c.date.value == "2021-02-18"  # дата из самого договора не перезаписывается
    conflict = [i for i in c.issues if i.field == "date"]
    assert conflict and conflict[0].severity == Severity.WARNING and "2021-02-15" in conflict[0].reason


def test_conflicting_references_without_main_date_are_rejected():
    a = make_card("a", "D1-01".replace("D1", "D100000001"), parent="D100000000-01", parent_date="2021-02-15")
    b = make_card("b", "D100000002-01", parent="D100000000-01", parent_date="2021-02-18")
    contracts, _ = link_chains([a, b])
    c = contracts["D100000000-01"]
    assert c.date.value is None
    assert any(i.severity == Severity.REJECTED for i in c.issues)


def test_supplement_to_supplement_and_whole_restatement():
    main = make_card("m", "КП-2020/262", DocType.MAIN_CONTRACT, date="2020-06-03",
                     key_terms=[(TermKey.DEFINITIONS, "группа тарифов Б: старый перечень", "п. 1.26"),
                                (TermKey.REMUNERATION, "ставки 2020", "Приложение №7")])
    ds_def = make_card("d1", "КП-2020/262-ДС1", parent="КП-2020/262", date="2023-06-02",
                       changes=[("КП-2020/262", "п. 1.25, п. 1.26", ChangeAction.RESTATE, [TermKey.DEFINITIONS], "группа тарифов Б: новый перечень")])
    ds_full = make_card("d2", "КП-2020/262-ДС2", parent="КП-2020/262", date="2024-04-01",
                        changes=[("КП-2020/262", "Договор целиком", ChangeAction.RESTATE, [TermKey.OTHER], "новая редакция")],
                        key_terms=[(TermKey.REMUNERATION, "ставки 2024", "Приложение №7"),
                                   (TermKey.DEFINITIONS, "определения 2024", "раздел 1")])
    ds_ds = make_card("d3", "КП-2020/262-ДС3", parent="КП-2020/262", date="2024-05-01",
                      changes=[("КП-2020/262-С1", "п. 1 – п. 14 Соглашения", ChangeAction.RESTATE, [TermKey.OBLIGATIONS], "сервисные услуги")])
    contracts, _ = link_chains([main, ds_def, ds_full, ds_ds])
    c = contracts["КП-2020/262"]
    assert any("КП-2020/262-С1" in i.reason for i in c.issues)  # изменяемого ДС нет в пакете

    cons = consolidate(c, {x.doc_id: x for x in (main, ds_def, ds_full, ds_ds)})
    assert [v.summary for v in cons.terms["remuneration"].active] == ["ставки 2024"]
    assert [v.summary for v in cons.terms["definitions"].active] == ["определения 2024"]
    # изменения соглашения вне цепочки видны в истории, но не в действующей редакции договора
    assert "(соглашения КП-2020/262-С1)" in cons.terms["obligations"].history[0].clause_ref
    assert cons.terms["obligations"].active == []
    assert any("КП-2020/262-С1" in n and "не включены" in n for n in cons.notes)
    defs_history = [v.summary for v in cons.terms["definitions"].history]
    assert defs_history == ["группа тарифов Б: старый перечень", "группа тарифов Б: новый перечень", "определения 2024"]


def test_city_is_normalized_for_clustering():
    from contract_agent.analysis import normalize_city
    assert {normalize_city(c) for c in ("г. Омск", "Омск", "город Омск", "г Омск")} == {"Омск"}
    assert normalize_city("  ") is None and normalize_city(None) is None


def test_same_clause():
    assert same_clause("Приложение №7", "Таблица 1 п.1.1 Приложения №7")
    assert same_clause("п. 1.26", "п. 1.25, п. 1.26")
    assert not same_clause("п. 12.2", "п. 14")
    assert not same_clause("Приложение №5", "Приложение №3")
    # диапазоны и составные ссылки, которые пишет модель
    assert same_clause("п. 1.1–1.37 новой редакции Договора", "п. 1.26")
    assert same_clause("раздел 8 новой редакции Договора; Приложение № 5, п. 9.1", "п. 8.14")
    assert not same_clause("раздел 8; Приложение № 5, п. 9.1", "п. 2.1.2 Приложения № 4")
    assert not same_clause("п. 2.1–2.3", "п. 3.1")
    assert same_clause("разделы 1–8 Соглашения", "п. 3.2.2.1, табл. 6")
    assert same_clause("п. 1 – п. 14 Соглашения", "п. 8.1")
    assert not same_clause("разделы 1–8 Соглашения", "п. 9.1")


def test_partial_amendment_of_range_keeps_old_provision():
    from contract_agent.analysis import is_partial
    assert is_partial("п. 1.25, п. 1.26", "п. 1.25")
    assert is_partial("п. 1.1–1.37", "п. 1.26")
    assert not is_partial("п. 8.14", "п. 8.14")


def test_clustering_and_comparison(orion_cards):
    other = make_card("k", "КП-2020/262", DocType.MAIN_CONTRACT, date="2020-06-03", form="ИП",
                      category=SubjectCategory.COMMERCIAL_REPRESENTATION, city="г. Омск",
                      key_terms=[(TermKey.REMUNERATION, "ставки 2020", "Приложение №7")])
    other2 = make_card("k2", "КП-2021/001", DocType.MAIN_CONTRACT, date="2021-02-18",
                       category=SubjectCategory.COMMERCIAL_REPRESENTATION, city="г. Москва")
    contracts, _ = link_chains(orion_cards + [other, other2])
    feats = {cid: contract_features(c) for cid, c in contracts.items()}
    clusters = cluster_items(feats, ["subject_category"], "contract")
    assert clusters[0].cluster_id == "C-P-1"
    assert clusters[0].members == ["КП-2020/262", "КП-2021/001"]
    assert "city" in clusters[0].differing and "counterparty_legal_form" in clusters[0].differing

    cards = {x.doc_id: x for x in orion_cards + [other, other2]}
    cons = {cid: consolidate(c, cards) for cid, c in contracts.items()}
    cmp = compare_contracts([contracts["КП-2020/262"], contracts["КП-2021/001"]], cons)
    assert cmp["metadata"]["subject_category"]["same"] is True
    assert cmp["terms"]["Вознаграждение / цена"]["missing_in"] == ["КП-2021/001"]


def test_change_touching_several_terms_is_active_in_one_term_only():
    main = make_card("m", "АР-1", DocType.MAIN_CONTRACT, date="2021-01-01",
                     key_terms=[(TermKey.REMUNERATION, "арендная плата 450 000", "п. 4.1"),
                                (TermKey.PAYMENT_PROCEDURE, "оплата до 5 числа", "п. 4.2")])
    ds = make_card("d", "АР-1-ДС1", parent="АР-1", date="2022-01-01",
                   changes=[("АР-1", "п. 4.1", ChangeAction.RESTATE,
                             [TermKey.PAYMENT_PROCEDURE, TermKey.REMUNERATION, TermKey.OBLIGATIONS], "арендная плата 480 000")])
    contracts, _ = link_chains([main, ds])
    cons = consolidate(contracts["АР-1"], {"m": main, "d": ds}, today="2026-01-01")
    # основное условие — то, где есть заменяемый п. 4.1, хотя модель назвала его вторым
    assert [v.summary for v in cons.terms["remuneration"].active] == ["арендная плата 480 000"]
    assert [v.summary for v in cons.terms["payment_procedure"].active] == ["оплата до 5 числа"]
    assert cons.terms["obligations"].active == []
    assert "см. «Вознаграждение / цена»" in cons.terms["obligations"].history[0].clause_ref
