from datetime import date

import pytest

from contract_agent.ingest import DocumentLoader, PageText, ParsedDocument
from contract_agent.schemas import (
    ChangeAction,
    ChangeExtraction,
    DocType,
    DocumentExtraction,
    EvidencedValue,
    KeyTermExtraction,
    LegalForm,
    RelatedContractExtraction,
    Severity,
    SubjectCategory,
    TermKey,
)
from contract_agent.validators import TextIndex, build_card, normalize_number, parse_date


def ev(value, quote=None, conf=0.95):
    return EvidencedValue(value=value, quote=quote, confidence=conf)


def extraction(**overrides) -> DocumentExtraction:
    base = dict(
        doc_type=DocType.SUPPLEMENTARY,
        title="Дополнительное соглашение",
        number=ev(None), date=ev(None), city=None,
        parent_number=ev(None), parent_date=ev(None), parent_title=None,
        counterparty=ev(None), counterparty_legal_form=LegalForm.UNKNOWN, signatory_branch=None,
        subject_category=SubjectCategory.JOINT_PROMOTION, subject_summary="",
        effective_from=ev(None), valid_until=ev(None), key_terms=[], changes=[], invalidates=[], related_contracts=[],
        anonymized_fields=[], extraction_notes=[],
    )
    base.update(overrides)
    return DocumentExtraction(**base)


@pytest.mark.parametrize("raw,expected", [
    ("03.06.2020", date(2020, 6, 3)),
    ("02.06.2023г.", date(2023, 6, 2)),
    ("«12» мая 2022 г.", date(2022, 5, 12)),
    ("01 апреля 2023 года", date(2023, 4, 1)),
    ("21 мая 2021", date(2021, 5, 21)),
    ("2024-04-15", date(2024, 4, 15)),
    ("31.02.2024", None),
    ("с момента подписания", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_normalize_number():
    assert normalize_number("№ ар–2021/015") == "АР-2021/015"
    assert normalize_number(" П-118/22-1 ") == "П-118/22-1"
    assert normalize_number("АБ 123_45") == "АБ123-45"


def test_quote_match_tolerates_line_breaks_and_punctuation():
    idx = TextIndex("Стороны договорились о взаимном\nпрекращении обязательств по совместному продвижению")
    assert idx.quote_match("договорились о взаимном прекращении обязательств по совместному") == 1.0
    assert idx.quote_match("стороны договорились расторгнуть договор аренды помещения") < 0.3


@pytest.fixture
def conservation_doc(settings):
    loader = DocumentLoader(settings)
    info = next(d for d in loader.inventory() if "консервация" in d.file)
    return info, loader.parse(info)


def test_document_verified_values_are_kept(conservation_doc):
    info, parsed = conservation_doc
    ex = extraction(
        number=ev("ПД-2022/045-ДС6", "Дополнительное соглашение № ПД-2022/045-ДС6"),
        parent_number=ev("ПД-2022/045", "К Договору № ПД-2022/045"),
        parent_date=ev("12.05.2022", "от «12» мая 2022 г."),
        changes=[ChangeExtraction(
            target_document="ПД-2022/045", target_clause="Программа продвижения", action=ChangeAction.TERMINATE,
            affected_terms=[TermKey.TERMINATION], new_value_summary="прекращение привлечения абонентов с 15.04.2024",
            effective_from="15.04.2024",
            quote="Стороны договорились о взаимном прекращении обязательств по совместному продвижению", confidence=0.95)],
        invalidates=[RelatedContractExtraction(number="ПД-2022/045-ДС5", date="01.03.2024", title=None, relation="недействительно")],
    )
    card = build_card(info, parsed, ex, 0.7, 0.6)
    assert card.number.value == "ПД-2022/045-ДС6"
    assert card.parent_number.value == "ПД-2022/045"
    assert card.parent_date.value == "2022-05-12"
    assert card.changes[0].effective_from == "2024-04-15"
    assert card.invalidates[0].number == "ПД-2022/045-ДС5"
    assert card.status == "ok"


def test_hallucinated_values_are_rejected_not_written(conservation_doc):
    info, parsed = conservation_doc
    ex = extraction(
        number=ev("ПД-2022/045-ДС6", "Дополнительное соглашение № ПД-2022/045-ДС6"),
        date=ev("15.03.2024", "дата подписания 15.03.2024"),             # такой даты в тексте нет
        parent_number=ev("D999999999-01", "К Договору № D999999999-01"),  # номера нет в тексте
        counterparty=ev("ООО «Партнёр Плюс»", "ООО «Партнёр Плюс»", conf=0.4),  # низкая уверенность
        key_terms=[KeyTermExtraction(term=TermKey.REMUNERATION, summary="ставка 70%", clause_ref="п. 1",
                                     quote="вознаграждение составляет 70 процентов от начислений", confidence=0.9)],
    )
    card = build_card(info, parsed, ex, 0.7, 0.6)
    assert card.date.value is None
    assert card.parent_number.value is None
    assert card.counterparty.value is None
    assert card.key_terms == []
    rejected = {i.field: i for i in card.rejected}
    assert {"date", "parent_number", "counterparty", "key_terms[0].remuneration"} <= set(rejected)
    assert rejected["date"].raw_value == "15.03.2024"
    assert "цитата не найдена" in rejected["date"].reason
    assert card.status == "partial"


def test_ocr_uncertain_fragment_lowers_confidence():
    parsed = ParsedDocument("doc_x", "x.pdf", [
        PageText(1, "ДОГОВОР № ПД-2022/045 г. Томск «12» мая 2022 г. Публичное акционерное общество", "ocr", "medium",
                 uncertain_fragments=["«12» мая 2022"]),
    ])
    from contract_agent.ingest import DocumentInfo
    info = DocumentInfo("doc_x", "x.pdf", "f", "x.pdf", 1, 1, 0, True)
    ex = extraction(doc_type=DocType.MAIN_CONTRACT,
                    number=ev("ПД-2022/045", "ДОГОВОР № ПД-2022/045"),
                    date=ev("12.05.2022", "«12» мая 2022 г.", conf=0.9))
    card = build_card(info, parsed, ex, 0.7, 0.6)
    assert card.number.value == "ПД-2022/045"
    assert card.date.value is None  # 0.9 * 0.6 < 0.7 — рукописная дата не записана
    reasons = [i.reason for i in card.issues if i.field == "date"]
    assert any("OCR" in r for r in reasons) and any("низкая уверенность" in r for r in reasons)


def test_file_named_project_gets_warning(settings):
    loader = DocumentLoader(settings)
    info = next(d for d in loader.inventory() if d.file.startswith("Проект"))
    parsed = ParsedDocument(info.doc_id, info.path, [PageText(1, "Договор агентирования № КП-2020/262 от 03.06.2020 г.", "text_layer")])
    card = build_card(info, parsed, extraction(doc_type=DocType.MAIN_CONTRACT, number=ev("КП-2020/262", "№ КП-2020/262")), 0.7, 0.6)
    assert any(i.severity == Severity.WARNING and "Проект" in i.reason for i in card.issues)


def test_value_must_be_linked_to_its_quote(conservation_doc):
    """Три способа «обмануть» проверку из ревью: настоящая цитата, но чужое значение."""
    info, parsed = conservation_doc
    real_quote = "Настоящее Соглашение вступает в силу с момента его подписания уполномоченными представителями Сторон"
    ex = extraction(
        number=ev("ПД-2022/045-ДС6", "Дополнительное соглашение № ПД-2022/045-ДС6"),
        counterparty=ev("ООО «Ромашка»", real_quote),                    # названия нет в цитате
        effective_from=ev("01.06.2024", real_quote),                      # «с момента подписания», но дата не та
        key_terms=[KeyTermExtraction(term=TermKey.REMUNERATION, summary="вознаграждение 95% от выручки",
                                     clause_ref="преамбула", quote="заключили настоящее соглашение к договору", confidence=0.95)],
        changes=[ChangeExtraction(
            target_document="ПД-2022/045", target_clause="п. 1", action=ChangeAction.RESTATE,
            affected_terms=[TermKey.REMUNERATION], new_value_summary="ставка 5%", effective_from=None,
            quote="Стороны договорились о взаимном прекращении обязательств по совместному продвижению", confidence=0.95)],
    )
    card = build_card(info, parsed, ex, 0.7, 0.6)
    assert card.counterparty.value is None and card.effective_from.value is None
    assert card.key_terms == [] and card.changes == []
    reasons = {i.field: i.reason for i in card.rejected}
    assert "слова названия отсутствуют в цитате" in reasons["counterparty"]
    assert "не совпадает с датой документа" in reasons["effective_from"]
    assert "95" in reasons["key_terms[0].remuneration"] and "5" in reasons["changes[0] п. 1"]


def test_our_party_is_not_accepted_as_counterparty(conservation_doc):
    info, parsed = conservation_doc
    quote = "Общество с ограниченной ответственностью «Альфа Сервис», именуемое в дальнейшем «Заказчик»"
    ex = extraction(number=ev("ПД-2022/045-ДС6", "Дополнительное соглашение № ПД-2022/045-ДС6"),
                    counterparty=ev("ООО «Альфа Сервис»", quote), counterparty_legal_form=LegalForm.OOO)
    card = build_card(info, parsed, ex, 0.7, 0.6, our_party="ООО «Альфа Сервис»")
    assert card.counterparty.value is None and card.counterparty_legal_form == LegalForm.UNKNOWN
    assert "«наша» сторона" in {i.field: i.reason for i in card.rejected}["counterparty"]
    # без настройки «нашей» стороны значение с настоящей цитатой принимается
    assert build_card(info, parsed, ex, 0.7, 0.6).counterparty.value == "ООО «Альфа Сервис»"


def test_facts_in_summary_are_checked_near_the_quote():
    from contract_agent.validators import TextIndex, facts_of
    text = ("4.1. Арендная плата составляет 450 000 (четыреста пятьдесят тысяч) рублей в месяц. " + "Прочие условия. " * 400
            + "9.9. Штраф составляет 95% от суммы.")
    idx = TextIndex(text)
    window = idx.window("Арендная плата составляет 450 000 четыреста пятьдесят тысяч рублей")
    assert "450 000" in window and "95%" not in window                 # далёкое число рядом с цитатой не считается
    numbers, dates = facts_of("плата 450 000 руб. в месяц по п. 4.1 с 01.01.2022")
    assert numbers == {"450000"} and [d.isoformat() for d in dates] == ["2022-01-01"]  # «п. 4.1» — ссылка, не значение
