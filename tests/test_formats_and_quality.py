"""Форматы входных данных, проверка типов аргументов, свертка с датой среза, метрики качества и отчёты."""
import zipfile

import pymupdf
import pytest

from contract_agent.analysis import consolidate, link_chains
from contract_agent.ingest import DocumentLoader
from contract_agent.knowledge_base import KnowledgeBase
from contract_agent.reflection import check_plan
from contract_agent.report import FULL_REPORT_NAME, REQUIRED_SECTIONS
from contract_agent.schemas import ChangeAction, DocType, OcrPage, OcrResult, TermKey, VerifiedValue
from contract_agent.tools import ToolContext, analyst_registry, write_reports

from .conftest import make_card

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>Договор аренды № D300000001-01</w:t></w:r></w:p>
<w:p><w:r><w:t>Арендатор уплачивает арендную плату </w:t></w:r><w:r><w:t>50 000 рублей в месяц.</w:t></w:r></w:p>
<w:p><w:r><w:t>Арендодатель передаёт во временное владение и пользование нежилое помещение площадью 120 кв. м. Срок аренды — 11 месяцев с даты подписания акта приёма-передачи. Оплата ежемесячно до 5 числа.</w:t></w:r></w:p>
</w:body></w:document>"""


class OcrOnlyLLM:
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "calls": 0}

    def ocr_pages(self, pages, doc_path=None):
        return OcrResult(pages=[OcrPage(page=n, text="Акт приёма-передачи помещения по договору аренды № D300000001-01. "
                                                     "Стороны подтверждают передачу помещения площадью 120 кв. м. Помещение передано в состоянии, "
                                                     "пригодном для использования, претензий к техническому состоянию стороны не имеют.",
                                        legibility="high", uncertain_fragments=[]) for n, _ in pages])


@pytest.fixture
def mixed_input(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    with zipfile.ZipFile(root / "аренда.docx", "w") as z:
        z.writestr("word/document.xml", DOCX_XML)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "scan")  # текст только на картинке — рендерим страницу в PNG
    page.get_pixmap(dpi=50).save(root / "акт.png")
    (root / "заметка.txt").write_text("Дополнительное соглашение к договору аренды: срок продлён до 31.12.2027 года.", encoding="utf-8")
    (root / "реестр.xlsx").write_bytes(b"PK\x03\x04 not really excel")
    (root / "битый.pdf").write_bytes(b"%PDF-1.4 garbage")
    (root / "Thumbs.db").write_bytes(b"\x00")
    return root


def ctx_for(settings, llm=None):
    return ToolContext(settings=settings, kb=KnowledgeBase(settings.kb_path), loader=DocumentLoader(settings, llm), llm=llm)


def test_inventory_detects_every_format_and_skips_nothing_silently(settings, mixed_input):
    docs = {d.file: d.format for d in DocumentLoader(settings.with_overrides(input_dir=mixed_input)).inventory()}
    assert docs == {"аренда.docx": "docx", "акт.png": "image", "заметка.txt": "text",
                    "реестр.xlsx": "unsupported", "битый.pdf": "corrupted"}  # системный Thumbs.db игнорируется


def test_each_format_is_read_by_its_method(settings, mixed_input):
    s = settings.with_overrides(input_dir=mixed_input)
    reg = analyst_registry()

    ctx = ctx_for(s)  # без LLM
    listing, _ = reg.execute(ctx, "list_documents", {})
    ids = {d["path"]: d["doc_id"] for d in listing["documents"]}
    assert "method=ocr" in listing["how_to_read"]["image"]

    res, err = reg.execute(ctx, "parse_document", {"doc_id": ids["аренда.docx"]})
    assert not err and res["recognized"] and "50 000 рублей" in res["preview"]

    res, err = reg.execute(ctx, "parse_document", {"doc_id": ids["реестр.xlsx"]})
    assert err and "не поддерживается" in res["error"] and "продолжайте" in res["hint"]
    assert ctx.kb.parse_status[ids["реестр.xlsx"]]["recognized"] is False  # зафиксирован как нераспознанный

    res, err = reg.execute(ctx, "parse_document", {"doc_id": ids["битый.pdf"]})
    assert err and "повреждён" in res["error"]

    res, err = reg.execute(ctx, "parse_document", {"doc_id": ids["заметка.txt"], "method": "ocr"})
    assert err and "OCR не применяется" in res["error"]

    res, err = reg.execute(ctx, "parse_document", {"doc_id": ids["акт.png"]})
    assert err and "нужен LLM" in res["error"]

    ctx = ctx_for(s, OcrOnlyLLM())  # с OCR изображение читается
    reg.execute(ctx, "list_documents", {})
    res, err = reg.execute(ctx, "parse_document", {"doc_id": ids["акт.png"], "method": "ocr"})
    assert not err and res["quality"]["source"] == "ocr" and "120 кв. м" in res["preview"]


def test_tool_arguments_are_type_checked(settings, mixed_input):
    ctx = ctx_for(settings.with_overrides(input_dir=mixed_input))
    reg = analyst_registry()
    reg.execute(ctx, "list_documents", {})
    reg.execute(ctx, "parse_document", {"doc_id": next(iter(ctx.kb.documents))})

    res, err = reg.execute(ctx, "search_documents", {"query": "арендная плата", "max_hits": "3"})
    assert not err  # "3" → 3: безопасное приведение типа
    for bad in ({"query": "аренда", "max_hits": 100},          # вне диапазона 1..20
                {"query": "аренда", "doc_ids": "doc_1"},        # строка вместо списка
                {"query": "аренда", "unexpected": True}):       # лишнее поле
        res, err = reg.execute(ctx, "search_documents", bad)
        assert err and res["error"].startswith("неверные аргументы"), bad
    res, err = reg.execute(ctx, "parse_document", {"doc_id": "x", "method": "magic"})
    assert err and "method" in res["error"]


def test_consolidation_ignores_supplement_own_terms_and_pending_changes():
    main = make_card("m", "D100000000-01", DocType.MAIN_CONTRACT, date="2020-01-01",
                     key_terms=[(TermKey.TERM, "бессрочно", "п. 11.1"), (TermKey.REMUNERATION, "ставка 300", "Приложение №7")])
    ds_now = make_card("a", "D100000001-01", parent="D100000000-01", date="2024-01-01", effective_from="2024-01-01",
                       key_terms=[(TermKey.TERM, "ДС действует в пределах срока Договора", "п. 3"),
                                  (TermKey.OTHER, "составлено в двух экземплярах", "п. 4")],
                       changes=[("D100000000-01", "Приложение №7", ChangeAction.RESTATE, [TermKey.REMUNERATION], "ставка 400")])
    ds_future = make_card("b", "D100000002-01", parent="D100000000-01", date="2026-01-01",
                          changes=[("D100000000-01", "Приложение №7", ChangeAction.RESTATE, [TermKey.REMUNERATION], "ставка 900")])
    ds_future.changes[0].effective_from = "2027-01-01"
    contracts, _ = link_chains([main, ds_now, ds_future])
    cons = consolidate(contracts["D100000000-01"], {c.doc_id: c for c in (main, ds_now, ds_future)}, today="2026-09-26")

    assert [v.summary for v in cons.terms["term"].active] == ["бессрочно"]  # «ДС действует в пределах…» не условие договора
    assert "other" not in cons.terms
    remuneration = cons.terms["remuneration"]
    assert [v.summary for v in remuneration.active] == ["ставка 400"]      # 900 ещё не вступила в силу
    assert remuneration.history[-1].summary == "ставка 900" and "вступит в силу" in remuneration.history[-1].replaced_by
    assert any("вступает в силу после" in n for n in cons.notes)


def test_reflection_reports_quality_metrics_and_reports_are_brief_and_full(settings, orion_cards):
    kb = KnowledgeBase(settings.kb_path)
    for c in orion_cards:
        kb.cards[c.doc_id] = c
        kb.parse_status[c.doc_id] = {"recognized": True, "quality": {"source": "text_layer"}}
    kb.contracts, kb.link_issues = link_chains(orion_cards)
    kb.consolidated = {cid: consolidate(c, kb.cards) for cid, c in kb.contracts.items()}
    orion_cards[0].date = VerifiedValue(value="2022-05-12", confidence=0.9)
    ctx = ToolContext(settings=settings, kb=kb, loader=DocumentLoader(settings), llm=None)

    write_reports(ctx)
    brief = settings.report_path.read_text(encoding="utf-8")
    full = settings.report_path.with_name(FULL_REPORT_NAME).read_text(encoding="utf-8")
    assert all(section in brief for section in REQUIRED_SECTIONS)
    assert len(brief) < len(full) and FULL_REPORT_NAME in brief

    verdict = check_plan(kb, settings.report_path)
    quality = verdict["quality"]
    assert quality["Значений подтверждено проверками"].endswith("(100%)")
    assert quality["Договоров без основного документа в пакете"] == "0"
    item = lambda v, key: next(i for i in v["items"] if i["key"] == key)  # noqa: E731
    assert not item(verdict, "report")["ok"]  # отчёт записан в обход generate_report — не засчитывается

    kb.report_generated_at = "2999-01-01T00:00:00"
    assert item(check_plan(kb, settings.report_path), "report")["ok"]
    kb.report_generated_at, kb.changed_at = "2026-01-01T00:00:00", "2026-01-02T00:00:00"
    assert "устарел" in item(check_plan(kb, settings.report_path), "report")["detail"]


def test_empty_consolidation_passes_only_with_analyst_explanation(settings):
    ds = make_card("x", "ДС-9", parent="Д-1", date="2018-04-01",
                   key_terms=[(TermKey.REMUNERATION, "1500 руб.", "п. 4")])
    ds.valid_until = VerifiedValue(value="2017-12-31", confidence=0.9)   # единственное соглашение давно истекло
    kb = KnowledgeBase(settings.kb_path)
    kb.cards = {"x": ds}
    kb.contracts, _ = link_chains([ds])
    kb.consolidated = {cid: consolidate(c, kb.cards, today="2026-01-01") for cid, c in kb.contracts.items()}
    check = lambda: next(i for i in check_plan(kb, settings.report_path)["items"] if i["key"] == "consolidation_content")  # noqa: E731
    assert not check()["ok"]
    kb.notes["contracts"] = {"Д-1": "Действующих условий нет: единственное ДС истекло 31.12.2017."}
    assert check()["ok"]
