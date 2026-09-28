from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contract_agent.config import Settings  # noqa: E402
from contract_agent.schemas import (  # noqa: E402
    Change,
    ChangeAction,
    DocType,
    DocumentCard,
    KeyTerm,
    LegalForm,
    RelatedContract,
    SubjectCategory,
    TermKey,
    VerifiedValue,
)

# Тесты самодостаточны: PDF генерируются на лету (синтетические тексты по образцу тестового пакета),
# поэтому папка data/ с документами заказчика для запуска тестов не нужна.
CONSERVATION_TEXT = (
    "ООО «Альфа Сервис», г. Москва\n"
    "Дополнительное соглашение № ПД-2022/045-ДС6\n"
    "К Договору № ПД-2022/045 от «12» мая 2022 г.\n"
    "г. Томск\n"
    "Общество с ограниченной ответственностью «Альфа Сервис», именуемое в дальнейшем «Заказчик», с одной стороны, "
    "и Партнер, с другой стороны, совместно именуемые «Стороны», заключили настоящее соглашение к договору "
    "№ ПД-2022/045 от «12» мая 2022 г. о нижеследующем:\n"
    "1. Стороны договорились о взаимном прекращении обязательств по совместному продвижению реализуемых ими услуг "
    "в рамках Совместного предложения путем проведения Программы продвижения в части поиска и привлечения новых "
    "Абонентов с 15.04.2024 г.\n"
    "2. Стороны продолжают выполнять обязательства по Договору, связанные с обслуживанием Абонентов, ранее "
    "подключивших Совместное предложение.\n"
    "3. Настоящее Соглашение вступает в силу с момента его подписания уполномоченными представителями Сторон.\n"
    "4. Ранее подписанное Соглашение № ПД-2022/045-ДС5 от 01.03.2024г. считать недействительным."
)
DRAFT_TEXT = ("Договор агентирования № КП-2020/262\nг. Омск 03.06.2020 г.\n"
              "Принципал поручает, а Агент обязуется от имени и за счёт Принципала заключать договоры с клиентами "
              "в письменной форме в порядке, предусмотренном Приложением № 2.")


def _text_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    html = "".join(f"<p>{line}</p>" for line in text.split("\n"))
    page.insert_htmlbox(pymupdf.Rect(50, 50, 550, 800), html)
    doc.save(path)


def _scan_pdf(path: Path, text: str) -> None:
    """PDF без текстового слоя: страница с текстом рендерится в картинку и вставляется изображением."""
    src = pymupdf.open()
    src.new_page().insert_htmlbox(pymupdf.Rect(50, 50, 550, 800), f"<p>{text}</p>")
    png = src[0].get_pixmap(dpi=60).tobytes("png")
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=png)
    doc.save(path)


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """Мини-пакет по образцу тестового: текстовое ДС, скан договора, файл «Проект Договора»."""
    root = tmp_path_factory.mktemp("sample_package")
    orion = root / "Партнёр"
    orion.mkdir()
    _text_pdf(orion / "ДС_консервация.pdf", CONSERVATION_TEXT)
    _scan_pdf(orion / "Партнёрский_договор_скан.pdf", "ДОГОВОР № ПД-2022/045 г. Томск «12» мая 2022 г.")
    ip = root / "Представительство"
    ip.mkdir()
    _text_pdf(ip / "Проект Договора.pdf", DRAFT_TEXT)
    return root


def vv(value):
    return VerifiedValue(value=value, confidence=0.95) if value else VerifiedValue()


def make_card(doc_id, number, doc_type=DocType.SUPPLEMENTARY, date=None, parent=None, parent_date=None, folder="f",
              category=SubjectCategory.COMMERCIAL_REPRESENTATION, key_terms=(), changes=(), invalidates=(), related=(),
              effective_from=None, title=None, parent_title=None, city="г. Москва", form=LegalForm.OOO) -> DocumentCard:
    return DocumentCard(
        doc_id=doc_id,
        file=f"{doc_id}.pdf",
        folder=folder,
        doc_type=doc_type,
        title=title or ("Договор" if doc_type == DocType.MAIN_CONTRACT else "Дополнительное соглашение"),
        number=vv(number),
        date=vv(date),
        parent_number=vv(parent),
        parent_date=vv(parent_date),
        parent_title=parent_title,
        effective_from=vv(effective_from),
        subject_category=category,
        subject_summary=f"документ {number}",
        counterparty_legal_form=form,
        city=city,
        key_terms=[KeyTerm(term=t, summary=s, clause_ref=ref, confidence=0.9) for t, s, ref in key_terms],
        changes=[Change(target_document=td, target_clause=clause, action=a, affected_terms=terms, new_value_summary=s, confidence=0.9)
                 for td, clause, a, terms, s in changes],
        invalidates=[RelatedContract(number=n, date=d, relation="недействительно") for n, d in invalidates],
        related_contracts=[RelatedContract(number=n, relation="упомянут") for n in related],
    )


@pytest.fixture
def orion_cards():
    """Цепочка партнёрского договора ПД-2022/045: договор + 4 ДС, два из которых отменены последующими."""
    main = make_card("m", "ПД-2022/045", DocType.MAIN_CONTRACT, date="2022-05-12", category=SubjectCategory.JOINT_PROMOTION,
                     title="Договор", key_terms=[(TermKey.REMUNERATION, "45% от оплаченных начислений", "Приложение №1, п. 1.6"),
                                                 (TermKey.CONTACTS, "e-mail old@partner.example", "п. 14")], city="г. Томск")
    ds2 = make_card("ds2", "ПД-2022/045-ДС2", parent="ПД-2022/045", parent_date="2022-05-12", effective_from="2023-10-01",
                    category=SubjectCategory.JOINT_PROMOTION,
                    changes=[("ПД-2022/045", "Приложение №1, п. 1.6", ChangeAction.RESTATE, [TermKey.REMUNERATION], "55% от оплаченных начислений")])
    ds4 = make_card("ds4", "ПД-2022/045-ДС4", parent="ПД-2022/045", parent_date="2022-05-12", effective_from="2023-10-01",
                    category=SubjectCategory.JOINT_PROMOTION,
                    changes=[("ПД-2022/045", "Приложение №1, п. 1.6", ChangeAction.RESTATE, [TermKey.REMUNERATION], "55% от начислений (новая версия)"),
                             ("ПД-2022/045", "п. 14", ChangeAction.RESTATE, [TermKey.CONTACTS], "e-mail new@partner.example")],
                    invalidates=[("ПД-2022/045-ДС2", "2024-02-20")])
    stop1 = make_card("stop1", "ПД-2022/045-ДС5", parent="ПД-2022/045", parent_date="2022-05-12", category=SubjectCategory.JOINT_PROMOTION,
                      changes=[("ПД-2022/045", "Программа продвижения", ChangeAction.TERMINATE, [TermKey.TERMINATION], "прекращение привлечения абонентов с 15.04.2024")])
    stop2 = make_card("stop2", "ПД-2022/045-ДС6", parent="ПД-2022/045", parent_date="2022-05-12", category=SubjectCategory.JOINT_PROMOTION,
                      changes=[("ПД-2022/045", "Программа продвижения", ChangeAction.TERMINATE, [TermKey.TERMINATION], "прекращение привлечения абонентов с 15.04.2024")],
                      invalidates=[("ПД-2022/045-ДС5", "2024-03-01")])
    return [main, ds2, ds4, stop1, stop2]


@pytest.fixture
def settings(tmp_path, sample_dir):
    return Settings(input_dir=sample_dir, work_dir=tmp_path / "work")
