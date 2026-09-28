"""Схема данных: что извлекаем из документов и как храним в базе знаний.

Два слоя моделей:
* ``*Extraction`` — то, что возвращает LLM (structured output). Все поля обязательные,
  отсутствие значения кодируется ``null`` — так схема однозначна для модели.
* ``DocumentCard``, ``Contract``, ``ConsolidatedTerms`` … — провалидированные данные в базе
  знаний. Сюда попадают только значения, прошедшие детерминированные проверки; всё, что
  проверки не прошло, лежит в ``rejected`` с причиной отказа.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Справочники
# ---------------------------------------------------------------------------
class DocType(str, Enum):
    MAIN_CONTRACT = "main_contract"                  # основной договор
    SUPPLEMENTARY = "supplementary_agreement"        # доп. соглашение к договору / к другому соглашению
    STANDALONE_AGREEMENT = "standalone_agreement"    # самостоятельное соглашение (своя нумерация, ссылается на другие договоры)
    OTHER = "other"


class SubjectCategory(str, Enum):
    COMMERCIAL_REPRESENTATION = "commercial_representation"  # коммерческое представительство / агентские продажи
    JOINT_PROMOTION = "joint_promotion"                      # партнёрство, совместное продвижение услуг
    AGENCY_ORDER = "agency_order"                            # поручение (приём платежей и т.п.)
    INCENTIVE_PROGRAM = "incentive_program"                  # планы продаж, бонусы, мотивация
    SERVICES = "services"                                    # оказание услуг
    LEASE = "lease"                                          # аренда
    SALE = "sale"                                            # купля-продажа / поставка
    OTHER = "other"


class LegalForm(str, Enum):
    OOO = "ООО"
    IP = "ИП"
    AO = "АО"
    PAO = "ПАО"
    OTHER = "other"
    UNKNOWN = "unknown"


class TermKey(str, Enum):
    """Фиксированный словарь «условий договора» — по нему строится свертка и сравнение."""
    SUBJECT = "subject"                        # предмет договора
    TERM = "term"                              # срок действия, пролонгация
    REMUNERATION = "remuneration"              # вознаграждение, ставки, цена
    PAYMENT_PROCEDURE = "payment_procedure"    # порядок и сроки расчётов
    SALES_PLANS = "sales_plans"                # плановые показатели, KPI
    OBLIGATIONS = "obligations"                # ключевые обязанности сторон
    LIABILITY = "liability"                    # ответственность, штрафы
    TERMINATION = "termination"                # расторжение, прекращение обязательств, консервация
    TERRITORY = "territory"                    # территория, торговые точки, каналы
    REPORTING = "reporting"                    # отчётность, акты
    DEFINITIONS = "definitions"                # ключевые определения (тарифные группы и т.п.)
    CONTACTS = "contacts"                      # реквизиты, каналы уведомлений
    PERSONAL_DATA = "personal_data"            # обработка персональных данных
    OTHER = "other"


class ChangeAction(str, Enum):
    RESTATE = "restate"                          # «изложить в следующей редакции»
    ADD = "add"                                  # дополнить
    DELETE = "delete"                            # исключить
    NEW_OBLIGATION = "new_obligation"            # новые обязательства / поручение без правки текста договора
    TERMINATE = "terminate"                      # прекращение обязательств / расторжение
    SUSPEND = "suspend"                          # консервация / приостановление
    INVALIDATE_AGREEMENT = "invalidate_agreement"  # признать ранее подписанное соглашение недействительным
    OTHER = "other"


# ---------------------------------------------------------------------------
# Что возвращает LLM
# ---------------------------------------------------------------------------
class EvidencedValue(BaseModel):
    value: str | None = Field(description="Нормализованное значение; null, если в тексте его нет или оно обезличено")
    quote: str | None = Field(description="Дословная цитата из текста (до 300 символов), на которой основано значение")
    confidence: float = Field(description="Уверенность от 0 до 1")


class KeyTermExtraction(BaseModel):
    term: TermKey
    summary: str = Field(description="Суть условия своими словами, 1–3 предложения, с конкретными числами")
    clause_ref: str | None = Field(description="Ссылка на пункт: 'п. 4.1', 'Приложение №7, табл. 1'")
    quote: str | None = Field(description="Дословная цитата из текста, подтверждающая условие")
    confidence: float = Field(description="Уверенность от 0 до 1")


class ChangeExtraction(BaseModel):
    target_document: str | None = Field(
        description="Номер документа, который изменяется (договор или ранее заключённое соглашение), например АР-2021/015"
    )
    target_clause: str = Field(description="Что меняется: 'п. 1.25', 'Приложение №7, табл. 1', 'договор целиком'")
    action: ChangeAction
    affected_terms: list[TermKey]
    new_value_summary: str = Field(description="Новая редакция условия кратко, с конкретными числами и датами")
    effective_from: str | None = Field(description="С какой даты действует изменение, ДД.ММ.ГГГГ, либо null")
    quote: str | None = Field(description="Дословная цитата из текста соглашения")
    confidence: float = Field(description="Уверенность от 0 до 1")


class RelatedContractExtraction(BaseModel):
    number: str
    date: str | None = Field(description="ДД.ММ.ГГГГ или null")
    title: str | None = Field(description="Как документ назван в тексте, например 'Договор «Кредитование»'")
    relation: str = Field(description="Роль связи: 'основной договор', 'изменяемое соглашение', 'упомянут в перечне' и т.п.")


class DocumentExtraction(BaseModel):
    doc_type: DocType
    title: str = Field(description="Вид документа как в заголовке: 'Договор коммерческого представительства', 'Дополнительное соглашение'")
    number: EvidencedValue = Field(description="Номер самого документа, например АР-2021/015-ДС1")
    date: EvidencedValue = Field(description="Дата заключения/подписания документа, ДД.ММ.ГГГГ")
    city: str | None
    parent_number: EvidencedValue = Field(description="Для доп. соглашения — номер договора/соглашения, к которому оно заключено")
    parent_date: EvidencedValue = Field(description="Дата родительского договора так, как она указана В ЭТОМ документе, ДД.ММ.ГГГГ")
    parent_title: str | None = Field(description="Вид родительского договора: 'Договор коммерческого представительства' и т.п.")
    counterparty: EvidencedValue = Field(
        description="Наименование контрагента — стороны, противоположной «нашей» (null, если обезличено); "
        "quote должна содержать это наименование"
    )
    counterparty_legal_form: LegalForm
    signatory_branch: str | None = Field(description="Филиал или подразделение подписанта, если указан")
    subject_category: SubjectCategory
    subject_summary: str = Field(description="Предмет документа, 1–2 предложения")
    effective_from: EvidencedValue = Field(description="Дата вступления в силу или начала распространения действия, ДД.ММ.ГГГГ")
    valid_until: EvidencedValue = Field(
        description="Дата окончания действия документа, если она прямо указана (ДД.ММ.ГГГГ); null — если бессрочно, "
        "с автопролонгацией или «в пределах срока действия Договора»"
    )
    key_terms: list[KeyTermExtraction] = Field(description="Существенные условия, которые ЭТОТ документ устанавливает")
    changes: list[ChangeExtraction] = Field(description="Для соглашений: какие изменения вносятся в договор/другие соглашения")
    invalidates: list[RelatedContractExtraction] = Field(description="Соглашения, которые этот документ признаёт недействительными")
    related_contracts: list[RelatedContractExtraction] = Field(description="Другие договоры/соглашения, упомянутые в документе")
    anonymized_fields: list[str] = Field(description="Какие сведения в тексте обезличены (вымараны)")
    extraction_notes: list[str] = Field(description="Сомнения, противоречия внутри текста, нечитаемые места")


class OcrPage(BaseModel):
    page: int
    text: str = Field(description="Дословная расшифровка страницы; нечитаемое помечать [неразборчиво]")
    legibility: str = Field(description="high | medium | low")
    uncertain_fragments: list[str] = Field(description="Фрагменты, в прочтении которых нет уверенности (включая рукописные)")


class OcrResult(BaseModel):
    pages: list[OcrPage]


# ---------------------------------------------------------------------------
# Что хранится в базе знаний
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    REJECTED = "rejected"   # значение НЕ записано в базу
    WARNING = "warning"     # значение записано, но требует внимания
    INFO = "info"


class ValidationIssue(BaseModel):
    field: str
    severity: Severity
    reason: str
    raw_value: Any = None
    doc_id: str | None = None


class VerifiedValue(BaseModel):
    value: str | None = None
    quote: str | None = None
    confidence: float = 0.0
    source: str = "llm"   # llm | derived | cross_document


class KeyTerm(BaseModel):
    term: TermKey
    summary: str
    clause_ref: str | None = None
    quote: str | None = None
    confidence: float = 0.0


class Change(BaseModel):
    target_document: str | None = None
    target_clause: str
    action: ChangeAction
    affected_terms: list[TermKey] = []
    new_value_summary: str
    effective_from: str | None = None   # ISO YYYY-MM-DD
    quote: str | None = None
    confidence: float = 0.0


class RelatedContract(BaseModel):
    number: str
    date: str | None = None   # ISO
    title: str | None = None
    relation: str = ""


class DocumentCard(BaseModel):
    doc_id: str
    file: str
    folder: str
    text_source: str = "text_layer"   # text_layer | ocr | mixed
    doc_type: DocType = DocType.OTHER
    title: str = ""
    number: VerifiedValue = VerifiedValue()
    date: VerifiedValue = VerifiedValue()          # ISO YYYY-MM-DD
    city: str | None = None
    parent_number: VerifiedValue = VerifiedValue()
    parent_date: VerifiedValue = VerifiedValue()   # ISO
    parent_title: str | None = None
    counterparty: VerifiedValue = VerifiedValue()
    counterparty_legal_form: LegalForm = LegalForm.UNKNOWN
    signatory_branch: str | None = None
    subject_category: SubjectCategory = SubjectCategory.OTHER
    subject_summary: str = ""
    effective_from: VerifiedValue = VerifiedValue()  # ISO
    valid_until: VerifiedValue = VerifiedValue()     # ISO, None — бессрочно
    key_terms: list[KeyTerm] = []
    changes: list[Change] = []
    invalidates: list[RelatedContract] = []
    related_contracts: list[RelatedContract] = []
    anonymized_fields: list[str] = []
    extraction_notes: list[str] = []
    issues: list[ValidationIssue] = []
    status: str = "ok"   # ok | partial | unrecognized

    @property
    def rejected(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.REJECTED]


class TimelineEntry(BaseModel):
    doc_id: str
    number: str | None
    date: str | None
    effective_from: str | None
    valid_until: str | None = None
    doc_type: DocType
    summary: str
    valid: bool = True
    invalid_reason: str | None = None


class Contract(BaseModel):
    contract_id: str                    # номер основного договора (или соглашения)
    title: str | None = None
    date: VerifiedValue = VerifiedValue()
    main_doc_id: str | None = None      # None -> основной договор в пакете отсутствует
    supplement_doc_ids: list[str] = []
    counterparty: str | None = None
    counterparty_legal_form: LegalForm = LegalForm.UNKNOWN
    city: str | None = None
    subject_category: SubjectCategory = SubjectCategory.OTHER
    subject_summary: str = ""
    folders: list[str] = []
    related_contract_ids: list[str] = []
    timeline: list[TimelineEntry] = []
    status: str = "active"              # active | partially_terminated | terminated | suspended
    issues: list[ValidationIssue] = []


class TermVersion(BaseModel):
    summary: str
    source_doc_id: str
    source_number: str | None
    clause_ref: str | None = None
    action: str = "base"
    effective_from: str | None = None
    replaced_by: str | None = None   # номер документа, заменившего/исключившего положение
    amended_by: list[str] = []       # частичные изменения отдельных пунктов положения


class TermState(BaseModel):
    term: TermKey
    active: list[TermVersion] = []    # действующие положения после применения всех соглашений
    history: list[TermVersion] = []   # все версии в хронологическом порядке (эволюция условия)


class ConsolidatedTerms(BaseModel):
    contract_id: str
    as_of: str | None = None
    status: str = "active"
    terms: dict[str, TermState] = {}
    applied_docs: list[str] = []
    skipped_docs: list[dict] = []
    notes: list[str] = []


class Cluster(BaseModel):
    cluster_id: str
    level: str                     # contract | document
    key: dict[str, str]
    members: list[str]
    common: dict[str, str] = {}
    differing: dict[str, dict[str, str]] = {}
