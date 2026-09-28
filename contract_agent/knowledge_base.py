"""База знаний агента — долговременная память между запусками.

Хранится одним JSON-файлом (workdir/knowledge_base.json): инвентарь, результаты
распознавания, карточки документов, цепочки договоров, свертки, кластеры и заметки
аналитика. Режим чата работает только поверх неё — без повторного чтения PDF.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .ingest import DocumentInfo
from .schemas import Cluster, ConsolidatedTerms, Contract, DocumentCard, ValidationIssue


class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.documents: dict[str, DocumentInfo] = {}
        self.parse_status: dict[str, dict] = {}       # doc_id -> {recognized, reason, quality, warnings}
        self.cards: dict[str, DocumentCard] = {}
        self.contracts: dict[str, Contract] = {}
        self.link_issues: list[ValidationIssue] = []
        self.consolidated: dict[str, ConsolidatedTerms] = {}
        self.clusters: dict[str, list[Cluster]] = {}  # "contract:subject_category" -> clusters
        self.notes: dict[str, dict] = {"executive_summary": {}, "clusters": {}, "contracts": {}}
        self.report_generated_at: str | None = None
        self.reflection: dict | None = None
        self.usage: dict = {}                          # расход токенов/стоимость последнего прогона
        self.changed_at: str | None = None             # когда последний раз менялись данные (для проверки свежести отчёта)

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "documents": {k: asdict(v) for k, v in self.documents.items()},
                "parse_status": self.parse_status,
                "cards": {k: v.model_dump(mode="json") for k, v in self.cards.items()},
                "contracts": {k: v.model_dump(mode="json") for k, v in self.contracts.items()},
                "link_issues": [i.model_dump(mode="json") for i in self.link_issues],
                "consolidated": {k: v.model_dump(mode="json") for k, v in self.consolidated.items()},
                "clusters": {k: [c.model_dump(mode="json") for c in v] for k, v in self.clusters.items()},
                "notes": self.notes,
                "report_generated_at": self.report_generated_at,
                "reflection": self.reflection,
                "usage": self.usage,
                "changed_at": self.changed_at,
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        kb = cls(path)
        if not path.exists():
            return kb
        data = json.loads(path.read_text(encoding="utf-8"))
        kb.documents = {k: DocumentInfo(**v) for k, v in data.get("documents", {}).items()}
        kb.parse_status = data.get("parse_status", {})
        kb.cards = {k: DocumentCard.model_validate(v) for k, v in data.get("cards", {}).items()}
        kb.contracts = {k: Contract.model_validate(v) for k, v in data.get("contracts", {}).items()}
        kb.link_issues = [ValidationIssue.model_validate(i) for i in data.get("link_issues", [])]
        kb.consolidated = {k: ConsolidatedTerms.model_validate(v) for k, v in data.get("consolidated", {}).items()}
        kb.clusters = {k: [Cluster.model_validate(c) for c in v] for k, v in data.get("clusters", {}).items()}
        kb.notes = data.get("notes", kb.notes)
        kb.report_generated_at = data.get("report_generated_at")
        kb.reflection = data.get("reflection")
        kb.usage = data.get("usage", {})
        kb.changed_at = data.get("changed_at")
        return kb

    # ------------------------------------------------------------------ mutations (thread-safe)
    def touch(self) -> None:
        """Отметить изменение данных: отчёт, построенный раньше, считается устаревшим."""
        self.changed_at = datetime.now().isoformat(timespec="microseconds")

    def put_card(self, card: DocumentCard) -> None:
        with self._lock:
            self.touch()
            self.cards[card.doc_id] = card
            # новые карточки делают устаревшими производные данные
            self.contracts, self.consolidated, self.clusters = {}, {}, {}
            self.save()

    def set_parse_status(self, doc_id: str, status: dict) -> None:
        with self._lock:
            self.parse_status[doc_id] = status
            self.save()

    # ------------------------------------------------------------------ queries
    def contract_of(self, doc_id: str) -> str | None:
        for cid, c in self.contracts.items():
            if c.main_doc_id == doc_id or doc_id in c.supplement_doc_ids:
                return cid
        return None

    def find_contract(self, query: str) -> Contract | None:
        """Поиск договора по номеру (в т.ч. номеру любого его соглашения) или по doc_id."""
        q = query.strip().upper()
        if q in self.contracts:
            return self.contracts[q]
        for card in self.cards.values():
            if (card.number.value or "").upper() == q or card.doc_id.upper() == q:
                cid = self.contract_of(card.doc_id)
                return self.contracts.get(cid) if cid else None
        return None

    def find_card(self, query: str) -> DocumentCard | None:
        q = query.strip()
        if q in self.cards:
            return self.cards[q]
        for card in self.cards.values():
            if (card.number.value or "").upper() == q.upper() or card.file == q:
                return card
        return None

    def all_rejected(self) -> list[ValidationIssue]:
        out = [i for c in self.cards.values() for i in c.rejected]
        out += [i for c in self.contracts.values() for i in c.issues if i.severity.value == "rejected"]
        return out
