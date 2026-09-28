"""Офлайн-демо: весь конвейер на реальных документах без обращения к API.

Вместо модели работает ``ReplayLLM`` — «кассета» с заранее подготовленными ответами:
* ``extractions.json`` — карточки документов в формате ``DocumentExtraction`` (то, что вернула бы модель);
* ``ocr.json`` — расшифровка сканированных страниц;
* ``notes.json`` — аналитические заметки для отчёта.

Кассета проходит через те же инструменты, проверки (validators.py), связывание, свертку, кластеризацию
и сборку отчёта, что и живой прогон. Не воспроизводится только одно: решения модели о порядке шагов.
Здесь порядок фиксирован и совпадает с типичным планом из системного промпта.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

from .config import Settings
from .ingest import OCR_ONLY_FORMATS, UNREADABLE_FORMATS, DocumentLoader
from .knowledge_base import KnowledgeBase
from .llm import LLMError
from .schemas import DocumentExtraction, OcrPage, OcrResult
from .tools import ToolContext, analyst_registry
from .tracing import Tracer

NOT_IN_CASSETTE = "[страница не расшифрована в демо-кассете]"


class ReplayLLM:
    """Тот же интерфейс, что у LLMClient, но ответы берутся из кассеты."""

    def __init__(self, settings: Settings, cassette_dir: Path):
        self.settings = settings
        self.extractions = json.loads((cassette_dir / "extractions.json").read_text(encoding="utf-8"))
        ocr_path = cassette_dir / "ocr.json"
        self.ocr = json.loads(ocr_path.read_text(encoding="utf-8")) if ocr_path.exists() else {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "calls": 0}

    def structured(self, system, content, schema, max_tokens=None):
        if schema is not DocumentExtraction:
            raise LLMError(f"в кассете нет ответов для схемы {schema.__name__}")
        m = re.match(r"Файл: (.+)", content if isinstance(content, str) else "")
        path = m.group(1).strip() if m else None
        if path not in self.extractions:
            raise LLMError(f"в кассете нет карточки для {path}")
        self.usage["calls"] += 1
        return DocumentExtraction.model_validate(self.extractions[path])

    def ocr_pages(self, pages, doc_path=None):
        doc = self.ocr.get(doc_path or "", {})
        if not doc:
            raise LLMError(f"в кассете нет OCR для {doc_path}")
        self.usage["calls"] += 1
        out = []
        for num, _png in pages:
            p = doc.get(str(num))
            out.append(OcrPage(page=num, text=p["text"], legibility=p["legibility"], uncertain_fragments=p.get("uncertain_fragments", []))
                       if p else OcrPage(page=num, text=NOT_IN_CASSETTE, legibility="medium", uncertain_fragments=[]))
        return OcrResult(pages=out)

    def agent_turn(self, *args, **kwargs):
        raise LLMError("в демо-режиме агентский цикл не запускается: шаги выполняет сценарий")


def run_demo(settings: Settings, cassette_dir: Path, verbose: bool = True) -> dict:
    llm = ReplayLLM(settings, cassette_dir)
    expected = sorted(llm.extractions)
    missing = [rel for rel in expected if not (settings.input_dir / rel).exists()]
    if len(missing) == len(expected):
        raise FileNotFoundError(
            f"В {settings.input_dir.resolve()} нет документов тестового пакета. Демо воспроизводит ответы модели "
            f"по конкретным файлам: положите папки пакета как есть, например {settings.input_dir}/{expected[0]} "
            f"(или укажите --input). Всего ожидается {len(expected)} файлов.")
    for p in (settings.kb_path, settings.report_path, *settings.logs_dir.glob("demo_*.jsonl")):
        p.unlink(missing_ok=True)
    ctx = ToolContext(settings=settings, kb=KnowledgeBase(settings.kb_path), loader=DocumentLoader(settings, llm), llm=llm)
    reg = analyst_registry()
    tracer = Tracer(settings.logs_dir, "demo", verbose=verbose)
    tracer.start("demo (replay)", "Офлайн-демо: ответы модели из кассеты, шаги — по плану системного промпта", "replay")

    def call(name: str, **args) -> dict:
        tracer.tool_call(name, args)
        started = time.perf_counter()
        result, is_error = reg.execute(ctx, name, args)
        tracer.tool_result(name, result, is_error, time.perf_counter() - started)
        return result

    def think(text: str) -> None:
        tracer.next_step()
        tracer.thought(text)

    think("Мысль: начинаю с инвентаризации пакета.\nСледующий шаг: list_documents.")
    docs = call("list_documents")["documents"]

    formats = Counter(d["format"] for d in docs)
    think(f"Мысль: {len(docs)} файлов, по форматам: {dict(formats)}.\n"
          "Следующий шаг: parse_document — метод по формату (сканы и изображения — ocr); нераспознанные повторить через ocr.")
    recognized = []
    for d in docs:
        if d["format"] in UNREADABLE_FORMATS:
            call("parse_document", doc_id=d["doc_id"])  # зафиксирует документ как нераспознанный
            continue
        method = "ocr" if d["format"] in OCR_ONLY_FORMATS else "auto"
        res = call("parse_document", doc_id=d["doc_id"], method=method)
        if res.get("status") == "error" and "ocr" in (res.get("hint") or ""):
            res = call("parse_document", doc_id=d["doc_id"], method="ocr")
        if res.get("recognized"):
            recognized.append(d["doc_id"])

    think(f"Мысль: распознано {len(recognized)} из {len(docs)}.\nСледующий шаг: extract_document_card по каждому распознанному.")
    for doc_id in recognized:
        call("extract_document_card", doc_id=doc_id)

    think("Мысль: карточки готовы.\nСледующий шаг: link_contract_chains — связать ДС с договорами по номерам.")
    chains = call("link_contract_chains")
    contract_ids = [c["contract_id"] for c in chains["contracts"]]

    think(f"Мысль: {len(contract_ids)} договоров.\nСледующий шаг: consolidate_contract по каждому.")
    for cid in contract_ids:
        call("consolidate_contract", contract_id=cid)

    think("Следующий шаг: кластеризация договоров по предмету, затем по форме контрагента и городу; документов — по типу и области изменений.")
    by_subject = call("cluster_contracts", level="contract", dimensions=["subject_category"])
    call("cluster_contracts", level="contract", dimensions=["counterparty_legal_form", "city"])
    call("cluster_contracts", level="document", dimensions=["doc_type", "main_change_area"])

    for cl in by_subject["clusters"]:
        if cl["size"] > 1:
            think(f"Мысль: в кластере {cl['cluster_id']} {cl['size']} договора — сравниваю условия.")
            call("compare_contracts", cluster_id=cl["cluster_id"])

    notes_path = cassette_dir / "notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {"executive_summary": "—"}
    think("Следующий шаг: generate_report с заметками аналитика, затем check_plan_completion.")
    call("generate_report", **notes)
    verdict = call("check_plan_completion")
    tracer.reflection(verdict, "завершение" if verdict["complete"] else "есть невыполненные пункты")
    tracer.final("demo finished")
    return {"report": str(settings.report_path), "log": str(tracer.path), "complete": verdict["complete"], "todo": verdict["todo"]}
