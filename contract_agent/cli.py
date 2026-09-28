"""Точка входа: python -m contract_agent <команда>."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil
import sys
from pathlib import Path

from rich.markdown import Markdown

from .agent import Agent
from .config import load_settings
from .ingest import DocumentLoader
from .knowledge_base import KnowledgeBase
from .llm import LLMError
from .providers import make_llm
from .prompts import ANALYST_SYSTEM
from .reflection import check_plan
from .tools import ToolContext, analyst_registry
from .tracing import Tracer, console

DEFAULT_TASK = (
    "Проанализируй пакет договорных документов во входной папке. Нужно: распознать все документы; "
    "кластеризовать договоры по предмету и другим значимым параметрам и описать различия внутри кластеров; "
    "собрать цепочки «договор → доп. соглашения» и для каждого договора сделать свертку актуальных условий "
    "с описанием их эволюции; сформировать Markdown-отчёт и проверить, что план выполнен."
)


def _context(args, need_llm: bool) -> ToolContext:
    settings = load_settings(
        input_dir=Path(args.input) if getattr(args, "input", None) else None,
        work_dir=Path(args.workdir) if getattr(args, "workdir", None) else None,
        model=getattr(args, "model", None),
    )
    if getattr(args, "fresh", False) and settings.work_dir.exists():
        # кеш распознанного текста оставляем: OCR дорогой и от анализа не зависит
        for p in (settings.kb_path, settings.report_path):
            p.unlink(missing_ok=True)
        shutil.rmtree(settings.work_dir / "chat", ignore_errors=True)
    llm = make_llm(settings) if need_llm else None
    kb = KnowledgeBase.load(settings.kb_path)
    return ToolContext(settings=settings, kb=kb, loader=DocumentLoader(settings, llm), llm=llm)


def cmd_inventory(args) -> int:
    ctx = _context(args, need_llm=False)
    docs = ctx.loader.inventory()
    for d in docs:
        console.print(f"{d.doc_id}  {d.format:<11} {d.pages:>3} стр.  {d.path}")
    by_format = {f: sum(d.format == f for d in docs) for f in sorted({d.format for d in docs})}
    console.print(f"\nВсего: {len(docs)}; по форматам: {by_format}")
    return 0


def cmd_analyze(args) -> int:
    ctx = _context(args, need_llm=True)
    tracer = Tracer(ctx.settings.logs_dir, "analyze", verbose=not args.quiet)
    task = args.task or DEFAULT_TASK
    tracer.start("analyze", task, ctx.settings.model)
    agent = Agent(ctx, analyst_registry(), ANALYST_SYSTEM, tracer, reflection_gate=True)
    result = agent.run(task)
    tracer.usage(ctx.llm.usage)
    console.rule("Итог агента")
    console.print(Markdown(result.text))
    console.print(f"\nОтчёт: [bold]{ctx.settings.report_path}[/]\nЖурнал мыслей: {tracer.path}\nБаза знаний: {ctx.settings.kb_path}")
    # код возврата 0 — только если цикл завершился штатно И reflection подтвердил выполнение плана
    return 0 if result.completed and result.plan_complete else 1


def cmd_chat(args) -> int:
    from .chat import ChatSession

    ctx = _context(args, need_llm=True)
    if not ctx.kb.contracts:
        console.print("[red]База знаний пуста — сначала выполните: python -m contract_agent analyze[/]")
        return 1
    session = ChatSession(ctx, args.session, verbose=not args.quiet)
    if args.question:
        console.print(Markdown(session.ask(" ".join(args.question))))
        return 0
    console.print("Вопросы по пакету договоров. Команды: /reset — очистить историю, /exit — выход.")
    while True:
        try:
            q = console.input("[bold cyan]Вы:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in {"/exit", "/quit"}:
            break
        if q == "/reset":
            session.reset()
            console.print("История очищена.")
            continue
        console.print(Markdown(session.ask(q)))
    return 0


def cmd_report(args) -> int:
    from .tools import check_plan_completion, cluster_contracts, consolidate_contract, link_contract_chains, write_reports

    ctx = _context(args, need_llm=False)
    if not ctx.kb.cards:
        console.print("[red]База знаний пуста[/]")
        return 1
    if args.revalidate:
        # новые правила проверки к уже извлечённым карточкам: текст документов берётся из кеша, модель не вызывается
        from .validators import revalidate_card

        for doc_id, card in list(ctx.kb.cards.items()):
            info = ctx.kb.documents[doc_id]
            method = ctx.kb.parse_status.get(doc_id, {}).get("method") or (
                "ocr" if ctx.kb.parse_status.get(doc_id, {}).get("force_ocr") else "auto")
            try:
                parsed = ctx.loader.parse(info, method=method)
            except Exception as exc:
                console.print(f"[yellow]{info.file}: текст недоступен ({exc}) — карточка не перепроверена[/]")
                continue
            ctx.kb.cards[doc_id] = revalidate_card(card, info, parsed, ctx.settings.min_confidence, ctx.settings.min_quote_match,
                                                   our_party=ctx.settings.our_party)
        ctx.kb.save()
        args.rebuild = True
    if args.rebuild:
        # детерминированная часть заново: цепочки, свертки, кластеры (с теми же измерениями); заметки агента сохраняются
        cluster_keys = list(ctx.kb.clusters)
        link_contract_chains(ctx)
        for cid in list(ctx.kb.contracts):
            consolidate_contract(ctx, cid)
        for key in cluster_keys:
            level, dims = key.split(":", 1)
            cluster_contracts(ctx, level=level, dimensions=dims.split(","))
    write_reports(ctx)
    ctx.kb.report_generated_at = datetime.now().isoformat(timespec="microseconds")  # пересборка — это новая генерация
    ctx.kb.save()
    verdict = check_plan_completion(ctx)
    console.print(f"Отчёты пересобраны: {ctx.settings.report_path} (+ report_full.md); план выполнен: {verdict['complete']}")
    for t in verdict["todo"]:
        console.print(f"  [yellow]не выполнено:[/] {t}")
    return 0


def cmd_check(args) -> int:
    ctx = _context(args, need_llm=False)
    verdict = check_plan(ctx.kb, ctx.settings.report_path)
    for item in verdict["items"]:
        console.print(f"{'✓' if item['ok'] else '✗'} {item['title']}" + (f" — {item['detail']}" if item["detail"] else ""))
    return 0 if verdict["complete"] else 1


def cmd_demo(args) -> int:
    from .demo import run_demo

    examples = Path(__file__).resolve().parents[1] / "examples"
    gold = json.loads((examples / "synthetic_package" / "gold.json").read_text(encoding="utf-8"))
    settings = load_settings(
        input_dir=Path(args.input) if args.input else examples / "synthetic_package" / "docs",
        work_dir=Path(args.workdir) if args.workdir else examples / "demo" / "output",
        as_of=args.as_of or gold["as_of"],
        our_party=gold["our_party"],
    )
    result = run_demo(settings, Path(args.cassette) if args.cassette else examples / "demo" / "cassette", verbose=not args.quiet)
    console.rule("Демо завершено")
    console.print(f"Отчёт: [bold]{result['report']}[/]\nЖурнал: {result['log']}\nПлан выполнен: {result['complete']}")
    for t in result["todo"]:
        console.print(f"  [yellow]не выполнено:[/] {t}")
    return 0 if result["complete"] else 1


def cmd_evaluate(args) -> int:
    from .evaluation import run

    root = Path(__file__).resolve().parents[1] / "examples"
    settings = load_settings(work_dir=Path(args.workdir) if args.workdir else None)
    kb_path = Path(args.kb) if args.kb else settings.kb_path
    result = run(kb_path, Path(args.gold) if args.gold else root / "synthetic_package" / "gold.json",
                 Path(args.out) if args.out else kb_path.with_name("evaluation.md"))
    t, ch = result["totals"], result["changes"]
    console.print(f"Документов: {result['documents']}; поля: correct {t['correct']}, пусто в обоих {t['empty']}, "
                  f"wrong {t['wrong']}, extra {t['extra']}, refused {t['refused']}, missed {t['missed']}")
    console.print(f"Изменения: найдено {ch['matched']} из {ch['gold']}, записано {ch['extracted']}; "
                  f"цепочки верно: {result['structure']['chains_correct']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="contract_agent", description="Агент-аналитик цепочек и кластеров договоров")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", help="папка с документами (по умолчанию ./data или CONTRACT_AGENT_INPUT)")
    common.add_argument("--workdir", help="рабочая папка: база знаний, отчёт, журналы (по умолчанию ./workdir)")
    common.add_argument("--model", help="модель Claude (по умолчанию claude-opus-5 или CONTRACT_AGENT_MODEL)")
    common.add_argument("-q", "--quiet", action="store_true", help="не печатать ход мысли в консоль (журнал пишется всегда)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory", parents=[common], help="список документов (без LLM)").set_defaults(func=cmd_inventory)
    a = sub.add_parser("analyze", parents=[common], help="полный анализ пакета агентом")
    a.add_argument("--task", help="свой текст задачи для агента")
    a.add_argument("--fresh", action="store_true", help="начать с чистой базы знаний (кеш OCR/текста сохраняется)")
    a.set_defaults(func=cmd_analyze)
    c = sub.add_parser("chat", parents=[common], help="вопросы по базе знаний (интерактивно или одним вопросом)")
    c.add_argument("question", nargs="*", help="вопрос; без аргумента — интерактивный режим")
    c.add_argument("--session", default="default", help="имя сессии диалога (память между запусками)")
    c.set_defaults(func=cmd_chat)
    d = sub.add_parser("demo", parents=[common], help="офлайн-демо на записанных ответах модели (без API)")
    d.add_argument("--cassette", help="папка кассеты (по умолчанию examples/demo/cassette)")
    d.add_argument("--as-of", help="дата, на которую строится свертка (ISO; по умолчанию — из эталона пакета)")
    d.set_defaults(func=cmd_demo)
    e = sub.add_parser("evaluate", parents=[common], help="сравнить базу знаний с размеченным эталоном (без LLM)")
    e.add_argument("--kb", help="база знаний (по умолчанию workdir/knowledge_base.json)")
    e.add_argument("--gold", help="эталон (по умолчанию examples/synthetic_package/gold.json)")
    e.add_argument("--out", help="куда записать отчёт об оценке (по умолчанию рядом с базой знаний)")
    e.set_defaults(func=cmd_evaluate)
    r = sub.add_parser("report", parents=[common], help="пересобрать отчёт из базы знаний (без LLM)")
    r.add_argument("--rebuild", action="store_true", help="заново построить цепочки, свертки и кластеры из карточек")
    r.add_argument("--revalidate", action="store_true",
                   help="перепроверить карточки текущими правилами (нужен кеш текста в workdir/cache), затем --rebuild")
    r.set_defaults(func=cmd_report)
    sub.add_parser("check", parents=[common], help="проверка выполнения плана (без LLM)").set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, FileNotFoundError) as exc:
        console.print(f"[bold red]Ошибка:[/] {exc}")
        return 2
