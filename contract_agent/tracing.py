"""Журнал хода мысли агента: консоль (для человека) + JSONL (для разбора и тестов).

Что пишется:
* thinking — сводка внутренних рассуждений модели (adaptive thinking, display=summarized);
* thought  — явный блок «Мысль / План / Следующий шаг», который модель пишет перед вызовами;
* tool_call / tool_result — какой инструмент, с какими аргументами, статус и краткий итог;
* reflection — результат проверки плана и решение продолжать/завершать;
* final / error / usage.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console(highlight=False)


class Tracer:
    def __init__(self, logs_dir: Path, run_name: str, verbose: bool = True):
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.verbose = verbose
        self.step = 0
        self._lock = threading.Lock()  # инструменты выполняются параллельно — запись в журнал должна быть атомарной

    def _write(self, event: str, **data) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "step": self.step, "event": event, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def start(self, mode: str, task: str, model: str) -> None:
        self._write("run_start", mode=mode, task=task, model=model)
        if self.verbose:
            console.print(Panel(task, title=f"[bold]{mode}[/] · {model}", border_style="cyan"))

    def next_step(self) -> None:
        self.step += 1

    def thinking(self, text: str) -> None:
        if not text.strip():
            return
        self._write("thinking", text=text)
        if self.verbose:
            console.print(f"[dim italic]  ∴ {text.strip()[:600]}[/]")

    def thought(self, text: str) -> None:
        if not text.strip():
            return
        self._write("thought", text=text)
        if self.verbose:
            console.print(Panel(text.strip(), title=f"шаг {self.step} · мысль", border_style="yellow", title_align="left"))

    def tool_call(self, name: str, args: dict) -> None:
        self._write("tool_call", tool=name, args=args)
        if self.verbose:
            short = json.dumps(args, ensure_ascii=False)
            console.print(f"  [bold blue]→ {name}[/]({short[:200]}{'…' if len(short) > 200 else ''})")

    def tool_result(self, name: str, result: dict, is_error: bool, seconds: float) -> None:
        status = result.get("status", "ok")
        self._write("tool_result", tool=name, status=status, is_error=is_error, seconds=round(seconds, 2), result=result)
        if self.verbose:
            color = {"ok": "green", "empty": "yellow", "error": "red"}.get(status, "white")
            detail = result.get("error") or result.get("message") or ""
            console.print(f"  [{color}]← {name}: {status}[/] [dim]{detail[:200]} ({seconds:.1f}s)[/]")

    def reflection(self, result: dict, action: str) -> None:
        self._write("reflection", result=result, action=action)
        if self.verbose:
            lines = [f"{'✓' if i['ok'] else '✗'} {i['title']}" + (f" — {i['detail']}" if i["detail"] and not i["ok"] else "")
                     for i in result["items"]]
            quality = [f"  {k}: {v}" for k, v in (result.get("quality") or {}).items()]
            body = "\n".join(lines) + ("\n\nКачество работы:\n" + "\n".join(quality) if quality else "") + f"\n\n→ {action}"
            console.print(Panel(body, title="reflection · проверка плана", border_style="magenta"))

    def final(self, text: str) -> None:
        self._write("final", text=text)

    def error(self, message: str) -> None:
        self._write("error", message=message)
        if self.verbose:
            console.print(f"[bold red]Ошибка:[/] {message}")

    def usage(self, usage: dict) -> None:
        self._write("usage", **usage)
        if self.verbose:
            console.print(f"[dim]Токены: вход {usage['input_tokens']:,}, выход {usage['output_tokens']:,}, "
                          f"из кеша {usage['cache_read_input_tokens']:,}, вызовов {usage['calls']}".replace(",", " ")
                          + (f", стоимость ${usage['cost_usd']:.2f}" if usage.get("cost_usd") else "") + "[/]")
