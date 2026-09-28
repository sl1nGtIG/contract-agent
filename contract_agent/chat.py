"""Режим вопросов и ответов поверх базы знаний с памятью диалога.

Долговременная память — база знаний (карточки, цепочки, свертки). Кратковременная —
история диалога: сохраняется в workdir/chat/<session>.json (только тексты реплик,
без служебных блоков), поэтому разговор можно продолжить в следующем запуске.
"""
from __future__ import annotations

import json
from pathlib import Path

from .agent import Agent
from .prompts import CHAT_SYSTEM
from .tools import ToolContext, chat_registry
from .tracing import Tracer

MAX_HISTORY_TURNS = 20


class ChatSession:
    def __init__(self, ctx: ToolContext, session: str = "default", verbose: bool = True):
        self.ctx = ctx
        self.path = ctx.settings.work_dir / "chat" / f"{session}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        self.tracer = Tracer(ctx.settings.logs_dir, f"chat_{session}", verbose=verbose)
        self.agent = Agent(ctx, chat_registry(), CHAT_SYSTEM, self.tracer, reflection_gate=False)

    def ask(self, question: str) -> str:
        self.tracer.start("chat", question, self.ctx.settings.model)
        recent = self.history[-2 * MAX_HISTORY_TURNS :]
        result = self.agent.run(question, history=recent)
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": result.text}]
        self.path.write_text(json.dumps(self.history, ensure_ascii=False, indent=1), encoding="utf-8")
        return result.text

    def reset(self) -> None:
        self.history = []
        if self.path.exists():
            self.path.unlink()


def history_path(work_dir: Path, session: str) -> Path:
    return work_dir / "chat" / f"{session}.json"
