"""Цикл агента (ReAct): модель рассуждает → вызывает инструменты → видит результат → …

Цикл написан вручную (а не через готовый runner), потому что нужны три вещи, которые
должны быть под контролем кода, а не модели:
1. журнал мыслей на каждом шаге (thinking + явный блок «Мысль/План/Следующий шаг»);
2. параллельное выполнение независимых вызовов с перехватом ошибок;
3. reflection-гейт: перед завершением код сам проверяет чек-лист плана и, если он не
   выполнен, возвращает агента к работе с перечнем недоделанного.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .llm import LLMError
from .reflection import check_plan
from .tools import ToolContext, ToolRegistry, dumps
from .tracing import Tracer


@dataclass
class AgentResult:
    text: str
    steps: int
    completed: bool                    # цикл завершился штатно (не лимит шагов и не ошибка API)
    messages: list
    plan_complete: bool | None = None  # итог reflection-чек-листа (None — режим без reflection, например чат)


class Agent:
    def __init__(self, ctx: ToolContext, registry: ToolRegistry, system: str, tracer: Tracer, reflection_gate: bool):
        if ctx.llm is None:
            raise LLMError("Агенту нужен LLM-клиент (ANTHROPIC_API_KEY)")
        self.ctx = ctx
        self.registry = registry
        self.system = system
        self.tracer = tracer
        self.reflection_gate = reflection_gate
        self.settings = ctx.settings

    def run(self, task: str, history: list | None = None) -> AgentResult:
        messages: list = list(history or []) + [{"role": "user", "content": task}]
        tools = self.registry.api_definitions()
        reflections = 0

        for step in range(1, self.settings.max_agent_steps + 1):
            self.tracer.next_step()
            try:
                response = self.ctx.llm.agent_turn(self.system, messages, tools)
            except LLMError as exc:
                self.tracer.error(str(exc))
                return AgentResult(f"Работа остановлена: {exc}", step, False, messages)

            self._log_reasoning(response)
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                final_text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                if self.reflection_gate:
                    verdict = check_plan(self.ctx.kb, self.settings.report_path)
                    if not verdict["complete"] and reflections < self.settings.max_reflection_rounds:
                        reflections += 1
                        self.tracer.reflection(verdict, f"план не выполнен — возвращаю агента к работе (раунд {reflections})")
                        messages.append({"role": "user", "content": self._reflection_prompt(verdict)})
                        continue
                    self.tracer.reflection(verdict, "завершение" if verdict["complete"] else "лимит раундов исчерпан — завершаю с пометкой")
                    if not verdict["complete"]:
                        final_text += "\n\n⚠️ Не все пункты плана выполнены: " + "; ".join(verdict["todo"])
                    self.tracer.final(final_text)
                    return AgentResult(final_text, step, True, messages, plan_complete=verdict["complete"])
                self.tracer.final(final_text)
                return AgentResult(final_text, step, True, messages)

            results = self._execute(tool_uses)
            messages.append({"role": "user", "content": results})

        msg = f"Достигнут лимит шагов ({self.settings.max_agent_steps}); работа не завершена"
        self.tracer.error(msg)
        return AgentResult(msg, self.settings.max_agent_steps, False, messages)

    # ------------------------------------------------------------------ helpers
    def _log_reasoning(self, response) -> None:
        for block in response.content:
            if block.type == "thinking":
                self.tracer.thinking(getattr(block, "thinking", "") or "")
            elif block.type == "text":
                self.tracer.thought(block.text)

    def _execute(self, tool_uses: list) -> list[dict]:
        def run_one(block) -> dict:
            self.tracer.tool_call(block.name, block.input)
            started = time.perf_counter()
            result, is_error = self.registry.execute(self.ctx, block.name, block.input)
            self.tracer.tool_result(block.name, result, is_error, time.perf_counter() - started)
            out = {"type": "tool_result", "tool_use_id": block.id, "content": dumps(result)}
            if is_error:
                out["is_error"] = True
            return out

        if len(tool_uses) == 1:
            return [run_one(tool_uses[0])]
        with ThreadPoolExecutor(max_workers=self.settings.max_parallel_tools) as pool:
            return list(pool.map(run_one, tool_uses))  # порядок результатов = порядок вызовов

    @staticmethod
    def _reflection_prompt(verdict: dict) -> str:
        todo = "\n".join(f"- {t}" for t in verdict["todo"])
        return (
            "Reflection: автоматическая проверка плана показала, что цель ещё не достигнута.\n"
            f"Не выполнено:\n{todo}\n\n"
            "Доделай эти пункты (если пункт невыполним — например, документ не распознаётся даже через OCR — "
            "явно зафиксируй это в заметках отчёта), затем снова вызови generate_report и check_plan_completion."
        )
