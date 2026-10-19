from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from projectpilot.tools import ToolRegistry, default_registry


@dataclass
class TraceStep:
    agent: str
    kind: str      # observe | task | decision | result
    message: str


class BaseAgent:
    name = "Agent"

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or default_registry
        self.reset()

    def reset(self) -> None:
        self.trace: list[TraceStep] = []
        self.tool_calls: Counter = Counter()

    def log(self, kind: str, message: str) -> None:
        self.trace.append(TraceStep(self.name, kind, message))

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        self.tool_calls[tool_name] += 1
        return self.registry.call(tool_name, **kwargs)