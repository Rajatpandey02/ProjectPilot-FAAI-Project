from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, func: Callable[..., Any]) -> None:
        # Overwrite silently so Streamlit's hot-reload never causes duplicate errors.
        self._tools[name] = Tool(name, description, func)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Available: {sorted(self._tools)}")
        return self._tools[name].func(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[tuple[str, str]]:
        return [(t.name, t.description) for t in self._tools.values()]


default_registry = ToolRegistry()


def tool(name: str, description: str):
    """Decorator that registers a function in the default registry."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        default_registry.register(name, description, func)
        return func
    return decorator