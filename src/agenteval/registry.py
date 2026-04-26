"""Global test registry and @agenteval.test decorator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class RegisteredTest:
    """Metadata for a test function registered with @agenteval.test."""

    fn: Callable[..., Any]
    name: str
    n: int
    threshold: float
    tags: list[str]
    module: str


class TestRegistry:
    """Singleton registry that collects all @agenteval.test-decorated functions."""

    _instance: Optional["TestRegistry"] = None

    def __init__(self) -> None:
        self._tests: list[RegisteredTest] = []

    @classmethod
    def global_registry(cls) -> "TestRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Replace the global registry with a fresh one. Used in tests."""
        cls._instance = cls()

    def register(self, entry: RegisteredTest) -> None:
        self._tests.append(entry)

    def clear(self) -> None:
        self._tests.clear()

    def get_all(self, tags: Optional[list[str]] = None) -> list[RegisteredTest]:
        """Return all registered tests, optionally filtered by tags."""
        if not tags:
            return list(self._tests)
        return [t for t in self._tests if any(tag in t.tags for tag in tags)]

    def snapshot(self) -> list[RegisteredTest]:
        """Return a copy of the current registered tests."""
        return list(self._tests)


def test(
    fn: Optional[Callable[..., Any]] = None,
    *,
    n: int = 20,
    threshold: float = 0.8,
    tags: Optional[list[str]] = None,
) -> Any:
    """Decorator that registers a test function with the global TestRegistry.

    Supports both bare and parameterized forms::

        @agenteval.test
        async def test_basic(tracer: Tracer) -> None: ...

        @agenteval.test(n=10, threshold=0.9, tags=["slow"])
        async def test_complex(tracer: Tracer) -> None: ...

    Args:
        fn: The test function (when used as bare decorator).
        n: Number of runs. Default: 20.
        threshold: Required pass rate. Default: 0.8.
        tags: Optional tags for filtering.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        registry = TestRegistry.global_registry()
        registry.register(
            RegisteredTest(
                fn=func,
                name=func.__name__,
                n=n,
                threshold=threshold,
                tags=tags or [],
                module=func.__module__,
            )
        )
        return func

    if fn is not None:
        # Called as @agenteval.test (no parentheses)
        return decorator(fn)

    # Called as @agenteval.test(...) (with parentheses)
    return decorator
