"""Tests for TestRegistry and @test decorator."""

from __future__ import annotations

import pytest

from agenteval.registry import RegisteredTest, TestRegistry
from agenteval.registry import test as agenteval_test
from agenteval.tracer import Tracer


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Reset the global registry before each test."""
    TestRegistry.reset()


class TestTestRegistry:
    def test_register_and_get(self) -> None:
        registry = TestRegistry.global_registry()

        async def my_test(tracer: Tracer) -> None:
            pass

        registry.register(RegisteredTest(fn=my_test, name="my_test", n=10, threshold=0.8, tags=[], module="test"))
        assert len(registry.get_all()) == 1

    def test_filter_by_tags(self) -> None:
        registry = TestRegistry.global_registry()

        async def fast_test(tracer: Tracer) -> None:
            pass

        async def slow_test(tracer: Tracer) -> None:
            pass

        registry.register(RegisteredTest(fn=fast_test, name="fast_test", n=5, threshold=0.8, tags=["fast"], module="test"))
        registry.register(RegisteredTest(fn=slow_test, name="slow_test", n=20, threshold=0.8, tags=["slow"], module="test"))

        fast = registry.get_all(tags=["fast"])
        assert len(fast) == 1
        assert fast[0].name == "fast_test"

    def test_clear(self) -> None:
        registry = TestRegistry.global_registry()

        async def t(tracer: Tracer) -> None:
            pass

        registry.register(RegisteredTest(fn=t, name="t", n=5, threshold=0.8, tags=[], module="test"))
        registry.clear()
        assert registry.get_all() == []

    def test_snapshot_is_copy(self) -> None:
        registry = TestRegistry.global_registry()

        async def t(tracer: Tracer) -> None:
            pass

        registry.register(RegisteredTest(fn=t, name="t", n=5, threshold=0.8, tags=[], module="test"))
        snap = registry.snapshot()
        registry.clear()
        assert len(snap) == 1


class TestTestDecorator:
    def test_bare_decorator(self) -> None:
        @agenteval_test
        async def my_test(tracer: Tracer) -> None:
            pass

        registry = TestRegistry.global_registry()
        entries = registry.get_all()
        assert len(entries) == 1
        assert entries[0].name == "my_test"
        assert entries[0].n == 20
        assert entries[0].threshold == 0.8

    def test_parameterized_decorator(self) -> None:
        @agenteval_test(n=5, threshold=0.9, tags=["integration"])
        async def my_test(tracer: Tracer) -> None:
            pass

        registry = TestRegistry.global_registry()
        entries = registry.get_all()
        assert len(entries) == 1
        assert entries[0].n == 5
        assert entries[0].threshold == 0.9
        assert entries[0].tags == ["integration"]

    def test_function_still_callable(self) -> None:
        import asyncio

        executed = []

        @agenteval_test(n=1)
        async def my_test(tracer: Tracer) -> None:
            executed.append(True)

        asyncio.run(my_test(Tracer()))
        assert executed == [True]

    def test_multiple_tests_registered(self) -> None:
        @agenteval_test
        async def test_a(tracer: Tracer) -> None:
            pass

        @agenteval_test
        async def test_b(tracer: Tracer) -> None:
            pass

        registry = TestRegistry.global_registry()
        assert len(registry.get_all()) == 2
