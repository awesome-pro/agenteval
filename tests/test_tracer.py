"""Tests for the Tracer class."""

from __future__ import annotations

import asyncio
import time

import pytest

from agenteval.tracer import Tracer, _ACTIVE_TRACER


def make_sync_tool(name: str = "search") -> tuple[list[str], callable]:
    calls: list[str] = []

    def tool(query: str) -> str:
        calls.append(query)
        return f"results for {query}"

    tool.__name__ = name
    return calls, tool


async def make_async_tool(name: str = "fetch") -> tuple[list[str], callable]:
    calls: list[str] = []

    async def tool(url: str) -> str:
        calls.append(url)
        return f"content of {url}"

    tool.__name__ = name
    return calls, tool


class TestTracerWrap:
    def test_wrap_sync_records_call(self) -> None:
        tracer = Tracer()
        _, raw = make_sync_tool("search")
        wrapped = tracer.wrap(raw)
        result = wrapped(query="python")
        assert result == "results for python"
        assert len(tracer._tool_calls) == 1
        tc = tracer._tool_calls[0]
        assert tc.name == "search"
        assert tc.arguments == {"query": "python"}
        assert tc.result == "results for python"
        assert tc.error is None
        assert tc.duration_seconds >= 0.0

    async def test_wrap_async_records_call(self) -> None:
        tracer = Tracer()

        async def fetch(url: str) -> str:
            return f"data from {url}"

        wrapped = tracer.wrap(fetch)
        result = await wrapped(url="http://example.com")
        assert result == "data from http://example.com"
        assert len(tracer._tool_calls) == 1
        tc = tracer._tool_calls[0]
        assert tc.name == "fetch"
        assert tc.error is None

    def test_wrap_custom_name(self) -> None:
        tracer = Tracer()
        _, raw = make_sync_tool("search")
        wrapped = tracer.wrap(raw, name="web_search")
        wrapped(query="test")
        assert tracer._tool_calls[0].name == "web_search"

    def test_wrap_records_error(self) -> None:
        tracer = Tracer()

        def broken(x: int) -> int:
            raise ValueError("boom")

        wrapped = tracer.wrap(broken)
        with pytest.raises(ValueError):
            wrapped(x=1)

        assert len(tracer._tool_calls) == 1
        assert tracer._tool_calls[0].error is not None
        assert "ValueError" in tracer._tool_calls[0].error

    async def test_wrap_async_records_error(self) -> None:
        tracer = Tracer()

        async def broken() -> None:
            raise RuntimeError("async boom")

        wrapped = tracer.wrap(broken)
        with pytest.raises(RuntimeError):
            await wrapped()

        assert tracer._tool_calls[0].error is not None


class TestTracerToolDecorator:
    def test_decorator_no_args(self) -> None:
        tracer = Tracer()

        @tracer.tool
        def calculator(x: int, y: int) -> int:
            return x + y

        assert calculator(x=1, y=2) == 3
        assert tracer._tool_calls[0].name == "calculator"

    def test_decorator_with_name(self) -> None:
        tracer = Tracer()

        @tracer.tool(name="math_op")
        def calculator(x: int, y: int) -> int:
            return x + y

        calculator(x=1, y=2)
        assert tracer._tool_calls[0].name == "math_op"


class TestRunContext:
    async def test_async_context_manager_sets_active_tracer(self) -> None:
        tracer = Tracer()
        async with tracer.run(input="hello") as run:
            assert Tracer.current() is tracer
        assert Tracer.current() is None

    def test_sync_context_manager(self) -> None:
        tracer = Tracer()
        with tracer.run(input="sync input") as run:
            run.set_output("sync output")
        trace = tracer.build_trace()
        assert trace.input == "sync input"
        assert trace.output == "sync output"

    async def test_set_output(self) -> None:
        tracer = Tracer()
        async with tracer.run(input="q") as run:
            run.set_output("answer")
        assert tracer._run_output == "answer"

    async def test_set_token_usage(self) -> None:
        tracer = Tracer()
        async with tracer.run(input="q") as run:
            run.set_token_usage({"input_tokens": 100, "output_tokens": 50})
        assert tracer._token_usage == {"input_tokens": 100, "output_tokens": 50}

    async def test_set_steps(self) -> None:
        tracer = Tracer()
        async with tracer.run(input="q") as run:
            run.set_steps(7)
        assert tracer._total_steps == 7

    async def test_add_metadata(self) -> None:
        tracer = Tracer()
        async with tracer.run(input="q") as run:
            run.add_metadata(model="gpt-4", version="1.0")
        assert tracer._metadata["model"] == "gpt-4"

    async def test_captures_agent_exception(self) -> None:
        tracer = Tracer()
        with pytest.raises(RuntimeError):
            async with tracer.run(input="q") as run:
                raise RuntimeError("agent crashed")
        assert tracer._run_error is not None
        assert "RuntimeError" in tracer._run_error

    async def test_does_not_capture_assertion_error(self) -> None:
        """AssertionErrors from .check() should not be stored as agent errors."""
        tracer = Tracer()
        with pytest.raises(AssertionError):
            async with tracer.run(input="q"):
                raise AssertionError("test failed")
        assert tracer._run_error is None

    async def test_duration_is_recorded(self) -> None:
        tracer = Tracer()
        async with tracer.run(input="q"):
            await asyncio.sleep(0.01)
        trace = tracer.build_trace()
        assert trace.duration_seconds >= 0.01


class TestBuildTrace:
    def test_build_trace_passed_true_when_clean(self) -> None:
        tracer = Tracer()
        with tracer.run(input="x") as run:
            run.set_output("y")
        trace = tracer.build_trace()
        assert trace.passed is True
        assert trace.error is None
        assert trace.assertion_errors == []

    def test_build_trace_passed_false_when_error(self) -> None:
        tracer = Tracer()
        tracer._run_error = "ValueError: something went wrong"
        trace = tracer.build_trace()
        assert trace.passed is False

    def test_build_trace_passed_false_when_assertion_errors(self) -> None:
        tracer = Tracer()
        tracer._assertion_errors = ["tool 'x' was not called"]
        trace = tracer.build_trace()
        assert trace.passed is False


class TestContextVar:
    async def test_concurrent_tracers_isolated(self) -> None:
        """Each concurrent task should have its own active tracer."""
        results: dict[str, Tracer | None] = {}

        async def task(label: str, tracer: Tracer) -> None:
            async with tracer.run(input=label):
                await asyncio.sleep(0.01)
                results[label] = Tracer.current()

        t1, t2 = Tracer(), Tracer()
        await asyncio.gather(task("a", t1), task("b", t2))

        assert results["a"] is t1
        assert results["b"] is t2

    def test_current_returns_none_outside_run(self) -> None:
        assert Tracer.current() is None
