"""Tests for the test runner."""

from __future__ import annotations

import asyncio

import pytest

from agenteval.runner import run
from agenteval.tracer import Tracer


class TestRunBasic:
    def test_all_passing(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            async with tracer.run(input="hello") as run_ctx:
                run_ctx.set_output("world")
            tracer.assert_that().no_errors().check()

        result = run(test_fn, n=5, concurrency=2)
        assert result.n_runs == 5
        assert result.n_passed == 5
        assert result.pass_rate == 1.0
        assert result.met_threshold is True

    def test_all_failing_assertion(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            async with tracer.run(input="hello"):
                pass
            tracer.assert_that().called_tool("never_called").check()

        result = run(test_fn, n=3, concurrency=2)
        assert result.n_passed == 0
        assert result.pass_rate == 0.0
        assert result.met_threshold is False

    def test_agent_exception_marks_failed(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            async with tracer.run(input="x"):
                raise RuntimeError("agent blew up")

        result = run(test_fn, n=3, concurrency=2)
        assert result.n_passed == 0
        for trace in result.traces:
            assert trace.passed is False
            assert trace.error is not None

    def test_pass_rate_calculated(self) -> None:
        call_count = {"n": 0}

        async def test_fn(tracer: Tracer) -> None:
            call_count["n"] += 1
            async with tracer.run(input="x") as run_ctx:
                run_ctx.set_output("ok")
            # Fail every other run
            if call_count["n"] % 2 == 0:
                tracer.assert_that().called_tool("missing").check()

        result = run(test_fn, n=10, concurrency=1)
        # 5 pass, 5 fail
        assert result.n_passed == 5
        assert result.pass_rate == pytest.approx(0.5)

    def test_threshold_applied(self) -> None:
        async def always_pass(tracer: Tracer) -> None:
            async with tracer.run(input="x"):
                pass

        result = run(always_pass, n=5, threshold=0.9)
        assert result.threshold == 0.9
        assert result.met_threshold is True

    def test_custom_name(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            async with tracer.run(input="x"):
                pass

        result = run(test_fn, n=1, name="my_custom_test")
        assert result.test_name == "my_custom_test"

    def test_tags_stored(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            async with tracer.run(input="x"):
                pass

        result = run(test_fn, n=1, tags=["slow", "integration"])
        assert result.tags == ["slow", "integration"]


class TestSyncTestFn:
    def test_sync_function_supported(self) -> None:
        def sync_test(tracer: Tracer) -> None:
            with tracer.run(input="sync") as run_ctx:
                run_ctx.set_output("result")
            tracer.assert_that().no_errors().check()

        result = run(sync_test, n=3, concurrency=2)
        assert result.n_passed == 3


class TestTraceContents:
    def test_traces_have_run_ids(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            async with tracer.run(input="x"):
                pass

        result = run(test_fn, n=4)
        run_ids = {t.run_id for t in result.traces}
        assert len(run_ids) == 4  # all unique

    def test_tool_calls_recorded(self) -> None:
        async def test_fn(tracer: Tracer) -> None:
            @tracer.tool
            def search(q: str) -> str:
                return "results"

            async with tracer.run(input="find stuff") as run_ctx:
                search(q="python")
                run_ctx.set_output("done")

        result = run(test_fn, n=3)
        for trace in result.traces:
            assert len(trace.tool_calls) == 1
            assert trace.tool_calls[0].name == "search"
