"""Test runner: executes a test function N times with bounded async concurrency."""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Optional

import anyio

from agenteval.models import AgentTrace, TestResult
from agenteval.tracer import Tracer, _ACTIVE_TRACER


async def _run_single(
    test_fn: Callable[[Tracer], Any],
    run_index: int,
) -> AgentTrace:
    """Execute one run of test_fn with a fresh Tracer. Returns the completed AgentTrace."""
    tracer = Tracer()
    token = _ACTIVE_TRACER.set(tracer)
    try:
        await test_fn(tracer)
    except AssertionError as e:
        # Assertion failures from tracer.assert_that().check()
        tracer._assertion_errors = [line.strip(" •") for line in str(e).splitlines() if line.strip()]
        tracer._assertion_errors = [str(e)]
    except Exception as e:
        if tracer._run_error is None:
            tracer._run_error = f"{type(e).__name__}: {e}"
    finally:
        _ACTIVE_TRACER.reset(token)
        # Close the run context if the test forgot to (e.g. exception before __aexit__)
        if tracer._start_time is not None and tracer._end_time is None:
            tracer._end_time = time.perf_counter()

    return tracer.build_trace()


async def _run_async(
    test_fn: Callable[[Tracer], Any],
    n: int,
    concurrency: int,
) -> list[AgentTrace]:
    """Run test_fn N times with bounded concurrency using anyio."""
    traces: list[AgentTrace] = []
    lock = anyio.Lock()
    semaphore = anyio.Semaphore(concurrency)

    async def bounded_run(i: int) -> None:
        async with semaphore:
            trace = await _run_single(test_fn, i)
            async with lock:
                traces.append(trace)

    async with anyio.create_task_group() as tg:
        for i in range(n):
            tg.start_soon(bounded_run, i)

    return traces


def run(
    test_fn: Callable[[Tracer], Any],
    *,
    n: int = 20,
    concurrency: int = 4,
    name: Optional[str] = None,
    threshold: float = 0.8,
    tags: Optional[list[str]] = None,
) -> TestResult:
    """Run a test function N times and return aggregated results.

    Accepts both sync and async test functions. Sync functions are run in a
    thread pool via anyio.to_thread.run_sync so they don't block the event loop.

    Args:
        test_fn: The test function. Signature: (tracer: Tracer) -> None.
        n: Number of times to run the test. Default: 20.
        concurrency: Maximum number of concurrent runs. Default: 4.
        name: Override the test name (defaults to test_fn.__name__).
        threshold: Pass rate required to consider the test successful. Default: 0.8.
        tags: Optional list of tags for filtering in run_suite().

    Returns:
        TestResult with pass rate, all traces, and statistics.

    Example::

        result = agenteval.run(test_my_agent, n=20, threshold=0.9)
        reporter.render_result(result)
    """
    actual_name = name or getattr(test_fn, "__name__", "unnamed_test")

    # Normalize: wrap sync test functions so the runner is purely async internally
    if not inspect.iscoroutinefunction(test_fn):
        original = test_fn

        async def async_wrapper(tracer: Tracer) -> None:
            await anyio.to_thread.run_sync(functools.partial(original, tracer))

        async_wrapper.__name__ = actual_name
        wrapped: Callable[[Tracer], Any] = async_wrapper
    else:
        wrapped = test_fn

    traces = anyio.run(_run_async, wrapped, n, concurrency)

    n_passed = sum(1 for t in traces if t.passed)
    return TestResult(
        test_name=actual_name,
        n_runs=n,
        n_passed=n_passed,
        pass_rate=n_passed / n if n > 0 else 0.0,
        threshold=threshold,
        traces=traces,
        tags=tags or [],
    )
