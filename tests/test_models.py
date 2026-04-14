"""Tests for agenteval data models."""

import time

import pytest

from agenteval.models import AgentTrace, SuiteResult, TestResult, ToolCall


def make_tool_call(name: str = "search", error: str | None = None) -> ToolCall:
    return ToolCall(
        name=name,
        arguments={"q": "test"},
        result="result",
        timestamp=time.time(),
        duration_seconds=0.1,
        error=error,
    )


def make_trace(passed: bool = True, duration: float = 1.0, steps: int | None = None) -> AgentTrace:
    return AgentTrace(
        run_id="run-1",
        input="hello",
        output="world",
        tool_calls=[make_tool_call()],
        total_steps=steps,
        duration_seconds=duration,
        passed=passed,
    )


class TestToolCall:
    def test_frozen(self) -> None:
        tc = make_tool_call()
        with pytest.raises(Exception):
            tc.name = "other"  # type: ignore[misc]

    def test_error_defaults_none(self) -> None:
        tc = make_tool_call()
        assert tc.error is None


class TestAgentTrace:
    def test_effective_steps_uses_total_steps_when_set(self) -> None:
        trace = make_trace(steps=5)
        assert trace.effective_steps == 5

    def test_effective_steps_falls_back_to_tool_calls(self) -> None:
        trace = AgentTrace(
            run_id="r",
            input="x",
            tool_calls=[make_tool_call(), make_tool_call()],
        )
        assert trace.effective_steps == 2

    def test_passed_defaults_true(self) -> None:
        trace = AgentTrace(run_id="r", input="x")
        assert trace.passed is True

    def test_assertion_errors_defaults_empty(self) -> None:
        trace = AgentTrace(run_id="r", input="x")
        assert trace.assertion_errors == []


class TestTestResult:
    def _make_result(self, n_passed: int = 8, n_runs: int = 10) -> TestResult:
        traces = [make_trace(passed=(i < n_passed)) for i in range(n_runs)]
        return TestResult(
            test_name="test_foo",
            n_runs=n_runs,
            n_passed=n_passed,
            pass_rate=n_passed / n_runs,
            threshold=0.8,
            traces=traces,
        )

    def test_passed_traces_filtered(self) -> None:
        result = self._make_result(n_passed=7, n_runs=10)
        assert len(result.passed_traces) == 7

    def test_failed_traces_filtered(self) -> None:
        result = self._make_result(n_passed=7, n_runs=10)
        assert len(result.failed_traces) == 3

    def test_avg_duration(self) -> None:
        result = self._make_result()
        assert result.avg_duration == pytest.approx(1.0)

    def test_avg_steps(self) -> None:
        result = self._make_result()
        assert result.avg_steps == pytest.approx(1.0)

    def test_met_threshold_true(self) -> None:
        result = self._make_result(n_passed=8, n_runs=10)
        assert result.met_threshold is True

    def test_met_threshold_false(self) -> None:
        result = self._make_result(n_passed=7, n_runs=10)
        assert result.met_threshold is False

    def test_invalid_pass_rate_raises(self) -> None:
        with pytest.raises(Exception):
            TestResult(
                test_name="t",
                n_runs=10,
                n_passed=10,
                pass_rate=1.5,
                threshold=0.8,
                traces=[],
            )


class TestSuiteResult:
    def _make_suite(self) -> SuiteResult:
        r1 = TestResult(
            test_name="a",
            n_runs=10,
            n_passed=9,
            pass_rate=0.9,
            threshold=0.8,
            traces=[],
        )
        r2 = TestResult(
            test_name="b",
            n_runs=10,
            n_passed=5,
            pass_rate=0.5,
            threshold=0.8,
            traces=[],
        )
        return SuiteResult(results=[r1, r2], start_time=0.0, end_time=5.0)

    def test_total_tests(self) -> None:
        assert self._make_suite().total_tests == 2

    def test_passed_tests(self) -> None:
        assert self._make_suite().passed_tests == 1

    def test_failed_tests(self) -> None:
        assert self._make_suite().failed_tests == 1

    def test_all_passed_false(self) -> None:
        assert self._make_suite().all_passed is False

    def test_duration_seconds(self) -> None:
        assert self._make_suite().duration_seconds == pytest.approx(5.0)

    def test_all_passed_true(self) -> None:
        r = TestResult(
            test_name="a", n_runs=10, n_passed=10, pass_rate=1.0, threshold=0.8, traces=[]
        )
        suite = SuiteResult(results=[r], start_time=0.0, end_time=1.0)
        assert suite.all_passed is True
