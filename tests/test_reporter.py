"""Tests for RichReporter."""

from __future__ import annotations

import json
import pathlib
import time

import pytest
from rich.console import Console

from agenteval.models import AgentTrace, SuiteResult, TestResult, ToolCall
from agenteval.reporter import RichReporter, _pass_rate_display


def make_result(
    name: str = "test_foo",
    n_passed: int = 8,
    n_runs: int = 10,
    threshold: float = 0.8,
) -> TestResult:
    traces = [
        AgentTrace(
            run_id=f"r{i}",
            input="q",
            output="ans",
            passed=(i < n_passed),
            duration_seconds=1.0,
        )
        for i in range(n_runs)
    ]
    return TestResult(
        test_name=name,
        n_runs=n_runs,
        n_passed=n_passed,
        pass_rate=n_passed / n_runs,
        threshold=threshold,
        traces=traces,
    )


class TestPassRateDisplay:
    def test_green_above_threshold(self) -> None:
        result = make_result(n_passed=9, n_runs=10, threshold=0.8)
        label, style = _pass_rate_display(result)
        assert "✅" in label
        assert style == "green"

    def test_yellow_between_half_and_threshold(self) -> None:
        result = make_result(n_passed=5, n_runs=10, threshold=0.8)
        label, style = _pass_rate_display(result)
        assert "⚠️" in label
        assert style == "yellow"

    def test_red_below_half_threshold(self) -> None:
        result = make_result(n_passed=2, n_runs=10, threshold=0.8)
        label, style = _pass_rate_display(result)
        assert "❌" in label
        assert style == "red"

    def test_fraction_in_label(self) -> None:
        result = make_result(n_passed=7, n_runs=10)
        label, _ = _pass_rate_display(result)
        assert "7/10" in label


class TestRichReporter:
    def _make_reporter(self) -> tuple[RichReporter, Console]:
        console = Console(width=120, no_color=True)
        reporter = RichReporter(console=console, show_failures=True)
        return reporter, console

    def test_render_result_no_crash(self) -> None:
        reporter, _ = self._make_reporter()
        reporter.render_result(make_result())

    def test_render_suite_no_crash(self) -> None:
        reporter, _ = self._make_reporter()
        r1 = make_result("test_a", n_passed=9)
        r2 = make_result("test_b", n_passed=5)
        suite = SuiteResult(results=[r1, r2], start_time=0.0, end_time=3.5)
        reporter.render_suite(suite)

    def test_render_empty_suite(self) -> None:
        reporter, _ = self._make_reporter()
        suite = SuiteResult(results=[], start_time=0.0, end_time=0.0)
        reporter.render_suite(suite)  # should not raise

    def test_export_json(self, tmp_path: pathlib.Path) -> None:
        reporter, _ = self._make_reporter()
        r1 = make_result("test_a")
        suite = SuiteResult(results=[r1], start_time=0.0, end_time=1.0)
        out = tmp_path / "report.json"
        reporter.export_json(suite, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "results" in data
        assert data["results"][0]["test_name"] == "test_a"

    def test_export_json_roundtrip(self, tmp_path: pathlib.Path) -> None:
        reporter, _ = self._make_reporter()
        r = make_result("test_roundtrip", n_passed=7)
        suite = SuiteResult(results=[r], start_time=1.0, end_time=4.0)
        out = tmp_path / "rt.json"
        reporter.export_json(suite, out)

        from agenteval.models import SuiteResult as SR
        loaded = SR.model_validate(json.loads(out.read_text()))
        assert loaded.results[0].test_name == "test_roundtrip"
        assert loaded.results[0].n_passed == 7

    def test_shows_failure_reasons(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = Console(width=120)
        reporter = RichReporter(console=console, show_failures=True)
        result = make_result("test_fail", n_passed=2, n_runs=5, threshold=0.8)
        # Patch assertion errors onto failed traces
        for trace in result.failed_traces:
            object.__setattr__(trace, "assertion_errors", ["tool 'x' was not called"])
        reporter.render_result(result)
