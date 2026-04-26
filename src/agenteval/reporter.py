"""Rich terminal reporter and JSON exporter."""

from __future__ import annotations

import json
import pathlib
from typing import Protocol, runtime_checkable

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from agenteval.models import SuiteResult, TestResult


@runtime_checkable
class Reporter(Protocol):
    def render_result(self, result: TestResult) -> None: ...
    def render_suite(self, suite: SuiteResult) -> None: ...
    def export_json(self, suite: SuiteResult, path: pathlib.Path) -> None: ...


def _pass_rate_display(result: TestResult) -> tuple[str, str]:
    """Return (label, style) for a test result's pass rate."""
    rate = result.pass_rate
    threshold = result.threshold
    fraction = f"{result.n_passed}/{result.n_runs}"

    if rate >= threshold:
        return f"✅ {fraction}  ({rate:.0%})", "green"
    elif rate >= threshold * 0.5:
        return f"⚠️  {fraction}  ({rate:.0%})", "yellow"
    else:
        return f"❌ {fraction}  ({rate:.0%})", "red"


class RichReporter:
    """Rich terminal reporter with color-coded pass rates and summary tables.

    Output format::

        test_basic_search       18/20  ✅ 90%   avg 1.2s   3.1 steps
        test_complex_reasoning   8/20  ⚠️  40%   avg 4.7s   7.2 steps
        test_hallucination       3/20  ❌ 15%   avg 2.1s   5.0 steps
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_failures: bool = True,
        show_traces: bool = False,
    ) -> None:
        self.console = console or Console()
        self.show_failures = show_failures
        self.show_traces = show_traces

    def render_result(self, result: TestResult) -> None:
        """Print a single test result line. Called after each test completes."""
        label, style = _pass_rate_display(result)
        self.console.print(
            f"  [bold]{result.test_name}[/bold]"
            f"  [{style}]{label}[/{style}]"
            f"  [dim]avg {result.avg_duration:.2f}s  {result.avg_steps:.1f} steps[/dim]"
        )

        if self.show_failures and result.failed_traces:
            for i, trace in enumerate(result.failed_traces[:3], 1):
                reasons: list[str] = []
                if trace.error:
                    reasons.append(f"error: {trace.error}")
                if trace.assertion_errors:
                    reasons.extend(trace.assertion_errors[:2])
                if reasons:
                    reason_text = " | ".join(r[:120] for r in reasons[:2])
                    self.console.print(f"    [dim red]↳ failure {i}: {reason_text}[/dim red]")

        if self.show_traces:
            self._print_traces(result)

    def render_suite(self, suite: SuiteResult) -> None:
        """Print a summary table for the full suite."""
        if not suite.results:
            self.console.print("[dim]No tests found.[/dim]")
            return

        total_runs = sum(r.n_runs for r in suite.results)
        self.console.print()

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title=f"agenteval results  —  {suite.total_tests} test(s)  ·  {total_runs} total runs  ·  {suite.duration_seconds:.1f}s",
        )
        table.add_column("Test", style="bold", no_wrap=True)
        table.add_column("Runs", justify="center")
        table.add_column("Pass Rate", justify="center")
        table.add_column("Avg Duration", justify="right")
        table.add_column("Avg Steps", justify="right")
        table.add_column("Threshold", justify="center")

        for result in suite.results:
            label, style = _pass_rate_display(result)
            table.add_row(
                result.test_name,
                f"{result.n_passed}/{result.n_runs}",
                Text(label, style=style),
                f"{result.avg_duration:.2f}s",
                f"{result.avg_steps:.1f}",
                f"{result.threshold:.0%}",
            )

        self.console.print(table)

        # Footer summary
        if suite.all_passed:
            self.console.print(
                f"\n  [bold green]✅ All {suite.total_tests} test(s) passed their threshold.[/bold green]"
            )
        else:
            self.console.print(
                f"\n  [bold red]❌ {suite.failed_tests}/{suite.total_tests} test(s) failed their threshold.[/bold red]"
            )

    def export_json(self, suite: SuiteResult, path: pathlib.Path) -> None:
        """Write suite results to a JSON file."""
        data = suite.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.console.print(f"\n  [dim]JSON report saved to {path}[/dim]")

    def _print_traces(self, result: TestResult) -> None:
        for i, trace in enumerate(result.traces, 1):
            status = "✅" if trace.passed else "❌"
            self.console.print(
                f"    [dim]{status} run {i}  {trace.duration_seconds:.2f}s  "
                f"{trace.effective_steps} steps  output={str(trace.output)[:60]!r}[/dim]"
            )
