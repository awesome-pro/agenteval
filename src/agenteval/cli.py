"""CLI entry point for agenteval."""

from __future__ import annotations

import pathlib
from typing import Annotated, Optional

import typer
from rich.console import Console

from agenteval.registry import TestRegistry
from agenteval.reporter import RichReporter
from agenteval.suite import run_suite

app = typer.Typer(
    name="agenteval",
    help="Evaluation toolkit for LLM agents.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(name="run")
def run_cmd(
    paths: Annotated[
        list[str],
        typer.Argument(help="Test files or directories to discover (default: current dir)"),
    ] = [".", ],  # noqa: B006
    pattern: str = typer.Option("test_*.py", "--pattern", "-p", help="File glob pattern"),
    tags: Optional[list[str]] = typer.Option(None, "--tag", "-t", help="Only run tests with this tag (repeatable)"),
    n: Optional[int] = typer.Option(None, "--n", help="Override number of runs per test"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Override pass rate threshold (0.0–1.0)"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Max concurrent runs"),
    output: Optional[pathlib.Path] = typer.Option(None, "--output", "-o", help="Write JSON report to this file"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable color output"),
    show_traces: bool = typer.Option(False, "--traces", help="Show per-trace details"),
    show_failures: bool = typer.Option(True, "--failures/--no-failures", help="Show failure reasons"),
) -> None:
    """Discover and run agenteval tests."""
    console = Console(no_color=no_color)
    reporter = RichReporter(console=console, show_traces=show_traces, show_failures=show_failures)

    # Reset registry so re-running the CLI in the same process doesn't double-count
    TestRegistry.reset()

    try:
        suite = run_suite(
            paths=paths,
            pattern=pattern,
            tags=tags or None,
            fail_under=threshold,
            n_override=n,
            concurrency=concurrency,
            reporter=reporter,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=2) from e

    if output is not None:
        reporter.export_json(suite, output)

    raise typer.Exit(code=0 if suite.all_passed else 1)


@app.command(name="report")
def report_cmd(
    json_file: Annotated[pathlib.Path, typer.Argument(help="JSON report file from a previous run")],
    show_traces: bool = typer.Option(False, "--traces", help="Show per-trace details"),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Pretty-print a saved JSON report."""
    import json as _json

    from agenteval.models import SuiteResult

    console = Console(no_color=no_color)

    if not json_file.exists():
        console.print(f"[bold red]File not found:[/bold red] {json_file}")
        raise typer.Exit(code=2)

    try:
        data = _json.loads(json_file.read_text(encoding="utf-8"))
        suite = SuiteResult.model_validate(data)
    except Exception as e:
        console.print(f"[bold red]Failed to load report:[/bold red] {e}")
        raise typer.Exit(code=2) from e

    reporter = RichReporter(console=console, show_traces=show_traces)
    for result in suite.results:
        reporter.render_result(result)
    reporter.render_suite(suite)
