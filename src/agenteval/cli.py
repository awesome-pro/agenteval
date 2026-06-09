"""CLI entry point for agenteval."""

from __future__ import annotations

import pathlib
from typing import Annotated

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
    pattern: Annotated[
        str, typer.Option("--pattern", "-p", help="File glob pattern")
    ] = "test_*.py",
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Only run tests with this tag (repeatable)"),
    ] = None,
    n: Annotated[
        int | None, typer.Option("--n", help="Override number of runs per test")
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Override pass rate threshold (0.0–1.0)"),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Max concurrent runs")
    ] = 4,
    output: Annotated[
        pathlib.Path | None,
        typer.Option("--output", "-o", help="Write JSON report to this file"),
    ] = None,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Disable color output")
    ] = False,
    show_traces: Annotated[
        bool, typer.Option("--traces", help="Show per-trace details")
    ] = False,
    show_failures: Annotated[
        bool, typer.Option("--failures/--no-failures", help="Show failure reasons")
    ] = True,
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
    show_traces: Annotated[
        bool, typer.Option("--traces", help="Show per-trace details")
    ] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
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
