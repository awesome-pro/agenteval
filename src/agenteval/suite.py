"""Test suite: file discovery and orchestration of multiple tests."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
import types
from typing import TYPE_CHECKING, Optional

from agenteval.models import SuiteResult, TestResult
from agenteval.registry import TestRegistry
from agenteval.runner import run as run_test

if TYPE_CHECKING:
    from agenteval.reporter import Reporter


def discover_test_files(
    paths: list[str | pathlib.Path],
    pattern: str = "test_*.py",
) -> list[pathlib.Path]:
    """Find test files matching a glob pattern under the given paths.

    Args:
        paths: Directories or individual files to search.
        pattern: Glob pattern for test files. Default: 'test_*.py'.

    Returns:
        Sorted list of matching Path objects.
    """
    found: list[pathlib.Path] = []
    for base in paths:
        base_path = pathlib.Path(base).resolve()
        if base_path.is_file():
            found.append(base_path)
        elif base_path.is_dir():
            found.extend(sorted(base_path.rglob(pattern)))
    return found


def import_test_file(path: pathlib.Path) -> types.ModuleType:
    """Import a test file by path, executing any top-level @agenteval.test decorators.

    The file's parent directory is temporarily added to sys.path so that
    relative imports within the test file resolve correctly.

    Args:
        path: Absolute path to the test file.

    Returns:
        The imported module.

    Raises:
        DiscoveryError: If the file cannot be imported.
    """
    from agenteval.exceptions import DiscoveryError

    module_name = f"_agenteval_discovered.{path.stem}"
    parent = str(path.parent)
    added_to_path = parent not in sys.path

    try:
        if added_to_path:
            sys.path.insert(0, parent)

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise DiscoveryError(f"Could not create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    except Exception as e:
        raise DiscoveryError(f"Failed to import test file {path}: {e}") from e
    finally:
        if added_to_path and parent in sys.path:
            sys.path.remove(parent)


def run_suite(
    paths: str | list[str] = ".",
    *,
    pattern: str = "test_*.py",
    tags: Optional[list[str]] = None,
    fail_under: Optional[float] = None,
    n_override: Optional[int] = None,
    concurrency: int = 4,
    reporter: Optional["Reporter"] = None,
) -> SuiteResult:
    """Discover and run all @agenteval.test-decorated tests.

    Args:
        paths: Directory/file paths to search. Default: current directory.
        pattern: Glob pattern for test files. Default: 'test_*.py'.
        tags: If set, only run tests with at least one matching tag.
        fail_under: Override all per-test thresholds with this value.
        n_override: Override per-test n (number of runs) with this value.
        concurrency: Max concurrent runs per test. Default: 4.
        reporter: Reporter instance for live output. Defaults to RichReporter.

    Returns:
        SuiteResult with all test results and aggregate statistics.
    """
    from agenteval.reporter import RichReporter

    if reporter is None:
        reporter = RichReporter()

    # Normalize paths
    path_list: list[str | pathlib.Path] = [paths] if isinstance(paths, str) else list(paths)

    # Snapshot registry state before importing so we only run newly discovered tests
    registry = TestRegistry.global_registry()
    pre_import_count = len(registry.snapshot())

    # Discover and import test files
    files = discover_test_files(path_list, pattern=pattern)
    for f in files:
        import_test_file(f)

    # Get tests registered during this import (or all, if run_suite is called directly)
    all_tests = registry.get_all(tags=tags)
    new_tests = all_tests[pre_import_count:] if pre_import_count < len(all_tests) else all_tests

    start_time = time.time()
    results: list[TestResult] = []

    for registered in new_tests:
        effective_n = n_override if n_override is not None else registered.n
        effective_threshold = fail_under if fail_under is not None else registered.threshold

        result = run_test(
            registered.fn,
            n=effective_n,
            concurrency=concurrency,
            name=registered.name,
            threshold=effective_threshold,
            tags=registered.tags,
        )
        results.append(result)
        reporter.render_result(result)

    end_time = time.time()
    suite_result = SuiteResult(results=results, start_time=start_time, end_time=end_time)
    reporter.render_suite(suite_result)
    return suite_result
