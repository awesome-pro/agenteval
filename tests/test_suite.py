"""Tests for test suite discovery and run_suite()."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from agenteval.registry import TestRegistry
from agenteval.suite import discover_test_files, import_test_file, run_suite


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    TestRegistry.reset()


class TestDiscoverTestFiles:
    def test_finds_test_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "test_foo.py").write_text("# test")
        (tmp_path / "test_bar.py").write_text("# test")
        (tmp_path / "helper.py").write_text("# helper")

        found = discover_test_files([tmp_path])
        names = {f.name for f in found}
        assert "test_foo.py" in names
        assert "test_bar.py" in names
        assert "helper.py" not in names

    def test_recursive_discovery(self, tmp_path: pathlib.Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "test_nested.py").write_text("# test")

        found = discover_test_files([tmp_path])
        names = {f.name for f in found}
        assert "test_nested.py" in names

    def test_accepts_single_file(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "test_single.py"
        f.write_text("# test")
        found = discover_test_files([f])
        assert len(found) == 1

    def test_custom_pattern(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "eval_foo.py").write_text("# eval")
        (tmp_path / "test_bar.py").write_text("# test")
        found = discover_test_files([tmp_path], pattern="eval_*.py")
        names = {f.name for f in found}
        assert "eval_foo.py" in names
        assert "test_bar.py" not in names

    def test_empty_dir(self, tmp_path: pathlib.Path) -> None:
        found = discover_test_files([tmp_path])
        assert found == []


class TestImportTestFile:
    def test_registers_decorated_tests(self, tmp_path: pathlib.Path) -> None:
        code = textwrap.dedent("""
            import agenteval
            from agenteval.tracer import Tracer

            @agenteval.test(n=3, threshold=0.7)
            async def test_example(tracer: Tracer) -> None:
                async with tracer.run(input="x") as run:
                    run.set_output("y")
        """)
        f = tmp_path / "test_example.py"
        f.write_text(code)
        import_test_file(f)

        registry = TestRegistry.global_registry()
        entries = registry.get_all()
        assert len(entries) == 1
        assert entries[0].name == "test_example"
        assert entries[0].n == 3
        assert entries[0].threshold == 0.7

    def test_raises_discovery_error_on_bad_file(self, tmp_path: pathlib.Path) -> None:
        from agenteval.exceptions import DiscoveryError

        f = tmp_path / "test_bad.py"
        f.write_text("this is not valid python ::::")
        with pytest.raises(DiscoveryError):
            import_test_file(f)


class TestRunSuite:
    def test_run_suite_finds_and_runs(self, tmp_path: pathlib.Path) -> None:
        code = textwrap.dedent("""
            import agenteval
            from agenteval.tracer import Tracer

            @agenteval.test(n=2, threshold=0.5)
            async def test_always_pass(tracer: Tracer) -> None:
                async with tracer.run(input="x") as run:
                    run.set_output("y")
                tracer.assert_that().no_errors().check()
        """)
        (tmp_path / "test_suite_example.py").write_text(code)

        suite = run_suite(str(tmp_path), pattern="test_suite_example.py")
        assert suite.total_tests == 1
        assert suite.results[0].test_name == "test_always_pass"
        assert suite.results[0].n_runs == 2
        assert suite.results[0].n_passed == 2

    def test_n_override(self, tmp_path: pathlib.Path) -> None:
        code = textwrap.dedent("""
            import agenteval
            from agenteval.tracer import Tracer

            @agenteval.test(n=20)
            async def test_override(tracer: Tracer) -> None:
                async with tracer.run(input="x"):
                    pass
        """)
        (tmp_path / "test_override.py").write_text(code)

        suite = run_suite(str(tmp_path), pattern="test_override.py", n_override=3)
        assert suite.results[0].n_runs == 3

    def test_fail_under_override(self, tmp_path: pathlib.Path) -> None:
        code = textwrap.dedent("""
            import agenteval
            from agenteval.tracer import Tracer

            @agenteval.test(n=2, threshold=0.5)
            async def test_threshold(tracer: Tracer) -> None:
                async with tracer.run(input="x"):
                    pass
        """)
        (tmp_path / "test_threshold.py").write_text(code)

        suite = run_suite(str(tmp_path), pattern="test_threshold.py", fail_under=0.99)
        assert suite.results[0].threshold == 0.99

    def test_empty_directory(self, tmp_path: pathlib.Path) -> None:
        suite = run_suite(str(tmp_path))
        assert suite.total_tests == 0
        assert suite.all_passed is True
