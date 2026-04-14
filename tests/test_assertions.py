"""Tests for AssertionSet."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from agenteval.assertions import AssertionSet
from agenteval.models import AgentTrace, ToolCall


def make_trace(
    tool_calls: list[ToolCall] | None = None,
    output: object = "hello world",
    error: str | None = None,
    duration: float = 1.0,
    steps: int | None = None,
) -> AgentTrace:
    return AgentTrace(
        run_id="r",
        input="q",
        output=output,
        tool_calls=tool_calls or [],
        error=error,
        duration_seconds=duration,
        total_steps=steps,
    )


def tc(name: str, args: dict | None = None, error: str | None = None) -> ToolCall:
    return ToolCall(
        name=name,
        arguments=args or {},
        result="ok",
        timestamp=time.time(),
        duration_seconds=0.1,
        error=error,
    )


class TestCalledTool:
    def test_passes_when_tool_called(self) -> None:
        trace = make_trace(tool_calls=[tc("search")])
        AssertionSet(trace).called_tool("search").check()

    def test_fails_when_tool_not_called(self) -> None:
        trace = make_trace()
        with pytest.raises(AssertionError, match="search"):
            AssertionSet(trace).called_tool("search").check()


class TestNeverCalledTool:
    def test_passes_when_tool_not_called(self) -> None:
        trace = make_trace()
        AssertionSet(trace).never_called_tool("delete").check()

    def test_fails_when_tool_was_called(self) -> None:
        trace = make_trace(tool_calls=[tc("delete")])
        with pytest.raises(AssertionError, match="delete"):
            AssertionSet(trace).never_called_tool("delete").check()


class TestToolCallCount:
    def test_exact_count(self) -> None:
        trace = make_trace(tool_calls=[tc("search"), tc("search")])
        AssertionSet(trace).tool_call_count("search", min=2, max=2).check()

    def test_within_range(self) -> None:
        trace = make_trace(tool_calls=[tc("search"), tc("search")])
        AssertionSet(trace).tool_call_count("search", min=1, max=3).check()

    def test_fails_below_min(self) -> None:
        trace = make_trace(tool_calls=[tc("search")])
        with pytest.raises(AssertionError, match="search"):
            AssertionSet(trace).tool_call_count("search", min=2, max=5).check()

    def test_fails_above_max(self) -> None:
        trace = make_trace(tool_calls=[tc("search"), tc("search"), tc("search")])
        with pytest.raises(AssertionError, match="search"):
            AssertionSet(trace).tool_call_count("search", min=1, max=2).check()

    def test_zero_min_default(self) -> None:
        trace = make_trace()
        AssertionSet(trace).tool_call_count("search", max=5).check()


class TestToolCalledBefore:
    def test_passes_correct_order(self) -> None:
        trace = make_trace(tool_calls=[tc("search"), tc("summarize")])
        AssertionSet(trace).tool_called_before("search", "summarize").check()

    def test_fails_wrong_order(self) -> None:
        trace = make_trace(tool_calls=[tc("summarize"), tc("search")])
        with pytest.raises(AssertionError, match="before"):
            AssertionSet(trace).tool_called_before("search", "summarize").check()

    def test_fails_if_first_tool_missing(self) -> None:
        trace = make_trace(tool_calls=[tc("summarize")])
        with pytest.raises(AssertionError, match="never called"):
            AssertionSet(trace).tool_called_before("search", "summarize").check()

    def test_fails_if_second_tool_missing(self) -> None:
        trace = make_trace(tool_calls=[tc("search")])
        with pytest.raises(AssertionError, match="never called"):
            AssertionSet(trace).tool_called_before("search", "summarize").check()


class TestToolCalledWithArgs:
    def test_subset_match_passes(self) -> None:
        trace = make_trace(tool_calls=[tc("search", args={"query": "python", "limit": 10})])
        AssertionSet(trace).tool_called_with_args("search", {"query": "python"}).check()

    def test_subset_match_fails(self) -> None:
        trace = make_trace(tool_calls=[tc("search", args={"query": "python"})])
        with pytest.raises(AssertionError):
            AssertionSet(trace).tool_called_with_args("search", {"query": "rust"}).check()

    def test_exact_match_passes(self) -> None:
        trace = make_trace(tool_calls=[tc("search", args={"query": "python"})])
        AssertionSet(trace).tool_called_with_args(
            "search", {"query": "python"}, match="exact"
        ).check()

    def test_exact_match_fails_extra_key(self) -> None:
        trace = make_trace(tool_calls=[tc("search", args={"query": "python", "limit": 5})])
        with pytest.raises(AssertionError):
            AssertionSet(trace).tool_called_with_args(
                "search", {"query": "python"}, match="exact"
            ).check()

    def test_fails_if_tool_not_called(self) -> None:
        trace = make_trace()
        with pytest.raises(AssertionError, match="never called"):
            AssertionSet(trace).tool_called_with_args("search", {}).check()


class TestCompletedWithinSteps:
    def test_passes(self) -> None:
        trace = make_trace(tool_calls=[tc("a"), tc("b"), tc("c")])
        AssertionSet(trace).completed_within_steps(5).check()

    def test_fails(self) -> None:
        trace = make_trace(tool_calls=[tc("a"), tc("b"), tc("c"), tc("d"), tc("e"), tc("f")])
        with pytest.raises(AssertionError, match="6 steps"):
            AssertionSet(trace).completed_within_steps(5).check()

    def test_uses_total_steps_override(self) -> None:
        trace = make_trace(steps=3)
        AssertionSet(trace).completed_within_steps(3).check()


class TestCompletedWithinSeconds:
    def test_passes(self) -> None:
        trace = make_trace(duration=0.5)
        AssertionSet(trace).completed_within_seconds(1.0).check()

    def test_fails(self) -> None:
        trace = make_trace(duration=2.5)
        with pytest.raises(AssertionError, match="2.50s"):
            AssertionSet(trace).completed_within_seconds(1.0).check()


class TestResponseContains:
    def test_passes(self) -> None:
        trace = make_trace(output="The answer is 42.")
        AssertionSet(trace).response_contains("42").check()

    def test_fails(self) -> None:
        trace = make_trace(output="The answer is unknown.")
        with pytest.raises(AssertionError, match="42"):
            AssertionSet(trace).response_contains("42").check()

    def test_case_insensitive(self) -> None:
        trace = make_trace(output="Hello World")
        AssertionSet(trace).response_contains("hello", case_sensitive=False).check()

    def test_none_output_fails(self) -> None:
        trace = make_trace(output=None)
        with pytest.raises(AssertionError, match="None"):
            AssertionSet(trace).response_contains("keyword").check()


class TestResponseMatchesSchema:
    class MySchema(BaseModel):
        name: str
        age: int

    def test_passes_with_dict_output(self) -> None:
        trace = make_trace(output={"name": "Alice", "age": 30})
        AssertionSet(trace).response_matches_schema(self.MySchema).check()

    def test_passes_with_json_string(self) -> None:
        trace = make_trace(output='{"name": "Bob", "age": 25}')
        AssertionSet(trace).response_matches_schema(self.MySchema).check()

    def test_fails_invalid_schema(self) -> None:
        trace = make_trace(output={"name": "Alice"})  # missing age
        with pytest.raises(AssertionError, match="MySchema"):
            AssertionSet(trace).response_matches_schema(self.MySchema).check()

    def test_fails_invalid_json_string(self) -> None:
        trace = make_trace(output="not valid json {")
        with pytest.raises(AssertionError, match="JSON"):
            AssertionSet(trace).response_matches_schema(self.MySchema).check()

    def test_fails_none_output(self) -> None:
        trace = make_trace(output=None)
        with pytest.raises(AssertionError, match="None"):
            AssertionSet(trace).response_matches_schema(self.MySchema).check()


class TestNoErrors:
    def test_passes_when_no_error(self) -> None:
        trace = make_trace(error=None)
        AssertionSet(trace).no_errors().check()

    def test_fails_when_error(self) -> None:
        trace = make_trace(error="ValueError: something went wrong")
        with pytest.raises(AssertionError, match="ValueError"):
            AssertionSet(trace).no_errors().check()


class TestCustom:
    def test_passes_returns_true(self) -> None:
        trace = make_trace()
        AssertionSet(trace).custom(lambda t: True).check()

    def test_fails_returns_false(self) -> None:
        trace = make_trace()
        with pytest.raises(AssertionError, match="returned False"):
            AssertionSet(trace).custom(lambda t: False).check()

    def test_fails_returns_string(self) -> None:
        trace = make_trace()
        with pytest.raises(AssertionError, match="custom message"):
            AssertionSet(trace).custom(lambda t: "custom message").check()

    def test_custom_message_on_false(self) -> None:
        trace = make_trace()
        with pytest.raises(AssertionError, match="my failure"):
            AssertionSet(trace).custom(lambda t: False, message="my failure").check()

    def test_exception_in_fn_is_captured(self) -> None:
        def bad_fn(t: AgentTrace) -> bool:
            raise RuntimeError("oops")

        trace = make_trace()
        with pytest.raises(AssertionError, match="RuntimeError"):
            AssertionSet(trace).custom(bad_fn).check()


class TestCollectAll:
    def test_collects_multiple_failures(self) -> None:
        trace = make_trace(duration=10.0)
        aset = (
            AssertionSet(trace)
            .called_tool("missing_tool")
            .completed_within_seconds(1.0)
            .response_contains("not_in_output")
        )
        assert len(aset.failures) == 3
        with pytest.raises(AssertionError, match="3 failure"):
            aset.check()

    def test_passed_property(self) -> None:
        trace = make_trace(tool_calls=[tc("search")])
        aset = AssertionSet(trace).called_tool("search")
        assert aset.passed is True

    def test_check_noop_when_no_failures(self) -> None:
        trace = make_trace()
        AssertionSet(trace).check()  # no raise


class TestChaining:
    def test_full_chain(self) -> None:
        trace = make_trace(
            tool_calls=[tc("search", args={"query": "python"}), tc("summarize")],
            output="Here is a summary",
            error=None,
            duration=0.5,
        )
        (
            AssertionSet(trace)
            .called_tool("search")
            .called_tool("summarize")
            .never_called_tool("delete")
            .tool_called_before("search", "summarize")
            .tool_called_with_args("search", {"query": "python"})
            .completed_within_steps(5)
            .completed_within_seconds(2.0)
            .response_contains("summary")
            .no_errors()
            .check()
        )
