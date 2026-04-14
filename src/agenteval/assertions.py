"""Fluent assertion library for inspecting AgentTrace objects."""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel

from agenteval.models import AgentTrace


class AssertionSet:
    """Fluent assertions on an AgentTrace.

    Failures are **collected**, not raised immediately. Call `.check()` at the
    end of a chain to raise a single AssertionError listing all failures.

    Usage::

        tracer.assert_that()
            .called_tool("search")
            .never_called_tool("delete")
            .completed_within_steps(5)
            .no_errors()
            .check()
    """

    def __init__(self, trace: AgentTrace) -> None:
        self._trace = trace
        self._failures: list[str] = []

    # ------------------------------------------------------------------ #
    # Tool call assertions
    # ------------------------------------------------------------------ #

    def called_tool(self, name: str) -> "AssertionSet":
        """Assert that the tool was called at least once."""
        calls = [tc for tc in self._trace.tool_calls if tc.name == name]
        if not calls:
            all_tools = [tc.name for tc in self._trace.tool_calls]
            self._failures.append(
                f"Expected tool '{name}' to be called, but it was not. "
                f"Tools called: {all_tools or '(none)'}"
            )
        return self

    def never_called_tool(self, name: str) -> "AssertionSet":
        """Assert that the tool was never called."""
        calls = [tc for tc in self._trace.tool_calls if tc.name == name]
        if calls:
            self._failures.append(
                f"Expected tool '{name}' to never be called, but it was called {len(calls)} time(s)."
            )
        return self

    def tool_call_count(
        self,
        name: str,
        *,
        min: int = 0,
        max: int = math.inf,  # type: ignore[assignment]
    ) -> "AssertionSet":
        """Assert that the tool was called between min and max times (inclusive)."""
        count = sum(1 for tc in self._trace.tool_calls if tc.name == name)
        if not (min <= count <= max):
            self._failures.append(
                f"Expected tool '{name}' to be called between {min} and "
                f"{'∞' if max == math.inf else max} times, but it was called {count} time(s)."
            )
        return self

    def tool_called_before(self, tool_a: str, tool_b: str) -> "AssertionSet":
        """Assert that tool_a was called before tool_b (at least one call each)."""
        calls = self._trace.tool_calls
        first_a = next((i for i, tc in enumerate(calls) if tc.name == tool_a), None)
        first_b = next((i for i, tc in enumerate(calls) if tc.name == tool_b), None)

        if first_a is None:
            self._failures.append(
                f"Ordering assertion failed: tool '{tool_a}' was never called."
            )
        elif first_b is None:
            self._failures.append(
                f"Ordering assertion failed: tool '{tool_b}' was never called."
            )
        elif first_a >= first_b:
            self._failures.append(
                f"Expected '{tool_a}' to be called before '{tool_b}', "
                f"but '{tool_b}' was called first (positions: {tool_a}={first_a}, {tool_b}={first_b})."
            )
        return self

    def tool_called_with_args(
        self,
        name: str,
        args: dict[str, Any],
        *,
        match: Literal["subset", "exact"] = "subset",
    ) -> "AssertionSet":
        """Assert that a tool was called with specific arguments.

        Args:
            name: Tool name to check.
            args: Expected arguments.
            match: 'subset' (default) checks all provided keys are present with
                   matching values. 'exact' requires the arguments dict to match exactly.
        """
        matching_calls = [tc for tc in self._trace.tool_calls if tc.name == name]
        if not matching_calls:
            self._failures.append(
                f"tool_called_with_args: tool '{name}' was never called."
            )
            return self

        def _matches(call_args: dict[str, Any]) -> bool:
            if match == "exact":
                return call_args == args
            # subset: all expected keys present with matching values
            return all(call_args.get(k) == v for k, v in args.items())

        if not any(_matches(tc.arguments) for tc in matching_calls):
            actual_args = [tc.arguments for tc in matching_calls]
            self._failures.append(
                f"tool '{name}' was called {len(matching_calls)} time(s), but none matched "
                f"the expected args {args} (match='{match}'). Actual args: {actual_args}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Step / time assertions
    # ------------------------------------------------------------------ #

    def completed_within_steps(self, n: int) -> "AssertionSet":
        """Assert that the agent finished in n steps or fewer."""
        actual = self._trace.effective_steps
        if actual > n:
            self._failures.append(
                f"Expected agent to complete within {n} steps, but took {actual} steps."
            )
        return self

    def completed_within_seconds(self, n: float) -> "AssertionSet":
        """Assert that the agent finished within n seconds."""
        actual = self._trace.duration_seconds
        if actual > n:
            self._failures.append(
                f"Expected agent to complete within {n:.2f}s, but took {actual:.2f}s."
            )
        return self

    # ------------------------------------------------------------------ #
    # Output assertions
    # ------------------------------------------------------------------ #

    def response_contains(self, keyword: str, *, case_sensitive: bool = True) -> "AssertionSet":
        """Assert that the final response contains a keyword."""
        output = self._trace.output
        if output is None:
            self._failures.append(
                f"response_contains: agent output is None, expected to contain '{keyword}'."
            )
            return self

        text = str(output)
        haystack = text if case_sensitive else text.lower()
        needle = keyword if case_sensitive else keyword.lower()

        if needle not in haystack:
            preview = text[:200] + "..." if len(text) > 200 else text
            self._failures.append(
                f"Expected response to contain '{keyword}', but it did not. "
                f"Response: {preview!r}"
            )
        return self

    def response_matches_schema(
        self,
        schema: type[BaseModel],
        *,
        parse_json: bool = True,
    ) -> "AssertionSet":
        """Assert that the final response matches a Pydantic schema.

        If the output is a string and parse_json=True (default), it will be
        JSON-parsed first before validation.
        """
        output = self._trace.output
        if output is None:
            self._failures.append(
                f"response_matches_schema: agent output is None, "
                f"expected to match {schema.__name__}."
            )
            return self

        data: Any = output
        if isinstance(output, str) and parse_json:
            try:
                data = json.loads(output)
            except json.JSONDecodeError as e:
                self._failures.append(
                    f"response_matches_schema: failed to JSON-parse output before "
                    f"validating against {schema.__name__}: {e}. "
                    f"Output was: {output[:200]!r}"
                )
                return self

        try:
            schema.model_validate(data)
        except Exception as e:
            self._failures.append(
                f"response_matches_schema: output does not match schema "
                f"{schema.__name__}: {e}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Error assertions
    # ------------------------------------------------------------------ #

    def no_errors(self) -> "AssertionSet":
        """Assert that the agent completed without any exceptions."""
        if self._trace.error is not None:
            self._failures.append(
                f"Expected no errors, but agent raised: {self._trace.error}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Custom / escape hatch
    # ------------------------------------------------------------------ #

    def custom(
        self,
        fn: Callable[[AgentTrace], Union[bool, str]],
        *,
        message: Optional[str] = None,
    ) -> "AssertionSet":
        """Run a custom assertion function against the trace.

        Args:
            fn: Callable that receives the AgentTrace and returns True (pass),
                False (fail), or a failure message string.
            message: Optional failure message to use when fn returns False.
        """
        try:
            result = fn(self._trace)
        except Exception as e:
            self._failures.append(
                f"custom assertion raised an exception: {type(e).__name__}: {e}"
            )
            return self

        if result is True:
            return self

        if result is False:
            self._failures.append(
                message or "custom assertion failed (returned False)."
            )
        elif isinstance(result, str):
            self._failures.append(result)

        return self

    # ------------------------------------------------------------------ #
    # Terminator
    # ------------------------------------------------------------------ #

    def check(self) -> None:
        """Raise AssertionError listing all collected failures. No-op if all passed."""
        if self._failures:
            lines = "\n".join(f"  • {f}" for f in self._failures)
            raise AssertionError(f"Trace assertions failed ({len(self._failures)} failure(s)):\n{lines}")

    # ------------------------------------------------------------------ #
    # Introspection (for use without raising)
    # ------------------------------------------------------------------ #

    @property
    def passed(self) -> bool:
        """True if no failures have been collected."""
        return len(self._failures) == 0

    @property
    def failures(self) -> list[str]:
        """List of all collected failure messages."""
        return list(self._failures)
