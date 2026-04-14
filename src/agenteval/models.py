"""Pydantic data models for agenteval traces and results."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, computed_field, field_validator


class ToolCall(BaseModel):
    """A single tool/function call recorded during an agent run."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any]
    result: Any = None
    timestamp: float
    duration_seconds: float
    error: Optional[str] = None


class AgentTrace(BaseModel):
    """Full trace of one agent run, including all tool calls and outcome."""

    run_id: str
    input: Any
    output: Any = None
    tool_calls: list[ToolCall] = []
    total_steps: Optional[int] = None
    duration_seconds: float = 0.0
    token_usage: Optional[dict[str, int]] = None
    error: Optional[str] = None
    assertion_errors: list[str] = []
    passed: bool = True
    metadata: dict[str, Any] = {}

    @computed_field  # type: ignore[misc]
    @property
    def effective_steps(self) -> int:
        """Number of steps: explicit total_steps if set, otherwise len(tool_calls)."""
        return self.total_steps if self.total_steps is not None else len(self.tool_calls)


class TestResult(BaseModel):
    """Aggregated results from running a test function N times."""

    test_name: str
    n_runs: int
    n_passed: int
    pass_rate: float
    threshold: float
    traces: list[AgentTrace]
    tags: list[str] = []

    @field_validator("pass_rate")
    @classmethod
    def _validate_pass_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"pass_rate must be between 0 and 1, got {v}")
        return v

    @computed_field  # type: ignore[misc]
    @property
    def passed_traces(self) -> list[AgentTrace]:
        return [t for t in self.traces if t.passed]

    @computed_field  # type: ignore[misc]
    @property
    def failed_traces(self) -> list[AgentTrace]:
        return [t for t in self.traces if not t.passed]

    @computed_field  # type: ignore[misc]
    @property
    def avg_duration(self) -> float:
        if not self.traces:
            return 0.0
        return sum(t.duration_seconds for t in self.traces) / len(self.traces)

    @computed_field  # type: ignore[misc]
    @property
    def avg_steps(self) -> float:
        if not self.traces:
            return 0.0
        return sum(t.effective_steps for t in self.traces) / len(self.traces)

    @computed_field  # type: ignore[misc]
    @property
    def met_threshold(self) -> bool:
        return self.pass_rate >= self.threshold


class SuiteResult(BaseModel):
    """Aggregated results for an entire test suite."""

    results: list[TestResult]
    start_time: float
    end_time: float

    @computed_field  # type: ignore[misc]
    @property
    def total_tests(self) -> int:
        return len(self.results)

    @computed_field  # type: ignore[misc]
    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.met_threshold)

    @computed_field  # type: ignore[misc]
    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests

    @computed_field  # type: ignore[misc]
    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0

    @computed_field  # type: ignore[misc]
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
