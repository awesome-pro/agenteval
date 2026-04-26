"""Core tracer: records tool calls and agent run boundaries."""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, overload

from agenteval.models import AgentTrace, ToolCall

if TYPE_CHECKING:
    from agenteval.assertions import AssertionSet

F = TypeVar("F", bound=Callable[..., Any])

# The active Tracer for the current async task / thread context.
# anyio.TaskGroup copies ContextVar state into each spawned task,
# so concurrent test runs each get their own tracer automatically.
_ACTIVE_TRACER: ContextVar[Optional["Tracer"]] = ContextVar("_ACTIVE_TRACER", default=None)


class RunContext:
    """Returned by `tracer.run(input=...)`. Marks the start/end of one agent invocation.

    Supports both sync and async context managers:

        async with tracer.run(input="hello") as run:
            result = await my_agent("hello")
            run.set_output(result)

        with tracer.run(input="hello") as run:
            result = my_sync_agent("hello")
            run.set_output(result)
    """

    def __init__(self, tracer: "Tracer", input: Any) -> None:
        self._tracer = tracer
        self._input = input

    def set_output(self, output: Any) -> None:
        """Record the agent's final response."""
        self._tracer._run_output = output

    def set_token_usage(self, usage: dict[str, int]) -> None:
        """Record token usage (e.g. {'input_tokens': 400, 'output_tokens': 120})."""
        self._tracer._token_usage = usage

    def set_steps(self, n: int) -> None:
        """Override the step count (useful when the agent tracks steps separately)."""
        self._tracer._total_steps = n

    def add_metadata(self, **kwargs: Any) -> None:
        """Attach arbitrary metadata to the trace."""
        self._tracer._metadata.update(kwargs)

    def _enter(self) -> "RunContext":
        self._tracer._run_input = self._input
        self._tracer._start_time = time.perf_counter()
        self._tracer._token = _ACTIVE_TRACER.set(self._tracer)
        return self

    def _exit(self, exc_type: Any, exc_val: Any) -> None:
        self._tracer._end_time = time.perf_counter()
        if exc_type is not None and exc_val is not None:
            # Only capture non-AssertionError exceptions as agent errors.
            # AssertionErrors from .check() are handled by the runner separately.
            if not isinstance(exc_val, AssertionError):
                self._tracer._run_error = f"{exc_type.__name__}: {exc_val}"
        if hasattr(self._tracer, "_token"):
            _ACTIVE_TRACER.reset(self._tracer._token)

    # Async context manager
    async def __aenter__(self) -> "RunContext":
        return self._enter()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._exit(exc_type, exc_val)
        return False  # do not suppress exceptions

    # Sync context manager
    def __enter__(self) -> "RunContext":
        return self._enter()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self._exit(exc_type, exc_val)
        return False


class Tracer:
    """Records all tool calls and run metadata for a single agent test run.

    Usage::

        async def test_my_agent(tracer: Tracer) -> None:
            search = tracer.wrap(my_search_tool)

            async with tracer.run(input="find Python tutorials") as run:
                result = await my_agent("find Python tutorials", search=search)
                run.set_output(result)

            tracer.assert_that().called_tool("my_search_tool").no_errors().check()
    """

    def __init__(self) -> None:
        self._tool_calls: list[ToolCall] = []
        self._run_input: Any = None
        self._run_output: Any = None
        self._run_error: Optional[str] = None
        self._token_usage: Optional[dict[str, int]] = None
        self._total_steps: Optional[int] = None
        self._metadata: dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._assertion_errors: list[str] = []
        self._run_id: str = str(uuid.uuid4())

    def run(self, input: Any, **metadata: Any) -> RunContext:
        """Context manager that marks the boundary of one agent invocation.

        Args:
            input: The input passed to the agent (prompt, dict, etc.)
            **metadata: Extra key/value pairs attached to the trace.
        """
        if metadata:
            self._metadata.update(metadata)
        return RunContext(self, input)

    @overload
    def wrap(self, fn: F) -> F: ...

    @overload
    def wrap(self, fn: F, *, name: str) -> F: ...

    def wrap(self, fn: Callable[..., Any], *, name: Optional[str] = None) -> Callable[..., Any]:
        """Wrap a tool function to automatically record calls, timing, and errors.

        Preserves the sync/async nature of the original function.

        Args:
            fn: The tool callable to wrap.
            name: Override the tool name (defaults to fn.__name__).
        """
        tool_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                ts = time.time()
                error: Optional[str] = None
                result: Any = None
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    duration = time.perf_counter() - start
                    arguments = _build_arguments(fn, args, kwargs)
                    self.record_tool_call(
                        name=tool_name,
                        arguments=arguments,
                        result=result,
                        duration_seconds=duration,
                        timestamp=ts,
                        error=error,
                    )

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                ts = time.time()
                error: Optional[str] = None
                result: Any = None
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    duration = time.perf_counter() - start
                    arguments = _build_arguments(fn, args, kwargs)
                    self.record_tool_call(
                        name=tool_name,
                        arguments=arguments,
                        result=result,
                        duration_seconds=duration,
                        timestamp=ts,
                        error=error,
                    )

            return sync_wrapper  # type: ignore[return-value]

    @overload
    def tool(self, fn: F) -> F: ...

    @overload
    def tool(self, fn: None = None, *, name: Optional[str] = None) -> Callable[[F], F]: ...

    def tool(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
    ) -> Any:
        """Decorator version of wrap(). Supports both @tracer.tool and @tracer.tool(name='x').

        Usage::

            @tracer.tool
            def search(query: str) -> str: ...

            @tracer.tool(name="web_search")
            async def search(query: str) -> str: ...
        """
        if fn is not None:
            return self.wrap(fn, name=name)

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            return self.wrap(f, name=name)

        return decorator

    def record_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        duration_seconds: float,
        timestamp: float,
        error: Optional[str] = None,
    ) -> None:
        """Manually record a tool call. Used by framework adapters (e.g. LangChain)."""
        self._tool_calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                result=result,
                timestamp=timestamp,
                duration_seconds=duration_seconds,
                error=error,
            )
        )

    def assert_that(self) -> "AssertionSet":
        """Return a fluent AssertionSet bound to the current trace snapshot."""
        from agenteval.assertions import AssertionSet

        return AssertionSet(self.build_trace())

    def build_trace(self) -> AgentTrace:
        """Materialize accumulated state into an immutable AgentTrace."""
        duration = 0.0
        if self._start_time is not None and self._end_time is not None:
            duration = self._end_time - self._start_time
        elif self._start_time is not None:
            duration = time.perf_counter() - self._start_time

        passed = self._run_error is None and len(self._assertion_errors) == 0

        return AgentTrace(
            run_id=self._run_id,
            input=self._run_input,
            output=self._run_output,
            tool_calls=list(self._tool_calls),
            total_steps=self._total_steps,
            duration_seconds=duration,
            token_usage=self._token_usage,
            error=self._run_error,
            assertion_errors=list(self._assertion_errors),
            passed=passed,
            metadata=dict(self._metadata),
        )

    @classmethod
    def current(cls) -> Optional["Tracer"]:
        """Get the Tracer active in the current async context.

        Used by framework adapters (e.g. AgentEvalCallbackHandler) to record
        tool calls without requiring an explicit tracer reference.
        """
        return _ACTIVE_TRACER.get()


def _build_arguments(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map positional args to parameter names using the function signature."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (ValueError, TypeError):
        # Fallback: store positional args by index
        result: dict[str, Any] = {f"arg{i}": v for i, v in enumerate(args)}
        result.update(kwargs)
        return result
