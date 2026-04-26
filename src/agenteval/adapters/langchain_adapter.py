"""LangChain callback handler adapter for agenteval.

Usage::

    from agenteval.adapters.langchain_adapter import AgentEvalCallbackHandler

    async def test_langchain_agent(tracer: Tracer) -> None:
        handler = AgentEvalCallbackHandler()

        async with tracer.run(input="find Italian restaurants") as run:
            result = await agent.ainvoke(
                {"input": "find Italian restaurants"},
                config={"callbacks": [handler]},
            )
            run.set_output(result.get("output", ""))

        tracer.assert_that().called_tool("restaurant_search").no_errors().check()

The handler reads the active Tracer from the _ACTIVE_TRACER ContextVar, so it
works automatically when used inside agenteval.run() — no explicit tracer
reference needed.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from agenteval.tracer import Tracer

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[assignment,misc]
    LLMResult = Any  # type: ignore[assignment,misc]


class AgentEvalCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
    """LangChain callback handler that records tool calls into the active Tracer.

    Records each tool invocation's name, arguments, result, duration, and any
    errors. Reads the active tracer via ``Tracer.current()`` (ContextVar), so
    multiple concurrent test runs each get their own tracer automatically.

    If no tracer is active (i.e., used outside of agenteval.run()), all
    callbacks are no-ops to avoid errors in non-test contexts.
    """

    def __init__(self) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for AgentEvalCallbackHandler. "
                "Install it with: pip install agenteval[langchain]"
            )
        super().__init__()
        # Maps LangChain run_id → (start_time, tool_name, parsed_args)
        self._pending: dict[str, tuple[float, str, dict[str, Any]]] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name: str = serialized.get("name", kwargs.get("name", "unknown"))
        try:
            import json
            args = json.loads(input_str) if isinstance(input_str, str) else {"input": input_str}
            if not isinstance(args, dict):
                args = {"input": args}
        except Exception:
            args = {"input": input_str}

        self._pending[str(run_id)] = (time.perf_counter(), tool_name, args)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tracer = Tracer.current()
        if tracer is None:
            return

        key = str(run_id)
        entry = self._pending.pop(key, None)
        if entry is None:
            return

        start_time, tool_name, args = entry
        duration = time.perf_counter() - start_time
        tracer.record_tool_call(
            name=tool_name,
            arguments=args,
            result=output,
            duration_seconds=duration,
            timestamp=time.time() - duration,
            error=None,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tracer = Tracer.current()
        if tracer is None:
            return

        key = str(run_id)
        entry = self._pending.pop(key, None)
        if entry is None:
            return

        start_time, tool_name, args = entry
        duration = time.perf_counter() - start_time
        tracer.record_tool_call(
            name=tool_name,
            arguments=args,
            result=None,
            duration_seconds=duration,
            timestamp=time.time() - duration,
            error=f"{type(error).__name__}: {error}",
        )
