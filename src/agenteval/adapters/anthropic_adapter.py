"""Anthropic tool use adapter for agenteval.

Usage::

    from agenteval.adapters.anthropic_adapter import wrap_tools, extract_token_usage

    async def test_my_agent(tracer: Tracer) -> None:
        tools = wrap_tools({"web_search": search_fn, "calculator": calc_fn}, tracer)

        async with tracer.run(input=prompt) as run:
            messages = [{"role": "user", "content": prompt}]
            while True:
                response = await client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    tools=anthropic_tool_schemas,
                    messages=messages,
                )
                run.set_token_usage(extract_token_usage(response))
                if response.stop_reason == "tool_use":
                    for block in response.content:
                        if block.type == "tool_use":
                            result = await tools[block.name](**block.input)
                            # append tool result to messages ...
                elif response.stop_reason == "end_turn":
                    text = next(
                        (b.text for b in response.content if b.type == "text"), ""
                    )
                    run.set_output(text)
                    break

        tracer.assert_that().called_tool("web_search").no_errors().check()
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agenteval.tracer import Tracer


def wrap_tools(
    tool_functions: dict[str, Callable[..., Any]],
    tracer: Tracer,
) -> dict[str, Callable[..., Any]]:
    """Wrap a dict of Anthropic tool functions with the tracer.

    Args:
        tool_functions: Mapping of tool name → callable.
        tracer: The active Tracer for the current test run.

    Returns:
        New dict with the same keys but wrapped callables that record
        calls, timing, and errors into the tracer.
    """
    return {name: tracer.wrap(fn, name=name) for name, fn in tool_functions.items()}


def extract_token_usage(response: Any) -> Optional[dict[str, int]]:
    """Extract token usage from an Anthropic Message response.

    Args:
        response: An anthropic.types.Message object.

    Returns:
        Dict with input_tokens, output_tokens, or None if unavailable.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    result: dict[str, int] = {}
    if hasattr(usage, "input_tokens"):
        result["input_tokens"] = usage.input_tokens
    if hasattr(usage, "output_tokens"):
        result["output_tokens"] = usage.output_tokens
    if hasattr(usage, "cache_read_input_tokens"):
        result["cache_read_input_tokens"] = usage.cache_read_input_tokens
    if hasattr(usage, "cache_creation_input_tokens"):
        result["cache_creation_input_tokens"] = usage.cache_creation_input_tokens
    return result or None
