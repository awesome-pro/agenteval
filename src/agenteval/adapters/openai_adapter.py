"""OpenAI function calling adapter for agenteval.

Usage::

    from agenteval.adapters.openai_adapter import wrap_tools, extract_token_usage

    async def test_my_agent(tracer: Tracer) -> None:
        tools = wrap_tools({"search": search_fn, "calculator": calc_fn}, tracer)

        async with tracer.run(input=prompt) as run:
            # Your OpenAI tool-calling loop:
            messages = [{"role": "user", "content": prompt}]
            while True:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=openai_tool_schemas,
                )
                run.set_token_usage(extract_token_usage(response))
                choice = response.choices[0]
                if choice.finish_reason == "tool_calls":
                    for tc in choice.message.tool_calls:
                        import json
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        result = await tools[fn_name](**fn_args)
                        # append tool result to messages ...
                elif choice.finish_reason == "stop":
                    run.set_output(choice.message.content)
                    break

        tracer.assert_that().called_tool("search").no_errors().check()
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from agenteval.tracer import Tracer


def wrap_tools(
    tool_functions: dict[str, Callable[..., Any]],
    tracer: Tracer,
) -> dict[str, Callable[..., Any]]:
    """Wrap a dict of OpenAI tool functions with the tracer.

    Args:
        tool_functions: Mapping of tool name → callable.
        tracer: The active Tracer for the current test run.

    Returns:
        New dict with the same keys but wrapped callables that record
        calls, timing, and errors into the tracer.

    Example::

        tools = wrap_tools({"search": search_fn, "weather": weather_fn}, tracer)
        result = await tools["search"](query="python news")
    """
    return {name: tracer.wrap(fn, name=name) for name, fn in tool_functions.items()}


def extract_token_usage(response: Any) -> Optional[dict[str, int]]:
    """Extract token usage from an OpenAI ChatCompletion response object.

    Args:
        response: An openai.types.chat.ChatCompletion object.

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens, or None.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }
