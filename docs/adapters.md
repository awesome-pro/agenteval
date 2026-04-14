# Adapters

Adapters are optional integration layers for specific LLM frameworks. The core agenteval library has no dependency on OpenAI, Anthropic, or LangChain — you only pull in what you need.

---

## OpenAI

```bash
pip install "agenteval[openai]"
```

The OpenAI adapter provides two helpers:

- **`wrap_tools(tools_dict, tracer)`** — takes a `dict` mapping tool names to callables, wraps each one with the tracer, and returns the same dict structure
- **`extract_token_usage(response)`** — pulls `prompt_tokens`, `completion_tokens`, and `total_tokens` from a `chat.completions.create` response

```python
from agenteval.adapters.openai_adapter import wrap_tools, extract_token_usage
from openai import AsyncOpenAI
import json

client = AsyncOpenAI()

@agenteval.test(n=15, threshold=0.8)
async def test_openai_agent(tracer: Tracer) -> None:
    # Wrap all tools at once
    tools = wrap_tools(
        {"search": search_fn, "calculator": calc_fn},
        tracer,
    )

    async with tracer.run(input=prompt) as run:
        messages = [{"role": "user", "content": prompt}]

        while True:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tool_schemas,  # your existing tool schema definitions
            )
            run.set_token_usage(extract_token_usage(response))
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                tool_results = []
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = await tools[tc.function.name](**args)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                messages.append(choice.message)
                messages.extend(tool_results)
            else:
                run.set_output(choice.message.content)
                break

    tracer.assert_that().called_tool("search").no_errors().check()
```

---

## Anthropic

```bash
pip install "agenteval[anthropic]"
```

Same interface as the OpenAI adapter:

- **`wrap_tools(tools_dict, tracer)`** — wraps a dict of callables
- **`extract_token_usage(response)`** — extracts `input_tokens`, `output_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens` from a `messages.create` response

```python
from agenteval.adapters.anthropic_adapter import wrap_tools, extract_token_usage
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

@agenteval.test(n=15, threshold=0.8)
async def test_anthropic_agent(tracer: Tracer) -> None:
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
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await tools[block.name](**block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                text = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                run.set_output(text)
                break

    tracer.assert_that().called_tool("web_search").no_errors().check()
```

---

## LangChain

```bash
pip install "agenteval[langchain]"
```

The LangChain integration works differently. Instead of wrapping tools manually, you pass a callback handler to LangChain's `invoke` or `ainvoke` call. The handler intercepts tool events and records them into the active tracer automatically.

```python
from agenteval.adapters.langchain_adapter import AgentEvalCallbackHandler

@agenteval.test(n=10, threshold=0.75)
async def test_langchain_agent(tracer: Tracer) -> None:
    handler = AgentEvalCallbackHandler()

    async with tracer.run(input=prompt) as run:
        result = await agent.ainvoke(
            {"input": prompt},
            config={"callbacks": [handler]},
        )
        run.set_output(result.get("output", ""))

    tracer.assert_that()
        .called_tool("my_tool")
        .no_errors()
        .check()
```

**How it finds the tracer:** `AgentEvalCallbackHandler` calls `Tracer.current()` internally, which reads a `ContextVar` set by `tracer.run()`. You don't pass the tracer to the handler explicitly.

**Concurrent runs:** `anyio.TaskGroup` (used by the runner) copies the current `ContextVar` state into each spawned task. So when 4 runs execute concurrently, each task has its own isolated tracer, and the callback handler in each task finds the right one. No locking or coordination needed.

---

## Using agenteval without an adapter

If you're not using one of the supported frameworks, you don't need an adapter at all. Just wrap your tools with `tracer.wrap()` directly:

```python
@agenteval.test(n=10)
async def test_custom_agent(tracer: Tracer) -> None:
    search = tracer.wrap(my_search_tool)
    write = tracer.wrap(my_write_tool)

    async with tracer.run(input="summarize the top 3 results for python") as run:
        result = await my_custom_agent(
            "summarize the top 3 results for python",
            search=search,
            write=write,
        )
        run.set_output(result)

    tracer.assert_that()
        .called_tool("my_search_tool")
        .called_tool("my_write_tool")
        .no_errors()
        .check()
```

The adapter layer is purely a convenience — it handles the boilerplate of looping over tool calls in framework-specific response formats. The underlying recording mechanism is the same `tracer.wrap()` / `tracer.record_tool_call()` system.
