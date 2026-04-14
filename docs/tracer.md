# Tracer

The `Tracer` is the central object in agenteval. A fresh one is created for every single test run, and it records everything: which tools were called, with what arguments, what they returned, how long each call took, and whether the overall run succeeded.

You receive a `Tracer` as the argument to your test function — you never construct one yourself.

```python
@agenteval.test(n=20)
async def test_my_agent(tracer: Tracer) -> None:
    # tracer is fresh for this run
    ...
```

---

## Wrapping tools

Before passing tools to your agent, wrap them with the tracer. The wrapper is transparent — the function behaves identically from the caller's perspective, it just also records everything on the side.

### Method form

```python
search = tracer.wrap(my_search_fn)
search = tracer.wrap(my_search_fn, name="web_search")  # override name in traces
```

### Decorator form

```python
@tracer.tool
def calculator(x: int, y: int) -> int:
    return x + y

# With a custom name
@tracer.tool(name="math_op")
async def calculator(x: int, y: int) -> int:
    return x + y
```

Both forms support sync and async functions. The wrapper:

- Records the function name (or your custom `name`)
- Binds positional arguments to parameter names using `inspect.signature`
- Records the return value
- Records the wall-clock duration of the call
- If the function raises, records the exception type and message — then re-raises it unchanged

---

## Marking the run boundary

Use `tracer.run()` as a context manager to mark where your agent invocation starts and ends. Everything inside the block is considered one run.

```python
# Async (the usual case)
async with tracer.run(input="find me a restaurant") as run:
    result = await my_agent("find me a restaurant", tools=tools)
    run.set_output(result)

# Sync
with tracer.run(input="find me a restaurant") as run:
    result = my_agent("find me a restaurant", tools=tools)
    run.set_output(result)
```

The context manager automatically captures start and end time, so `trace.duration_seconds` is always populated.

### RunContext methods

Inside the `async with` / `with` block, you have access to a `RunContext` object (the `run` variable above). Use it to record extra information:

```python
async with tracer.run(input=prompt) as run:
    result = await my_agent(prompt)
    run.set_output(result)

    # Token usage — pass whatever dict your API returns
    run.set_token_usage({"input_tokens": 400, "output_tokens": 120})

    # Override step count if your agent tracks it differently
    run.set_steps(7)

    # Arbitrary metadata — useful for debugging
    run.add_metadata(model="gpt-4o", temperature=0.7, attempt=2)
```

`set_output`, `set_token_usage`, `set_steps`, and `add_metadata` can all be called anywhere inside the block, in any order.

### Exception handling

If your agent raises an unhandled exception inside the `run` block, it's captured as `trace.error` (as a string) and the run is marked as failed. The exception is re-raised so your test function can see it too.

`AssertionError` exceptions from `.check()` are explicitly excluded — those are handled separately by the runner and stored in `trace.assertion_errors`.

---

## Recording tool calls manually

Framework adapters (like the LangChain callback handler) use this method to record tool invocations without going through `tracer.wrap()`:

```python
tracer.record_tool_call(
    name="search",
    arguments={"query": "python tutorials", "limit": 5},
    result="10 results found",
    duration_seconds=0.34,
    timestamp=1234567890.0,
    error=None,
)
```

You probably won't need this directly unless you're building an adapter for a new framework.

---

## Assertions

After your agent run completes, call `tracer.assert_that()` to get an `AssertionSet` bound to the current trace:

```python
tracer.assert_that()
    .called_tool("search")
    .no_errors()
    .check()
```

See [assertions.md](assertions.md) for the full API.

---

## Getting the active tracer from context

In async code, you can retrieve the tracer that's active in the current task using `Tracer.current()`. This is how the LangChain adapter finds the tracer without it being passed explicitly:

```python
tracer = Tracer.current()  # returns None if not inside a tracer.run() block
```

agenteval uses a `ContextVar` internally. When the runner spawns concurrent test runs with `anyio.TaskGroup`, each task automatically gets its own copy of the context variable — so concurrent runs are fully isolated without any locking.

---

## Building the trace

The runner calls `tracer.build_trace()` automatically after each test run. You shouldn't need to call it yourself, but you can:

```python
trace = tracer.build_trace()  # returns an immutable AgentTrace
```

The returned `AgentTrace` is a Pydantic model, so it serializes to JSON cleanly.
