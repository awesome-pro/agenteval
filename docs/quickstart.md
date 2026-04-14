# Quickstart

This guide gets you from zero to a running eval test in about five minutes.

---

## Installation

```bash
pip install agenteval
```

If you know which LLM framework you're using, grab the adapter too:

```bash
pip install "agenteval[openai]"       # OpenAI function calling
pip install "agenteval[anthropic]"    # Anthropic tool use
pip install "agenteval[langchain]"    # LangChain callbacks
pip install "agenteval[all]"          # all of the above
```

---

## Your first test

Create a file called `tests/test_my_agent.py`. The file name has to start with `test_` — that's how the CLI discovers it.

```python
import agenteval
from agenteval import Tracer


# A simple tool your agent calls during a run.
async def web_search(query: str) -> str:
    # Replace this with your real implementation.
    return f"Top results for: {query}"


# Your agent — replace this with your actual agent logic.
async def my_agent(prompt: str, search) -> str:
    results = await search(query=prompt)
    return f"Based on my search: {results}"


# agenteval will run this function 10 times.
# At least 80% of those runs must pass the assertions below.
@agenteval.test(n=10, threshold=0.8)
async def test_agent_uses_search(tracer: Tracer) -> None:
    # 1. Wrap the tools your agent will call.
    #    The tracer records every invocation transparently.
    search = tracer.wrap(web_search)

    # 2. Mark the agent run boundary.
    #    Duration is tracked automatically.
    async with tracer.run(input="What is agenteval?") as run:
        result = await my_agent("What is agenteval?", search=search)
        run.set_output(result)

    # 3. Assert on what happened across this single run.
    #    All failures are collected — .check() raises once with all of them.
    (tracer.assert_that()
        .called_tool("web_search")     # did the agent actually search?
        .no_errors()                   # did it complete without exceptions?
        .completed_within_steps(5)     # did it stay under 5 tool calls?
        .response_contains("search")   # does the output look reasonable?
        .check())
```

Run it:

```bash
agenteval run tests/
```

You should see something like:

```
  test_agent_uses_search   10/10  ✅ 100%   avg 0.00s   1.0 steps

agenteval results  —  1 test(s)  ·  10 total runs  ·  0.04s

All tests passed.
```

---

## Understanding the output

Each row in the results table has:

- **Test name** — the function name, minus the `test_` prefix in display
- **Runs** — how many passed vs. how many ran total
- **Pass rate** — green ✅ if it met the threshold, yellow ⚠️ if it's between 50% and the threshold, red ❌ below that
- **Avg duration** — mean wall-clock time across all runs
- **Avg steps** — mean number of tool calls across all runs

---

## Running programmatically

You don't need the CLI. You can run tests directly from Python:

```python
import agenteval

# Run a single test and inspect results
result = agenteval.run(test_agent_uses_search, n=20)
print(f"Pass rate: {result.pass_rate:.0%}")
print(f"Met threshold: {result.met_threshold}")

# Look at individual traces
for trace in result.failed_traces:
    print(f"Run {trace.run_id[:8]}:")
    print(f"  Error: {trace.error}")
    print(f"  Assertion failures: {trace.assertion_errors}")

# Run a whole directory of tests
suite = agenteval.run_suite("tests/", fail_under=0.85)
print(f"{suite.passed_tests}/{suite.total_tests} tests passed")
```

---

## Testing a sync agent

If your agent isn't async, that's fine — just use the sync form of the context manager:

```python
@agenteval.test(n=5)
def test_sync_agent(tracer: Tracer) -> None:
    tool = tracer.wrap(my_sync_tool)

    with tracer.run(input="hello") as run:
        result = my_sync_agent("hello", tool=tool)
        run.set_output(result)

    tracer.assert_that().no_errors().check()
```

---

## Next steps

- [Tracer](tracer.md) — everything about wrapping tools and recording runs
- [Assertions](assertions.md) — the full list of available assertions
- [Adapters](adapters.md) — framework-specific integration for OpenAI, Anthropic, LangChain
- [CLI reference](cli.md) — all CLI flags and options
