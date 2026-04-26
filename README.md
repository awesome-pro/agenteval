# agenteval

[![CI](https://github.com/awesome-pro/agenteval/actions/workflows/ci.yml/badge.svg)](https://github.com/awesome-pro/agenteval/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A lightweight, framework-agnostic toolkit for evaluating and observing LLM agents.

<!-- SCREENSHOT PLACEHOLDER -->
> **[IMAGE TO ADD]** Run `agenteval run tests/` against your demo agent and take a
> screenshot of the terminal output showing the results table. Save it as
> `assets/demo.png` and replace this block with:
> `![agenteval terminal output](assets/demo.png)`

---

## The problem this solves

Standard unit tests don't work for agents. If you write `assert result == "expected answer"`, you've already lost — because the same prompt can produce a different tool-calling sequence, different wording, or even a different conclusion on the next run. Flaky by design.

The correct mental model for agent reliability is statistical. Your agent doesn't "pass" or "fail" — it passes **80% of the time** or **60% of the time**. That's a meaningful number you can track, regress against, and improve. agenteval is built around that idea.

You define a test, tell it how many times to run (`n=20`) and what pass rate you'll accept (`threshold=0.85`), and get back a proper reliability score — along with per-run traces, timing, step counts, and failure breakdowns you can actually debug with.

```
test_find_restaurants      18/20  ✅ 90%   avg 1.4s   2.8 steps
test_multi_step_reasoning   9/20  ⚠️  45%   avg 4.1s   6.3 steps
test_no_hallucination      15/20  ✅ 75%   avg 2.0s   3.5 steps
```

---

## Install

```bash
pip install agenteval-py
```

If you're using a specific framework, grab the extras:

```bash
pip install "agenteval-py[openai]"       # OpenAI function calling
pip install "agenteval-py[anthropic]"    # Anthropic tool use
pip install "agenteval-py[langchain]"    # LangChain callback integration
pip install "agenteval-py[all]"          # everything
```

Requires Python 3.11+.

The PyPI package name is `agenteval-py`, but the Python import and CLI
stay the same:

```bash
agenteval --help
```

```python
import agenteval
```

---

## Quick start

```python
import agenteval
from agenteval import Tracer

# Your real agent and tools go here.
# This is a stand-in for demonstration.
async def web_search(query: str) -> str:
    return f"search results for: {query}"

async def my_agent(prompt: str, search) -> str:
    results = await search(query=prompt)
    return f"Based on my research: {results}"


# This test runs 20 times. It must pass at least 85% of those runs.
@agenteval.test(n=20, threshold=0.85)
async def test_agent_uses_search(tracer: Tracer) -> None:
    # Wrap your tools. The tracer records every call — name, args, result, timing.
    search = tracer.wrap(web_search)

    # Mark the run boundary. Duration is captured automatically.
    async with tracer.run(input="What is agenteval?") as run:
        result = await my_agent("What is agenteval?", search=search)
        run.set_output(result)

    # Assert on what happened. Failures are collected, not raised one-by-one.
    (tracer.assert_that()
        .called_tool("web_search")       # did the agent actually search?
        .completed_within_steps(5)       # not wandering through 10 tool calls?
        .completed_within_seconds(10.0)  # not timing out?
        .response_contains("research")   # did it produce a real response?
        .no_errors()                     # no unhandled exceptions?
        .check())                        # raise once with all failures listed
```

Run it with the CLI:

```bash
agenteval run tests/
```

Or directly in Python:

```python
result = agenteval.run(test_agent_uses_search, n=10)
print(f"{result.n_passed}/{result.n_runs} passed ({result.pass_rate:.0%})")

# Drill into failures
for trace in result.failed_traces:
    print(f"Run {trace.run_id}: {trace.error or trace.assertion_errors}")
```

---

## How it works

<!-- ARCHITECTURE DIAGRAM PLACEHOLDER -->
> **[IMAGE TO ADD]** A simple box diagram showing the flow:
> `Test Function → Tracer (wraps tools) → Runner (N concurrent runs) → Reporter`
> Save it as `assets/architecture.png` and replace this block with:
> `![Architecture](assets/architecture.png)`

There are four moving parts:

**Tracer** — handed to your test function fresh each run. You wrap your tools with it (`tracer.wrap(fn)`), and it records every call: the function name, the arguments it received, the result it returned, how long it took, and whether it threw. Nothing in your actual agent code needs to change.

**Runner** — executes your test function N times concurrently (bounded by `concurrency`, default 4). It handles both sync and async test functions, catches any unhandled exceptions, and aggregates everything into a `TestResult`.

**Assertions** — a fluent chainable API (`tracer.assert_that().called_tool(...).no_errors().check()`) that collects failures instead of raising at the first one. When you call `.check()`, you get a single `AssertionError` listing every problem from that run.

**Reporter** — renders a color-coded summary table in the terminal with pass rates, average timing, step counts, and optional per-run trace details. Exports JSON for CI pipelines.

---

## Writing tests

### The test decorator

```python
@agenteval.test(n=20, threshold=0.85, tags=["search", "integration"])
async def test_my_agent(tracer: Tracer) -> None:
    ...
```

- `n` — how many times to run this test (default: 20)
- `threshold` — minimum pass rate to consider this test passing (default: 0.8)
- `tags` — optional labels for filtering with `agenteval run --tag`

Sync test functions work too — agenteval wraps them automatically:

```python
@agenteval.test(n=10)
def test_sync_agent(tracer: Tracer) -> None:
    tool = tracer.wrap(my_sync_tool)
    with tracer.run(input="hello") as run:
        result = my_sync_agent("hello", tool=tool)
        run.set_output(result)
    tracer.assert_that().no_errors().check()
```

### Wrapping tools

```python
# Method form — returns a wrapped version of the function
search = tracer.wrap(my_search_fn)
search = tracer.wrap(my_search_fn, name="web_search")  # custom name in traces

# Decorator form — useful when defining tools inline
@tracer.tool
def calculator(x: int, y: int) -> int:
    return x + y

@tracer.tool(name="math")
async def calculator(x: int, y: int) -> int:
    return x + y
```

Both sync and async functions are supported. The wrapper is transparent — it preserves return values and re-raises exceptions exactly as before, just with recording on the side.

### Recording the run

```python
async with tracer.run(input="user prompt here") as run:
    result = await my_agent("user prompt here")
    run.set_output(result)
    run.set_token_usage({"input_tokens": 400, "output_tokens": 120})
    run.add_metadata(model="gpt-4o", temperature=0.7)
```

The context manager captures start/end time automatically. Any unhandled exception inside the block is recorded as `trace.error` (AssertionErrors from `.check()` are handled separately by the runner).

---

## Assertions

All assertion methods return `self` so you can chain them. Failures are collected — `.check()` raises once at the end listing every issue, not just the first one.

```python
tracer.assert_that()
    # Tool usage
    .called_tool("search")                              # called at least once
    .never_called_tool("delete_record")                 # must never be called
    .tool_call_count("search", min=1, max=3)            # call count range
    .tool_called_before("search", "summarize")          # ordering check
    .tool_called_with_args("search", {"q": "python"})  # argument subset match
    .tool_called_with_args("search", {"q": "python"}, match="exact")  # exact match

    # Performance
    .completed_within_steps(8)          # at most 8 tool calls
    .completed_within_seconds(15.0)     # wall-clock time limit

    # Output quality
    .response_contains("Python")                        # substring match
    .response_contains("python", case_sensitive=False)  # case-insensitive
    .response_matches_schema(MyPydanticModel)           # structured output validation

    # Errors
    .no_errors()                                        # no unhandled exceptions

    # Custom escape hatch
    .custom(lambda t: len(t.tool_calls) >= 2)
    .custom(lambda t: "gpt-4" in str(t.metadata), message="wrong model used")

    .check()  # raise AssertionError if anything above failed
```

You can also inspect without raising:

```python
aset = tracer.assert_that().called_tool("search").no_errors()
if not aset.passed:
    print(aset.failures)  # list[str] — all failure messages
```

---

## Framework adapters

### OpenAI

```bash
pip install "agenteval-py[openai]"
```

```python
from agenteval.adapters.openai_adapter import wrap_tools, extract_token_usage
import json

@agenteval.test(n=15, threshold=0.8)
async def test_openai_agent(tracer: Tracer) -> None:
    tools = wrap_tools({"search": search_fn, "calculator": calc_fn}, tracer)

    async with tracer.run(input=prompt) as run:
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

### Anthropic

```bash
pip install "agenteval-py[anthropic]"
```

```python
from agenteval.adapters.anthropic_adapter import wrap_tools, extract_token_usage

@agenteval.test(n=15, threshold=0.8)
async def test_anthropic_agent(tracer: Tracer) -> None:
    tools = wrap_tools({"web_search": search_fn}, tracer)

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
                text = next(b.text for b in response.content if b.type == "text")
                run.set_output(text)
                break

    tracer.assert_that().called_tool("web_search").no_errors().check()
```

### LangChain

```bash
pip install "agenteval-py[langchain]"
```

```python
from agenteval.adapters.langchain_adapter import AgentEvalCallbackHandler

@agenteval.test(n=10, threshold=0.75)
async def test_langchain_agent(tracer: Tracer) -> None:
    handler = AgentEvalCallbackHandler()
    # The handler auto-connects to the active tracer via ContextVar.
    # Multiple concurrent runs each get their own isolated tracer — no locking needed.

    async with tracer.run(input=prompt) as run:
        result = await agent.ainvoke(
            {"input": prompt},
            config={"callbacks": [handler]},
        )
        run.set_output(result.get("output", ""))

    tracer.assert_that().called_tool("my_tool").no_errors().check()
```

---

## CLI reference

```bash
# Discover and run all test_*.py files
agenteval run tests/

# Adjust runs and threshold for this session
agenteval run tests/ --n 10 --threshold 0.9

# Filter by tag
agenteval run tests/ --tag integration

# Show per-trace details (useful when debugging failures)
agenteval run tests/ --traces

# Save a JSON report for CI or later inspection
agenteval run tests/ --output report.json

# Pretty-print a saved report
agenteval report report.json
agenteval report report.json --traces
```

**Exit codes:** `0` if all tests met their threshold, `1` if any failed, `2` on error (missing file, import failure, etc.). This makes it straightforward to gate on in CI/CD pipelines.

| Flag | Description | Default |
|------|-------------|---------|
| `--n` | Override run count for all tests | test's own `n` |
| `--threshold` | Override threshold for all tests | test's own threshold |
| `--concurrency` | Max concurrent agent runs | `4` |
| `--tag` | Filter tests by tag (repeatable) | all tests |
| `--pattern` | File glob for test discovery | `test_*.py` |
| `--output` | Write JSON report to a file | none |
| `--traces` | Show per-run trace details | off |
| `--failures / --no-failures` | Show failure reasons | on |

---

## Data models

```python
AgentTrace(
    run_id,            # unique UUID for this run
    input,             # what was passed to the agent
    output,            # what the agent returned
    tool_calls,        # list[ToolCall] — every recorded tool invocation
    duration_seconds,  # wall-clock time for the entire agent run
    total_steps,       # number of tool calls (or manual override via run.set_steps())
    token_usage,       # dict from run.set_token_usage() — None if not provided
    error,             # string if the agent raised an unhandled exception
    assertion_errors,  # list of failure messages from .check()
    passed,            # True only if error is None and assertion_errors is empty
    metadata,          # dict from run.add_metadata()
)

TestResult(
    test_name, n_runs, n_passed,
    pass_rate,          # n_passed / n_runs
    threshold,
    traces,             # all AgentTrace objects for this test
    passed_traces,      # computed — only traces where passed=True
    failed_traces,      # computed — only traces where passed=False
    avg_duration,       # mean duration across all runs
    avg_steps,          # mean step count across all runs
    met_threshold,      # True if pass_rate >= threshold
    tags,
)

SuiteResult(
    results,            # list[TestResult], one per test function
    start_time, end_time,
    total_tests, passed_tests, failed_tests,
    all_passed,
    duration_seconds,
)
```

---

## Using in CI

The exit code makes this drop-in ready. A GitHub Actions example:

```yaml
- name: Run agent evals
  run: agenteval run tests/ --n 5 --output eval-report.json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Upload eval report
  uses: actions/upload-artifact@v4
  with:
    name: eval-report
    path: eval-report.json
  if: always()
```

Keep `n` lower in CI to control API costs — `5` or `10` runs is usually enough to catch regressions.

---

## Project structure

```
agenteval/
├── src/agenteval/
│   ├── __init__.py          # public API surface
│   ├── tracer.py            # Tracer + RunContext — the core recording layer
│   ├── assertions.py        # AssertionSet — fluent assertion API
│   ├── runner.py            # runs a test function N times concurrently
│   ├── registry.py          # @agenteval.test decorator + global test registry
│   ├── suite.py             # discovers and runs test files
│   ├── reporter.py          # Rich terminal output + JSON export
│   ├── cli.py               # Typer CLI (agenteval run / agenteval report)
│   ├── models.py            # AgentTrace, TestResult, SuiteResult, ToolCall
│   ├── exceptions.py        # TracerError
│   └── adapters/
│       ├── openai_adapter.py      # wrap_tools + extract_token_usage for OpenAI
│       ├── anthropic_adapter.py   # same for Anthropic
│       └── langchain_adapter.py   # AgentEvalCallbackHandler for LangChain
├── tests/                   # full test suite for the framework itself
├── docs/                    # detailed documentation per topic
│   ├── quickstart.md
│   ├── tracer.md
│   ├── assertions.md
│   ├── adapters.md
│   └── cli.md
└── pyproject.toml
```

---

## Development setup

```bash
git clone https://github.com/awesome-pro/agenteval
cd agenteval

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# Run the full test suite (no API keys needed)
pytest tests/ -v

# Lint and test
ruff check src/ tests/ --select F,I
pytest tests/ -v
```

The test suite runs completely offline — all agent runs inside the tests use mock functions. Three Python versions are tested in CI (3.11, 3.12, 3.13).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up, the branching strategy, and what to include in a pull request.

---

## License

MIT — see [LICENSE](LICENSE) for details.
