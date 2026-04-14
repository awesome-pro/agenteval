# Assertions

`AssertionSet` is returned by `tracer.assert_that()`. All methods return `self` for chaining. The important design decision here: failures are **collected**, not raised immediately. When you call `.check()` at the end, you get one `AssertionError` that lists every problem from that run — not just the first one.

```python
tracer.assert_that()
    .called_tool("search")
    .no_errors()
    .completed_within_steps(5)
    .check()  # raises once with all failures, or no-ops if everything passed
```

---

## Tool call assertions

### `.called_tool(name)`

Passes if the named tool was called at least once during the run.

```python
.called_tool("web_search")
```

Fails if the tool was never called. Failure message includes the tool name and lists what tools were actually called.

### `.never_called_tool(name)`

Passes if the named tool was never called.

```python
.never_called_tool("delete_record")
```

Useful for safety checks — asserting your agent never reaches for a destructive tool when it shouldn't.

### `.tool_call_count(name, min=0, max=None)`

Checks that the tool was called within a count range.

```python
.tool_call_count("search", min=1, max=3)   # called 1, 2, or 3 times
.tool_call_count("search", min=2)           # called at least twice
.tool_call_count("search", max=1)           # called at most once
```

### `.tool_called_before(first, second)`

Checks ordering — the first call to `first` happened before the first call to `second`.

```python
.tool_called_before("search", "summarize")
```

Fails if either tool was never called, or if `second` was called before `first`.

### `.tool_called_with_args(name, args, match="subset")`

Checks that at least one call to the named tool was made with the given arguments.

```python
# Subset match (default) — the call just needs to include these args
.tool_called_with_args("search", {"query": "python"})

# Exact match — the arguments dict must match exactly
.tool_called_with_args("search", {"query": "python"}, match="exact")
```

For subset matching, extra keys in the actual call are ignored. For exact matching, the argument dicts must be equal.

---

## Performance assertions

### `.completed_within_steps(n)`

Passes if the number of tool calls is at most `n`. This uses `trace.effective_steps`, which is either the value from `run.set_steps()` (if you set it manually) or the count of recorded tool calls.

```python
.completed_within_steps(8)
```

Catching step count regressions is one of the more useful things you can track — an agent that used to do a task in 3 tool calls and now takes 10 is worth knowing about.

### `.completed_within_seconds(n)`

Passes if `trace.duration_seconds` is at most `n`.

```python
.completed_within_seconds(15.0)
```

---

## Output assertions

### `.response_contains(text, case_sensitive=True)`

Passes if `str(trace.output)` contains the given string.

```python
.response_contains("Python")                        # case-sensitive
.response_contains("python", case_sensitive=False)  # case-insensitive
```

Fails if `trace.output` is `None`, or if the string isn't found.

### `.response_matches_schema(PydanticModel)`

Validates the output against a Pydantic v2 model. Handles both dict outputs and JSON string outputs automatically.

```python
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

.response_matches_schema(SearchResult)
```

Passes if `model.model_validate(output)` succeeds. Fails with the Pydantic validation error message included.

---

## Error assertions

### `.no_errors()`

Passes if `trace.error` is `None`. This means no unhandled exception was raised inside the `tracer.run()` block.

```python
.no_errors()
```

This is one of the most commonly useful assertions — you often want to verify the agent completed without crashing before checking anything else.

---

## Custom assertions

### `.custom(fn, message=None)`

An escape hatch for things the built-in assertions don't cover. Pass a callable that takes an `AgentTrace` and returns:

- `True` — assertion passed
- `False` — assertion failed (uses `message` if provided, otherwise a default)
- A string — assertion failed with that string as the failure message

```python
# Simple boolean check
.custom(lambda t: len(t.tool_calls) >= 2)

# With a failure message
.custom(lambda t: len(t.tool_calls) >= 2, message="agent should use at least 2 tools")

# Return a string for a dynamic failure message
.custom(lambda t: f"too many steps: {len(t.tool_calls)}" if len(t.tool_calls) > 5 else True)
```

If the custom function raises an exception, it's caught and reported as a failure (not propagated).

---

## Checking and inspecting

```python
# Raise AssertionError if any failures were collected. No-op if all passed.
.check()

# Check without raising — useful when you want to branch on the result
aset = tracer.assert_that().called_tool("search").no_errors()
if aset.passed:
    print("all good")
else:
    for msg in aset.failures:
        print(f"  - {msg}")
```

`.passed` is a bool. `.failures` is a `list[str]` of failure messages.

---

## A complete example

```python
(tracer.assert_that()
    .called_tool("search")
    .never_called_tool("delete_record")
    .tool_call_count("search", min=1, max=4)
    .tool_called_before("search", "summarize")
    .tool_called_with_args("search", {"query": "python"})
    .completed_within_steps(8)
    .completed_within_seconds(15.0)
    .response_contains("summary")
    .no_errors()
    .custom(lambda t: t.token_usage is not None, message="token usage not recorded")
    .check())
```

If three of these fail on a given run, `.check()` raises with all three failure messages listed, not just the first one.
