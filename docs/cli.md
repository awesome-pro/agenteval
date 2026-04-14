# CLI Reference

The `agenteval` CLI has two commands: `run` for executing tests, and `report` for inspecting saved results.

---

## `agenteval run`

Discovers test files, imports them, runs all `@agenteval.test`-decorated functions, and prints a results table.

```
agenteval run [OPTIONS] [PATHS]...
```

`PATHS` can be files or directories. When a directory is given, agenteval recursively searches for files matching `--pattern` (default: `test_*.py`). If no paths are given, it searches the current directory.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --pattern TEXT` | File glob pattern for discovery | `test_*.py` |
| `-t, --tag TEXT` | Only run tests with this tag (repeatable) | all tests |
| `--n INTEGER` | Override run count for every test | each test's own `n` |
| `--threshold FLOAT` | Override pass rate threshold (0.0–1.0) | each test's own threshold |
| `-c, --concurrency INT` | Max concurrent agent runs | `4` |
| `-o, --output PATH` | Write JSON report to this file | none |
| `--no-color` | Disable color output | color on |
| `--traces` | Show per-run trace details after the summary | off |
| `--failures / --no-failures` | Show failure reasons | on |

### Exit codes

- `0` — every test met its threshold
- `1` — one or more tests failed their threshold
- `2` — error before tests could run (bad path, import failure, etc.)

This makes it straightforward to use in CI — just let the exit code determine whether the step succeeds or fails.

### Examples

```bash
# Run everything in tests/
agenteval run tests/

# Run with 10 repetitions per test and a 90% threshold
agenteval run tests/ --n 10 --threshold 0.9

# Only run tests tagged "fast" — useful for a quick pre-push check
agenteval run tests/ --tag fast

# Show trace details — handy when you're debugging a specific failure
agenteval run tests/test_search.py --traces

# Run without color (for log files or CI output that doesn't support ANSI)
agenteval run tests/ --no-color

# Save a JSON report and also display results in the terminal
agenteval run tests/ --output results.json

# Run tests from multiple directories at once
agenteval run tests/unit tests/integration
```

---

## `agenteval report`

Pretty-prints a JSON report that was saved with `--output`. Useful for re-examining results from a CI run without re-running the tests.

```
agenteval report [OPTIONS] JSON_FILE
```

### Options

| Option | Description |
|--------|-------------|
| `--traces` | Show per-run trace details |
| `--no-color` | Disable color output |

### Examples

```bash
# Print the summary table
agenteval report results.json

# Include per-trace detail (tool calls, durations, errors)
agenteval report results.json --traces
```

---

## JSON report format

When you pass `--output`, agenteval writes the full `SuiteResult` to JSON. The structure matches the data models documented in the main README:

```json
{
  "results": [
    {
      "test_name": "test_agent_uses_search",
      "n_runs": 20,
      "n_passed": 18,
      "pass_rate": 0.9,
      "threshold": 0.85,
      "met_threshold": true,
      "avg_duration": 1.42,
      "avg_steps": 2.8,
      "tags": [],
      "traces": [
        {
          "run_id": "...",
          "input": "What is agenteval?",
          "output": "Based on my research: ...",
          "tool_calls": [
            {
              "name": "web_search",
              "arguments": {"query": "What is agenteval?"},
              "result": "search results for: ...",
              "timestamp": 1700000000.0,
              "duration_seconds": 0.012,
              "error": null
            }
          ],
          "duration_seconds": 0.015,
          "total_steps": null,
          "token_usage": null,
          "error": null,
          "assertion_errors": [],
          "passed": true,
          "metadata": {}
        }
      ]
    }
  ],
  "start_time": "2024-01-01T12:00:00",
  "end_time": "2024-01-01T12:00:05"
}
```

This format is stable and designed to be easy to parse for dashboards, alerting, or historical tracking.

---

## Using in CI (GitHub Actions example)

```yaml
- name: Install agenteval
  run: pip install "agenteval[openai]"

- name: Run agent evals
  run: agenteval run tests/ --n 5 --no-color --output eval-report.json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

- name: Upload eval report
  uses: actions/upload-artifact@v4
  with:
    name: eval-report
    path: eval-report.json
  if: always()
```

The step fails (exit code 1) automatically if any test doesn't meet its threshold. The report is uploaded even on failure so you can inspect what went wrong.
