# Changelog

All notable changes to agenteval are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

Nothing yet.

---

## [0.1.0] — 2024-01-01

Initial release.

### Added

- **Tracer** — records tool calls (name, arguments, result, duration, error) and run boundaries via `tracer.wrap()`, `@tracer.tool`, and `async with tracer.run()`
- **AssertionSet** — fluent, chainable assertion API with collected failures: `called_tool`, `never_called_tool`, `tool_call_count`, `tool_called_before`, `tool_called_with_args`, `completed_within_steps`, `completed_within_seconds`, `response_contains`, `response_matches_schema`, `no_errors`, `custom`
- **Runner** — executes a test function N times concurrently using `anyio`, supports both sync and async test functions
- **`@agenteval.test` decorator** — registers test functions with `n`, `threshold`, and `tags` parameters; supports bare and parameterized form
- **Suite runner** — discovers `test_*.py` files, imports them, and runs all registered tests
- **RichReporter** — color-coded terminal output (✅ ⚠️ ❌) with pass rate, timing, step count, and optional per-run trace details; JSON export for CI
- **CLI** — `agenteval run` and `agenteval report` commands via Typer
- **OpenAI adapter** — `wrap_tools()` and `extract_token_usage()` for OpenAI function calling
- **Anthropic adapter** — `wrap_tools()` and `extract_token_usage()` for Anthropic tool use (including cache token fields)
- **LangChain adapter** — `AgentEvalCallbackHandler` that auto-connects to the active `Tracer` via `ContextVar`, enabling concurrent runs with full isolation
- **Data models** — `AgentTrace`, `ToolCall`, `TestResult`, `SuiteResult` — all Pydantic v2, fully serializable to JSON
- **Typed** — `py.typed` marker included; full strict Pyright compliance
- **CI** — GitHub Actions workflow testing Python 3.11, 3.12, and 3.13
