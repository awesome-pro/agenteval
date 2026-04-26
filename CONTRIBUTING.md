# Contributing to agenteval

Thanks for taking the time to contribute. This document covers everything you need to get the development environment running, the conventions the codebase follows, and what a good pull request looks like.

---

## Getting set up

You'll need Python 3.11 or later. The project uses `pip` for dependency management and `hatchling` as the build backend.

```bash
git clone https://github.com/awesome-pro/agenteval
cd agenteval

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (runs ruff on every commit automatically)
pre-commit install
```

That's it. Run the tests to make sure everything is working:

```bash
pytest tests/ -v
```

All tests pass without any API keys. The framework is tested against mock agents.

---

## Project layout

```
src/agenteval/       — library source code
tests/               — test suite (mirrors src/agenteval/ structure)
docs/                — topic-based documentation
.github/workflows/   — CI configuration (lint + tests + build on 3.11/3.12/3.13)
```

The source lives under `src/` to keep it cleanly separated from tests and config. When you add a new module, mirror it in `tests/` with a `test_` prefix.

---

## Code conventions

**Type annotations** — the package ships with type information and uses Pyright
for type checking. If you're working on typing-heavy changes, run `pyright src/`
before pushing.

**Formatting and linting** — Ruff handles both. The pre-commit hook runs it automatically on changed files. You can also run it manually:

```bash
ruff check src/ tests/ --select F,I
ruff format src/
```

**Async first** — the runner, tracer, and suite are all async-native. If you're adding functionality that touches the execution path, prefer `async def` and `anyio` primitives over `asyncio` directly. This keeps things backend-agnostic.

**No framework dependencies in core** — `src/agenteval/` (outside `adapters/`) should not import `openai`, `anthropic`, or `langchain`. Framework-specific code belongs in `src/agenteval/adapters/`.

**Keep the public API surface small** — `__init__.py` exports only what users actually need. If something is implementation detail, don't add it to `__all__`.

---

## Running specific test groups

```bash
# Run everything
pytest tests/ -v

# Run a specific file
pytest tests/test_tracer.py -v

# Run with a keyword filter
pytest tests/ -k "assertion" -v

# Run adapter tests only
pytest tests/adapters/ -v
```

---

## Making changes

**For bug fixes:** open an issue first if the behavior is surprising or the fix is non-obvious. Small, clear bugs can go straight to a PR.

**For new features:** please open an issue or discussion first before building anything significant. This saves everyone time if the design needs to change.

**For documentation:** PRs that improve clarity, fix typos, or add examples are always welcome without prior discussion.

---

## Pull request checklist

Before submitting, make sure:

- [ ] `pytest tests/ -v` passes
- [ ] `ruff check src/ tests/ --select F,I` passes with no errors
- [ ] `python -m build` completes successfully
- [ ] New behavior has corresponding tests
- [ ] If you added a public API, it's documented in `docs/` and/or the relevant docstring
- [ ] Commit messages are descriptive (what changed and roughly why, not just "fix bug")

---

## Commit style

No strict convention here, but a good commit message answers two questions: what changed, and why. One-liners are fine for small changes:

```
fix: tool wrapper now records arguments for positional-only params
```

For anything that took real thought, add a short body:

```
feat: add response_matches_schema assertion

Agents returning structured JSON are common enough that a schema
validation assertion makes sense as a first-class feature. Uses
Pydantic v2 model_validate under the hood, which also handles
JSON string inputs automatically.
```

---

## Questions

If something in the codebase is confusing or the docs are unclear, opening an issue to ask is completely fine. Clear documentation is a feature.
