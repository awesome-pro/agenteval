# agenteval + FastAPI Example

A real FastAPI application with internal tool endpoints and an agent endpoint, tested with **agenteval** for reliability and behavioral correctness.

## What This Demonstrates

- A FastAPI agent that orchestrates multiple tool endpoints (order lookup, policy fetch, ticket creation)
- **agenteval** tests that validate:
  - Tool call ordering and arguments
  - Response schema compliance
  - Reliability across multiple runs (pass rate thresholds)
  - No errors occur during execution

This is a **minimal, runnable** example showing how to use `agenteval-py` to test FastAPI agents.

## Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure environment (optional):**

```bash
cp .env.example .env
# Edit .env if you want custom HOST/PORT
```

3. **Run the FastAPI server:**

```bash
uvicorn app.main:app --reload
```

The server starts at `http://127.0.0.1:8000`. Check health at `http://127.0.0.1:8000/health`.

## Run Tests

With the server running, execute the agenteval tests:

```bash
agenteval run tests/
```

Or run a specific tag:

```bash
agenteval run tests/ --tag support
```

Or run directly in Python:

```python
import agenteval
from tests.test_support_agent import test_refund_agent_reliability

result = agenteval.run(test_refund_agent_reliability, n=10)
print(f"{result.n_passed}/{result.n_runs} passed ({result.pass_rate:.0%})")
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app with tool and agent endpoints
├── tests/
│   ├── __init__.py
│   └── test_support_agent.py  # agenteval tests for the agent
├── .env.example
├── requirements.txt
└── README.md
```

## Key Files

- **app/main.py**: FastAPI application with:
  - `/tools/orders/{order_id}` – lookup order details
  - `/tools/policy` – fetch refund policy
  - `/tools/tickets` – create support ticket
  - `/agent/refund` – agent endpoint that orchestrates tool calls

- **tests/test_support_agent.py**: agenteval tests that wrap the tool endpoints with `tracer.wrap()` and validate:
  - Tool call ordering (lookup_order → fetch_refund_policy → create_support_ticket)
  - Response schema compliance
  - Reliability thresholds (e.g., 90% pass rate over 10 runs)

## How It Works

1. The FastAPI server exposes tool endpoints and an agent endpoint.
2. agenteval tests wrap the tool endpoint calls with `tracer.wrap()` to record execution.
3. Tests run multiple times (e.g., `n=10`) and check that the agent meets reliability thresholds (`threshold=0.9`).
4. Assertions validate tool call order, arguments, response schema, and error-free execution.

## Example Output

```
test_refund_agent_reliability    9/10  ✅ 90%   avg 0.3s   3.0 steps
test_agent_tool_call_order       8/8   ✅ 100%  avg 0.3s   3.0 steps

✅ All tests passed
```

## Learn More

- [agenteval documentation](https://github.com/awesome-pro/agenteval)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
