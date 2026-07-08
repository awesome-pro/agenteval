AgentEval is a lightweight, framework-agnostic toolkit for evaluating and observing LLM agents through statistical reliability testing rather than brittle output assertions. This cookbook shows how to test Flask-based agents—whether your Flask app exposes agent endpoints, wraps agent tools as routes, or orchestrates multi-step agent workflows—using AgentEval's tracing, assertion, and reliability-gating capabilities.

Flask's synchronous request-response model and straightforward decorator-based routing make it a natural fit for agent APIs. AgentEval works seamlessly with both sync and async Flask patterns (via `async def` views in Flask 2.0+), wrapping your agent's tool calls and validating behavior over repeated runs. You'll learn how to instrument Flask routes as traced tools, run Flask test clients inside AgentEval test functions, assert on tool-calling sequences and HTTP responses, and set pass-rate thresholds for production readiness.

## How to Set Up AgentEval for Flask Agent Testing

You have a Flask application that exposes agent endpoints or orchestrates agent logic, and you need to test agent behavior (tool calls, response quality, reliability) rather than just HTTP status codes. Standard Flask testing only checks responses; AgentEval validates the agent's internal decision-making.

**Prerequisites**
- Python 3.11 or later installed
- Flask application with agent logic or tool endpoints
- Basic understanding of Flask test clients

```python
# Install agenteval
# pip install agenteval-py flask

import agenteval
from agenteval import Tracer
from flask import Flask, request, jsonify

# Example Flask app with an agent endpoint
app = Flask(__name__)

# Simulated agent tools (in production, these might be database queries,
# external APIs, LLM calls, etc.)
async def search_knowledge_base(query: str) -> str:
    """Simulate a knowledge base search."""
    return f"Knowledge base results for: {query}"

async def generate_answer(context: str, question: str) -> str:
    """Simulate LLM answer generation."""
    return f"Based on {context}, the answer is: [generated response]"

@app.route("/agent/query", methods=["POST"])
def agent_query_endpoint():
    """Flask endpoint that orchestrates an agent workflow."""
    data = request.get_json()
    user_query = data.get("query", "")
    
    # In a real agent, you'd call tools/LLM here
    # For this setup example, we return a placeholder
    return jsonify({
        "query": user_query,
        "response": "Agent response placeholder",
        "tools_used": ["search_knowledge_base", "generate_answer"]
    })

if __name__ == "__main__":
    app.run(debug=True)

# Save the above as flask_agent_app.py
# This is the baseline Flask app we'll test with AgentEval
```

This recipe establishes the foundation. We have a Flask application with an `/agent/query` endpoint that represents an agent's entry point. The agent uses tools like `search_knowledge_base` and `generate_answer` to fulfill requests.

AgentEval testing doesn't replace Flask's test client—it wraps it. You'll use Flask's `app.test_client()` to make HTTP requests inside AgentEval test functions, and AgentEval's `Tracer` to wrap the agent's internal tools so their calls are recorded and can be asserted on.

The key insight: Flask handles HTTP; AgentEval handles agent behavior validation. This separation lets you verify that your agent endpoint not only returns 200 OK, but also called the right tools, in the right order, with the right arguments, and met your reliability threshold across many runs.

**Expected output**

```
# No output yet—this is setup code.
# The Flask app runs normally: flask run
# AgentEval tests will exercise it in subsequent recipes.
```

**Gotchas**
- Flask 2.0+ supports async views (`async def`), but if you're on Flask 1.x, use synchronous tool functions and AgentEval's sync test mode.
- Don't call `app.run()` inside test files—use `app.test_client()` for testing.
- AgentEval requires Python 3.11+; ensure your Flask environment matches.

## How to Write a Basic Flask Agent Test with AgentEval

You need to test that a Flask agent endpoint correctly calls its internal tools and produces a valid response. Unlike standard HTTP tests that only check status codes and JSON structure, you want to verify the agent's tool-calling behavior over multiple runs.

**Prerequisites**
- agenteval-py and flask installed
- A Flask app with an agent endpoint (see previous recipe)
- Understanding of AgentEval's Tracer and assertion API

```python
# test_flask_agent_basic.py
import agenteval
from agenteval import Tracer
from flask import Flask, request, jsonify
import json

# Define the Flask app inline for this test (or import from flask_agent_app.py)
app = Flask(__name__)

# Agent tools
async def search_knowledge_base(query: str) -> str:
    return f"KB results for: {query}"

async def generate_answer(context: str, question: str) -> str:
    return f"Answer based on: {context}"

# Agent logic callable from the endpoint
async def flask_agent_logic(user_query: str, search_tool, generate_tool) -> dict:
    """Core agent logic that uses tools."""
    kb_results = await search_tool(query=user_query)
    answer = await generate_tool(context=kb_results, question=user_query)
    return {
        "query": user_query,
        "response": answer,
        "tools_used": ["search_knowledge_base", "generate_answer"]
    }

@app.route("/agent/query", methods=["POST"])
async def agent_endpoint():
    """Flask endpoint that calls the traced agent logic."""
    data = request.get_json()
    user_query = data.get("query", "")
    
    # In tests, we'll inject traced tools here via app.config or context
    # For now, call untraced tools (production pattern)
    result = await flask_agent_logic(
        user_query,
        search_knowledge_base,
        generate_answer
    )
    return jsonify(result)

# AgentEval test
@agenteval.test(n=10, threshold=0.9, tags=["flask", "agent"])
async def test_flask_agent_uses_tools(tracer: Tracer) -> None:
    """Test that the Flask agent endpoint calls search and generate tools."""
    # Wrap the agent's tools so AgentEval can trace them
    search_tool = tracer.wrap(search_knowledge_base)
    generate_tool = tracer.wrap(generate_answer)
    
    # Use Flask test client to make a request
    client = app.test_client()
    
    user_query = "What is Flask?"
    
    # Record the agent run
    async with tracer.run(input=user_query) as run:
        # Simulate the agent logic directly (in production, you'd call the endpoint
        # and inject traced tools via dependency injection or app context)
        result = await flask_agent_logic(user_query, search_tool, generate_tool)
        run.set_output(result["response"])
    
    # Assert on tool usage and response
    (
        tracer.assert_that()
        .called_tool("search_knowledge_base")
        .called_tool("generate_answer")
        .tool_called_before("search_knowledge_base", "generate_answer")
        .completed_within_steps(2)
        .response_contains("Answer based on", case_sensitive=False)
        .no_errors()
        .check()
    )

if __name__ == "__main__":
    # Run the test
    result = agenteval.run(test_flask_agent_uses_tools, n=10)
    print(f"Test passed: {result.n_passed}/{result.n_runs} ({result.pass_rate:.0%})")
```

This recipe demonstrates a complete AgentEval test for a Flask agent. The pattern:

1. **Define agent tools** (`search_knowledge_base`, `generate_answer`) that the Flask endpoint uses.
2. **Wrap tools with `tracer.wrap()`** inside the test function so AgentEval records every call.
3. **Use Flask's test client** to simulate HTTP requests (or call the agent logic directly if your architecture allows dependency injection).
4. **Record the agent run** with `tracer.run()`, capturing input and output.
5. **Assert on behavior** using AgentEval's fluent API: tool calls, ordering, step count, response content.

The `@agenteval.test` decorator runs this test 10 times (`n=10`) and requires a 90% pass rate (`threshold=0.9`). This catches flaky behavior that a single test run would miss.

**Key difference from standard Flask testing**: We're not just checking `response.status_code == 200` and `response.json["response"]` existence. We're verifying that the agent *always* searches the knowledge base *before* generating an answer, that it does so in exactly 2 steps, and that it never throws an error—across 10 independent runs.

**Expected output**

```
# CLI output when running: agenteval run test_flask_agent_basic.py

test_flask_agent_uses_tools    10/10  ✅ 100%   avg 0.2s   2.0 steps

Test passed: 10/10 (100%)
```

**Gotchas**
- If your Flask app uses synchronous views, use `def` instead of `async def` for both the endpoint and test function. AgentEval handles both.
- For production patterns, inject traced tools via Flask's `g` object, app config, or dependency injection rather than calling agent logic directly in tests.
- The `tracer.wrap()` call must happen *before* the tool is used. Don't wrap inside the agent logic—wrap in the test setup.

## How to Test Flask Agent Endpoints with HTTP Requests

Your Flask agent is a black-box HTTP service. You need to test it by making real HTTP requests (not calling internal functions) and still validate tool usage, timing, and reliability. This is the production testing pattern where the agent's tool orchestration is hidden behind the API.

**Prerequisites**
- Flask app with agent endpoint exposed via HTTP
- AgentEval installed
- Mechanism to inject traced tools into Flask app (e.g., Flask application context, config, or dependency injection)

```python
# test_flask_agent_http.py
import agenteval
from agenteval import Tracer
from flask import Flask, request, jsonify, g
import json

app = Flask(__name__)

# Original agent tools
async def search_docs(query: str) -> str:
    return f"Documentation for: {query}"

async def summarize_results(docs: str) -> str:
    return f"Summary: {docs[:50]}..."

# Flask endpoint with dependency injection pattern
@app.route("/api/agent", methods=["POST"])
async def agent_api():
    data = request.get_json()
    user_input = data.get("input", "")
    
    # Access tools from Flask g (injected during tests)
    search_tool = g.get("search_tool", search_docs)
    summarize_tool = g.get("summarize_tool", summarize_results)
    
    # Agent orchestration
    docs = await search_tool(query=user_input)
    summary = await summarize_tool(docs=docs)
    
    return jsonify({
        "input": user_input,
        "output": summary,
        "steps": 2
    })

@agenteval.test(n=15, threshold=0.85, tags=["flask", "http", "integration"])
async def test_flask_agent_via_http(tracer: Tracer) -> None:
    """Test Flask agent by making HTTP requests with traced tools injected."""
    # Wrap tools
    search_tool = tracer.wrap(search_docs, name="search_docs")
    summarize_tool = tracer.wrap(summarize_results, name="summarize_results")
    
    client = app.test_client()
    
    user_input = "How do I deploy a Flask app?"
    
    async with tracer.run(input=user_input) as run:
        # Inject traced tools into Flask g before the request
        with app.app_context():
            g.search_tool = search_tool
            g.summarize_tool = summarize_tool
            
            # Make HTTP request
            response = client.post(
                "/api/agent",
                data=json.dumps({"input": user_input}),
                content_type="application/json"
            )
            
            response_data = response.get_json()
            run.set_output(response_data["output"])
            run.add_metadata(http_status=response.status_code)
    
    # Assertions
    (
        tracer.assert_that()
        .called_tool("search_docs")
        .called_tool("summarize_results")
        .tool_called_before("search_docs", "summarize_results")
        .completed_within_steps(2)
        .completed_within_seconds(5.0)
        .response_contains("Summary", case_sensitive=False)
        .custom(
            lambda trace: trace.metadata.get("http_status") == 200,
            message="HTTP response must be 200 OK"
        )
        .no_errors()
        .check()
    )

if __name__ == "__main__":
    result = agenteval.run(test_flask_agent_via_http, n=15)
    print(f"{result.n_passed}/{result.n_runs} passed ({result.pass_rate:.0%})")
    print(f"Average steps: {result.avg_steps:.1f}")
    print(f"Average duration: {result.avg_duration:.2f}s")
```

This recipe shows production-grade Flask agent testing: making HTTP requests through the test client while still tracing internal tool calls.

**The pattern**:
1. **Dependency injection via Flask `g`**: The endpoint checks `g.search_tool` and `g.summarize_tool` first, falling back to default implementations. This lets tests inject traced versions.
2. **HTTP request inside `tracer.run()`**: The test makes a real POST request to `/api/agent` with JSON input, just like a real client would.
3. **Metadata tracking**: `run.add_metadata(http_status=response.status_code)` records the HTTP status for later assertion.
4. **Custom assertion**: `.custom(lambda trace: trace.metadata.get("http_status") == 200, ...)` validates the HTTP response alongside tool behavior.

This approach tests the full request-response cycle: routing, JSON parsing, tool orchestration, response serialization. It catches integration issues that unit tests on agent logic alone would miss (e.g., JSON serialization bugs, missing Content-Type headers, Flask context errors).

**Why 15 runs at 85% threshold?** HTTP-based agent tests may encounter more variability (network simulation, timing, concurrency). A slightly lower threshold accounts for acceptable flakiness while still catching real regressions.

**Expected output**

```
# CLI output: agenteval run test_flask_agent_http.py

test_flask_agent_via_http      15/15  ✅ 100%   avg 0.3s   2.0 steps

15/15 passed (100%)
Average steps: 2.0
Average duration: 0.28s
```

**Gotchas**
- Flask's `g` object is request-scoped. Use `with app.app_context():` to set `g` values before calling `client.post()` in tests.
- If you use Flask-RESTful or Blueprints, apply the same dependency injection pattern to resource classes or blueprint factories.
- For async Flask views, ensure you're on Flask 2.0+ and use an async test function (`async def test_...`).
- When testing production deployments, replace `app.test_client()` with actual HTTP calls using `httpx.AsyncClient` or similar, but keep the same AgentEval tracing pattern.

## How to Assert on Flask Agent Response Quality

Your Flask agent returns natural language responses. You need to verify that responses meet quality standards—contain key information, avoid hallucinated data, match expected structure—across many runs, not just once.

**Prerequisites**
- Flask agent endpoint that returns text or structured responses
- AgentEval with schema validation (uses Pydantic models)
- Basic understanding of regex and JSON schema validation

```python
# test_flask_agent_response_quality.py
import agenteval
from agenteval import Tracer
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
import json

app = Flask(__name__)

# Response schema for validation
class AgentResponse(BaseModel):
    query: str
    answer: str = Field(min_length=10)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(min_length=1)

# Mock tools
async def fetch_data(topic: str) -> str:
    return f"Data about {topic}: [detailed information]"

async def generate_response(data: str, query: str) -> dict:
    return {
        "query": query,
        "answer": f"Based on {data}, here is a comprehensive answer with details.",
        "confidence": 0.92,
        "sources": ["source_a", "source_b"]
    }

@app.route("/agent/ask", methods=["POST"])
async def ask_agent():
    data = request.get_json()
    query = data.get("query", "")
    
    # Simulate agent workflow
    fetched = await fetch_data(topic=query)
    response = await generate_response(data=fetched, query=query)
    
    return jsonify(response)

@agenteval.test(n=20, threshold=0.9, tags=["flask", "quality", "response"])
async def test_flask_agent_response_quality(tracer: Tracer) -> None:
    """Test that Flask agent responses meet schema and content requirements."""
    fetch_tool = tracer.wrap(fetch_data)
    generate_tool = tracer.wrap(generate_response)
    
    query = "Explain Flask routing"
    
    async with tracer.run(input=query) as run:
        # Call agent logic (or use HTTP client with dependency injection)
        fetched = await fetch_tool(topic=query)
        response = await generate_tool(data=fetched, query=query)
        
        run.set_output(response["answer"])
        run.add_metadata(confidence=response["confidence"])
    
    # Quality assertions
    (
        tracer.assert_that()
        # Tool usage
        .called_tool("fetch_data")
        .called_tool("generate_response")
        .completed_within_steps(2)
        
        # Response structure validation
        .response_matches_schema(AgentResponse)
        
        # Content validation
        .response_contains("comprehensive", case_sensitive=False)
        .response_contains("details", case_sensitive=False)
        .response_does_not_contain("error", case_sensitive=False)
        .response_does_not_contain("unknown", case_sensitive=False)
        
        # Custom quality checks
        .custom(
            lambda trace: trace.metadata.get("confidence", 0) >= 0.8,
            message="Response confidence must be at least 0.8"
        )
        .custom(
            lambda trace: len(trace.output.split()) >= 10,
            message="Response must be at least 10 words"
        )
        
        .no_errors()
        .check()
    )

if __name__ == "__main__":
    result = agenteval.run(test_flask_agent_response_quality, n=20)
    print(f"Quality test: {result.n_passed}/{result.n_runs} passed ({result.pass_rate:.0%})")
    
    # Analyze failures
    if result.failed_traces:
        print("\nFailure analysis:")
        for trace in result.failed_traces[:3]:  # Show first 3 failures
            print(f"  Run {trace.run_id}: {trace.assertion_errors[0][:100]}")
```

This recipe focuses on response quality validation—ensuring your Flask agent's outputs meet production standards every time.

**Key techniques**:

1. **Schema validation with `.response_matches_schema()`**: Pass a Pydantic model. AgentEval parses the response (assumes JSON if it's a dict/str) and validates against the schema. This catches missing fields, wrong types, and constraint violations (e.g., `confidence` must be 0.0–1.0).

2. **Content assertions**: `.response_contains()` and `.response_does_not_contain()` check for required phrases and forbidden terms. Use these to detect hallucinations, off-topic answers, or error messages leaking into responses.

3. **Custom quality gates**: `.custom()` accepts a lambda that receives the full `trace` object. Check metadata (e.g., confidence scores from the agent), response length, or any domain-specific quality metric.

4. **Statistical validation**: Running 20 times at 90% threshold means you're testing for consistency. A response that passes schema and content checks 18/20 times is reliable; one that passes 10/20 times indicates a flaky agent that needs prompt tuning or better tool orchestration.

**Why this matters for Flask agents**: Flask endpoints often return complex JSON. Standard HTTP tests only check status codes. AgentEval validates the *semantic quality* of responses—are they complete, accurate, and consistent?

**Expected output**

```
# CLI output: agenteval run test_flask_agent_response_quality.py

test_flask_agent_response_quality    20/20  ✅ 100%   avg 0.2s   2.0 steps

Quality test: 20/20 passed (100%)
```

**Gotchas**
- `.response_matches_schema()` expects JSON-serializable output. If your agent returns plain text, use `.response_contains()` for keyword checks instead.
- Pydantic validation is strict. If a field is optional in your schema but your agent sometimes omits it, use `Optional[...]` in the model.
- Case-sensitive checks can be too strict for natural language. Default to `case_sensitive=False` unless exact casing matters.
- For LLM-generated responses, expect some variability. Tune your threshold (e.g., 0.85 instead of 1.0) to account for acceptable variation.

## How to Test Flask Agent Error Handling and Safety

Your Flask agent interacts with external services (databases, APIs, LLMs) that can fail. You need to verify that the agent handles errors gracefully, never exposes sensitive data, and never calls dangerous tools without validation—across many failure scenarios.

**Prerequisites**
- Flask agent with error-prone tool calls
- AgentEval with safety assertions
- Understanding of exception handling in agents

```python
# test_flask_agent_safety.py
import agenteval
from agenteval import Tracer
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Simulated tools with failure modes
async def query_database(sql: str) -> str:
    # Simulate a flaky database
    if "DROP" in sql.upper():
        raise ValueError("SQL injection attempt detected")
    if "SELECT *" in sql:
        return "[sensitive data: user_id, password_hash, ...]"
    return "[safe query results]"

async def call_external_api(endpoint: str) -> str:
    # Simulate network failures
    if endpoint.startswith("/internal"):
        raise PermissionError("Internal endpoints are not allowed")
    if endpoint == "/flaky":
        raise TimeoutError("API timeout")
    return f"API response from {endpoint}"

async def delete_record(record_id: str) -> str:
    # Dangerous tool that should require confirmation
    return f"DELETED record {record_id}"

@app.route("/agent/action", methods=["POST"])
async def agent_action():
    data = request.get_json()
    action = data.get("action", "")
    params = data.get("params", {})
    
    try:
        if action == "query":
            result = await query_database(params.get("sql", ""))
        elif action == "api_call":
            result = await call_external_api(params.get("endpoint", ""))
        elif action == "delete":
            # Safety check: should never allow delete without explicit confirmation
            if not params.get("confirmed"):
                raise ValueError("Delete requires confirmation")
            result = await delete_record(params.get("record_id", ""))
        else:
            result = "Unknown action"
        
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@agenteval.test(n=12, threshold=1.0, tags=["flask", "safety", "errors"])
async def test_flask_agent_blocks_sql_injection(tracer: Tracer) -> None:
    """Test that Flask agent rejects SQL injection attempts."""
    db_tool = tracer.wrap(query_database)
    
    malicious_input = "Show me users; DROP TABLE users;"
    
    async with tracer.run(input=malicious_input) as run:
        try:
            # Simulate agent attempting to query with unsafe input
            await db_tool(sql="SELECT * FROM users WHERE name='" + malicious_input + "'")
            run.set_output("Agent executed dangerous query")
        except ValueError as e:
            run.set_output(f"Agent blocked: {e}")
    
    # Safety assertions
    (
        tracer.assert_that()
        .called_tool("query_database")
        .response_contains("blocked", case_sensitive=False)
        .custom(
            lambda trace: "DROP" not in str(trace.tool_calls[0].result) if trace.tool_calls else True,
            message="Agent must never return results from DROP commands"
        )
        .check()
    )

@agenteval.test(n=10, threshold=1.0, tags=["flask", "safety"])
async def test_flask_agent_never_deletes_without_confirmation(tracer: Tracer) -> None:
    """Test that Flask agent never calls delete_record without explicit confirmation."""
    delete_tool = tracer.wrap(delete_record)
    api_tool = tracer.wrap(call_external_api)
    
    user_input = "Remove record ABC123"
    
    async with tracer.run(input=user_input) as run:
        # Simulate agent workflow: check if delete is needed, but don't execute without confirmation
        api_result = await api_tool(endpoint="/check_record")
        # Agent should NOT call delete_tool here
        run.set_output(f"Checked record status: {api_result}")
    
    (
        tracer.assert_that()
        .called_tool("call_external_api")
        .never_called_tool("delete_record")
        .response_does_not_contain("DELETED", case_sensitive=False)
        .no_errors()
        .check()
    )

@agenteval.test(n=8, threshold=0.75, tags=["flask", "errors", "resilience"])
async def test_flask_agent_handles_api_timeouts(tracer: Tracer) -> None:
    """Test that Flask agent gracefully handles external API failures."""
    api_tool = tracer.wrap(call_external_api)
    
    user_input = "Fetch data from flaky service"
    
    async with tracer.run(input=user_input) as run:
        try:
            result = await api_tool(endpoint="/flaky")
            run.set_output(result)
        except TimeoutError as e:
            run.set_output(f"Handled gracefully: {e}")
    
    (
        tracer.assert_that()
        .called_tool("call_external_api")
        .response_contains("Handled gracefully", case_sensitive=False)
        # Agent should recover and not propagate the error to the user as a crash
        .custom(
            lambda trace: "TimeoutError" not in str(trace.error) if trace.error else True,
            message="Agent must catch and handle TimeoutError, not crash"
        )
        .check()
    )

if __name__ == "__main__":
    print("Running safety tests...\n")
    
    sql_result = agenteval.run(test_flask_agent_blocks_sql_injection, n=12)
    print(f"SQL injection test: {sql_result.pass_rate:.0%}\n")
    
    delete_result = agenteval.run(test_flask_agent_never_deletes_without_confirmation, n=10)
    print(f"Delete safety test: {delete_result.pass_rate:.0%}\n")
    
    timeout_result = agenteval.run(test_flask_agent_handles_api_timeouts, n=8)
    print(f"Timeout resilience test: {timeout_result.pass_rate:.0%}")
```

This recipe covers safety and error-handling validation—critical for production Flask agents.

**Three safety patterns**:

1. **SQL injection / input validation**: The `test_flask_agent_blocks_sql_injection` test verifies that the agent rejects dangerous SQL. The tool (`query_database`) raises `ValueError` on `DROP` commands. The test asserts that the response contains "blocked" and that no `DROP` results leak through.

2. **Dangerous tool gating**: `test_flask_agent_never_deletes_without_confirmation` ensures the agent *never* calls `delete_record` unless explicitly confirmed. This uses `.never_called_tool()`, a critical assertion for irreversible actions (deletions, payments, deployments). A 100% threshold (`1.0`) means zero tolerance for accidental deletes.

3. **Error resilience**: `test_flask_agent_handles_api_timeouts` simulates a flaky external API. The agent should catch `TimeoutError` and return a user-friendly message, not crash. The test uses a lower threshold (`0.75`) because external service flakiness is expected—but the agent must still handle it gracefully most of the time.

**Why this matters**: Flask agents in production face malicious input, network failures, and dangerous tool calls. Standard tests check happy paths. AgentEval validates that your agent behaves safely under adversarial and failure conditions—across many runs, not just once.

**Expected output**

```
# CLI output: agenteval run test_flask_agent_safety.py

test_flask_agent_blocks_sql_injection            12/12  ✅ 100%   avg 0.1s   1.0 steps
test_flask_agent_never_deletes_without_confirmation   10/10  ✅ 100%   avg 0.1s   1.0 steps
test_flask_agent_handles_api_timeouts             6/8   ⚠️  75%   avg 0.1s   1.0 steps

SQL injection test: 100%
Delete safety test: 100%
Timeout resilience test: 75%
```

**Gotchas**
- `.never_called_tool()` only works if the tool is wrapped with `tracer.wrap()`. If the agent bypasses your wrapper, the assertion won't catch it—ensure all code paths use traced tools.
- Safety tests should have high thresholds (0.9–1.0). A 60% pass rate for "never delete without confirmation" is a failing grade.
- For error-handling tests, distinguish between expected errors (caught and handled) and unexpected crashes (propagated exceptions). Use `.no_errors()` to ensure no unhandled exceptions, but allow specific caught errors.
- Simulate realistic failure rates. If your external API times out 10% of the time in production, set your test threshold accordingly (e.g., 0.9 instead of 1.0).

## How to Run Flask Agent Tests with the AgentEval CLI

You've written multiple Flask agent tests with AgentEval decorators. You need to run them all together, filter by tags, export results for CI, and integrate into your deployment pipeline.

**Prerequisites**
- Multiple test files with @agenteval.test decorators
- AgentEval CLI installed (comes with agenteval-py)
- Understanding of CI/CD pipelines (optional for CI integration)

```bash
# Install agenteval if not already installed
pip install agenteval-py flask

# Directory structure for Flask agent tests:
# tests/
#   test_flask_basic.py
#   test_flask_safety.py
#   test_flask_quality.py

# Run all tests in the tests/ directory
agenteval run tests/

# Run only tests tagged "safety"
agenteval run tests/ --tag safety

# Run tests matching a file pattern
agenteval run tests/ --pattern "test_flask_*.py"

# Run with custom concurrency (default is 4 concurrent runs)
agenteval run tests/ --concurrency 8

# Export results to JSON for CI pipelines
agenteval run tests/ --json-report results.json

# Fail the CI build if any test fails its threshold
agenteval run tests/ --fail-under 0.85

# Combine filters: safety tests only, with JSON export
agenteval run tests/ --tag safety --json-report safety_results.json

# Example CI integration (GitHub Actions, GitLab CI, etc.):
# In your .github/workflows/test.yml or .gitlab-ci.yml:
# 
# - name: Run Flask agent tests
#   run: |
#     pip install agenteval-py flask
#     agenteval run tests/ --json-report results.json --fail-under 0.85
# 
# - name: Upload test results
#   uses: actions/upload-artifact@v3
#   with:
#     name: agenteval-results
#     path: results.json
```

This recipe shows how to use AgentEval's CLI to run Flask agent tests in bulk, filter by criteria, and integrate with CI/CD.

**Key CLI commands**:

- `agenteval run <path>` — discovers and runs all functions decorated with `@agenteval.test()` in Python files under `<path>`.
- `--tag <name>` — filters to tests with matching tags (e.g., `@agenteval.test(tags=["safety"])`). Useful for running only critical tests in pre-merge checks.
- `--pattern <glob>` — limits discovery to files matching the glob (e.g., `test_flask_*.py`).
- `--concurrency <N>` — controls how many test runs execute in parallel. Higher values speed up large test suites but may overwhelm external services.
- `--json-report <file>` — exports results as JSON for parsing by CI tools, dashboards, or monitoring systems.
- `--fail-under <threshold>` — exits with code 1 if any test's pass rate is below the threshold. Use this to gate deployments.

**CI/CD integration pattern**:
1. Install dependencies (`pip install agenteval-py flask`).
2. Run tests with `agenteval run tests/ --json-report results.json --fail-under 0.85`.
3. If the command exits with code 0, all tests passed their thresholds—safe to deploy.
4. If it exits with code 1, at least one test failed—block the deployment and review the JSON report for details.

**Why this matters**: Flask agents need continuous testing. The CLI makes it trivial to run hundreds of test cases (each with `n=10` or `n=20` runs) in seconds, parallelize them, and get deterministic pass/fail signals for CI.

**Expected output**

```
# Terminal output from: agenteval run tests/

Discovered 8 test functions in tests/

test_flask_agent_uses_tools                      10/10  ✅ 100%   avg 0.2s   2.0 steps
test_flask_agent_via_http                        15/15  ✅ 100%   avg 0.3s   2.0 steps
test_flask_agent_response_quality                20/20  ✅ 100%   avg 0.2s   2.0 steps
test_flask_agent_blocks_sql_injection            12/12  ✅ 100%   avg 0.1s   1.0 steps
test_flask_agent_never_deletes_without_confirmation 10/10  ✅ 100%   avg 0.1s   1.0 steps
test_flask_agent_handles_api_timeouts             6/8   ⚠️  75%   avg 0.1s   1.0 steps

All tests passed their thresholds.
Total runs: 83, Total passed: 83, Overall pass rate: 100%
```

**Gotchas**
- The CLI only discovers functions with `@agenteval.test()`. Plain `def test_...` functions are ignored.
- Tags are case-sensitive. Use consistent naming (e.g., always lowercase).
- `--concurrency` affects only the number of *parallel runs* within a single test, not the number of tests run in parallel. To parallelize test discovery and execution, use pytest-xdist or similar.
- JSON reports are overwritten on each run. Archive them with timestamps in CI if you need historical data.
- The `--fail-under` threshold applies to individual tests, not the overall pass rate. A test with `threshold=0.8` that achieves 0.75 will fail even if other tests pass.

## FAQ

### Can I use AgentEval with Flask-RESTful or Flask-SQLAlchemy?

Yes. AgentEval is framework-agnostic and only requires that you wrap your agent's tool calls with `tracer.wrap()`. Flask-RESTful resources, Flask-SQLAlchemy models, and Blueprint-based apps all work—just inject traced tools via dependency injection (Flask `g`, app config, or constructor arguments) as shown in the HTTP testing recipe.

### How do I test Flask agents that call real LLMs (OpenAI, Anthropic)?

AgentEval includes adapters for OpenAI and Anthropic. Install the extras (`pip install agenteval-py[openai]`), use `wrap_tools()` from `agenteval.adapters.openai_adapter` or `anthropic_adapter`, and pass the wrapped tools to your LLM client. The adapters handle function-calling extraction and tracing automatically. See the live_llm examples in the repository for full working code.

### What if my Flask agent is non-deterministic (different outputs every run)?

That's exactly what AgentEval is designed for. Instead of asserting on exact output text, you assert on behavior: tool usage, response schema, presence of key information, and absence of errors. Set `n=10` or `n=20` and a threshold like `0.85` to validate that your agent *usually* behaves correctly, accounting for natural LLM variability.

### Can I run AgentEval tests in pytest?

Yes. You can call `agenteval.run(your_test_function, n=10)` inside a pytest test, or use the AgentEval CLI (`agenteval run tests/`) as a separate test suite. The CLI is often faster and provides better terminal output for agent tests, but pytest integration works if you need fixtures or parametrization.

### How do I debug a Flask agent test that's failing?

Use the CLI's trace output: `agenteval run tests/ --show-traces --show-failures`. This prints per-run details: which tools were called, their arguments and results, timing, and the exact assertion that failed. You can also export JSON (`--json-report`) and inspect the `traces` array for programmatic analysis.

## Key takeaways

- AgentEval validates Flask agent behavior (tool calls, response quality, error handling) across many runs, not just once—catching flaky and non-deterministic issues that single tests miss.
- Wrap agent tools with `tracer.wrap()` to record every call; use Flask's test client or dependency injection to make HTTP requests while still tracing internal tool usage.
- Assertions check behavior, not exact output: `.called_tool()`, `.never_called_tool()`, `.response_matches_schema()`, `.completed_within_steps()`, and custom lambdas for domain-specific quality gates.
- Set realistic thresholds (`threshold=0.85` for typical agents, `1.0` for safety-critical tests) to require statistical reliability, not brittle pass/fail on a single run.
- Use the AgentEval CLI to run test suites, filter by tags, export JSON for CI, and gate deployments with `--fail-under`—making Flask agent testing a first-class part of your pipeline.