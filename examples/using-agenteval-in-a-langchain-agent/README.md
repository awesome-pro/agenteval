# Evaluate a LangChain agent with agenteval: Tool-Calling, Schema Validation, and Reliability Testing

This quickstart shows how to wrap a LangChain agent with agenteval's Tracer to assert that your agent calls the right tools, follows expected ordering, matches response schemas, and meets reliability thresholds over repeated runs.

## Prerequisites

- Python 3.11 or later installed
- A LangChain agent with tool-calling capability (e.g. using OpenAI function calling or Anthropic tool use)
- OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable set (depending on your LangChain LLM provider)
- Basic familiarity with LangChain agents and tools

## Install agenteval with LangChain support

Install the `agenteval-py` package with the `langchain` extra to get LangChain callback integration. If you're using OpenAI or Anthropic as your LLM provider, install those extras too.

```bash
pip install "agenteval-py[langchain,openai]"
```

## Set up your LangChain agent and tools

Create a LangChain agent with a few tools. This example uses a customer support agent with `lookup_order`, `fetch_refund_policy`, and `create_support_ticket` tools. The agent uses OpenAI's function calling to decide which tools to invoke.

The tools themselves are plain async Python functions—LangChain's `@tool` decorator and `StructuredTool.from_function` both work. The key is that agenteval will wrap these tools with `tracer.wrap()` to record every invocation.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# Define your tools as async functions
async def lookup_order(order_id: str) -> str:
    """Look up order details by ID."""
    return f"Order {order_id}: delivered 2 days ago, item=headphones, country=US"

async def fetch_refund_policy(country: str, item: str) -> str:
    """Fetch the refund policy for a given country and item."""
    return f"Country {country}, item {item}: 30-day return window, refund to original payment method"

async def create_support_ticket(order_id: str, reason: str, priority: str) -> str:
    """Create a support ticket."""
    return f"TICKET-{priority.upper()}-{order_id}-{reason.replace(' ', '-').upper()}"

# Convert async functions to LangChain StructuredTools
lookup_order_tool = StructuredTool.from_function(
    coroutine=lookup_order,
    name="lookup_order",
    description="Look up order details by order ID"
)

fetch_refund_policy_tool = StructuredTool.from_function(
    coroutine=fetch_refund_policy,
    name="fetch_refund_policy",
    description="Fetch refund policy for a country and item"
)

create_support_ticket_tool = StructuredTool.from_function(
    coroutine=create_support_ticket,
    name="create_support_ticket",
    description="Create a support ticket with order_id, reason, and priority"
)

tools = [lookup_order_tool, fetch_refund_policy_tool, create_support_ticket_tool]

# Create the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant. Use tools to gather information before responding. Always check the order and policy before making claims."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
```

## Wrap the LangChain agent with agenteval's Tracer

agenteval's `Tracer` records every tool call, timing, and response. Wrap each tool with `tracer.wrap()` to instrument them, then pass the wrapped versions to your LangChain agent.

Inside `tracer.run()`, invoke the agent. The tracer captures the input prompt, the agent's output, and every tool call made along the way. After the agent finishes, use `tracer.assert_that()` to validate behavior: which tools were called, their order, argument values, response schemas, and more.

```python
import agenteval
from agenteval import Tracer
from pydantic import BaseModel, Field

# Define the expected response schema
class RefundReply(BaseModel):
    eligible: bool
    policy: str
    ticket_id: str
    evidence: list[str] = Field(min_length=2)

@agenteval.test(n=10, threshold=0.8, tags=["support", "langchain"])
async def test_langchain_refund_agent(tracer: Tracer) -> None:
    # Wrap the raw async tool functions with the tracer
    wrapped_lookup = tracer.wrap(lookup_order)
    wrapped_policy = tracer.wrap(fetch_refund_policy)
    wrapped_ticket = tracer.wrap(create_support_ticket)
    
    # Recreate LangChain tools from the wrapped functions
    instrumented_tools = [
        StructuredTool.from_function(
            coroutine=wrapped_lookup,
            name="lookup_order",
            description="Look up order details by order ID"
        ),
        StructuredTool.from_function(
            coroutine=wrapped_policy,
            name="fetch_refund_policy",
            description="Fetch refund policy for a country and item"
        ),
        StructuredTool.from_function(
            coroutine=wrapped_ticket,
            name="create_support_ticket",
            description="Create a support ticket with order_id, reason, and priority"
        ),
    ]
    
    # Rebuild the agent executor with instrumented tools
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer support assistant. Use tools to gather information before responding. Return a JSON object with keys: eligible (bool), policy (str), ticket_id (str), evidence (list of strings with at least 2 items)."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    agent = create_openai_tools_agent(llm, instrumented_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=instrumented_tools, verbose=False)
    
    # Run the agent inside the tracer context
    user_prompt = "I want a refund for order A1007"
    async with tracer.run(input=user_prompt) as run:
        result = await agent_executor.ainvoke({"input": user_prompt})
        output = result["output"]
        run.set_output(output)
    
    # Assert on the agent's behavior
    (
        tracer.assert_that()
        .called_tool("lookup_order")  # Must call lookup_order
        .called_tool("fetch_refund_policy")  # Must call fetch_refund_policy
        .called_tool("create_support_ticket")  # Must create a ticket
        .tool_called_before("lookup_order", "fetch_refund_policy")  # Order must be looked up first
        .tool_called_before("fetch_refund_policy", "create_support_ticket")  # Policy before ticket
        .tool_called_with_args("create_support_ticket", {"priority": "normal"})  # Ticket priority must be normal
        .response_matches_schema(RefundReply)  # Response must match the schema
        .response_contains("30-day", case_sensitive=False)  # Must mention the policy
        .completed_within_steps(5)  # Should finish in 5 or fewer tool calls
        .no_errors()  # No unhandled exceptions
        .check()
    )
```

## Run the evaluation

Save the test in a file (e.g. `test_langchain_agent.py`), then run it with the agenteval CLI. The test runs 10 times (as specified by `n=10`), and at least 8 of those runs must pass for the test to succeed (threshold=0.8).

The CLI outputs a summary table showing pass rate, average duration, and average step count. If any runs fail, you'll see the assertion errors and can drill into individual traces.

```bash
agenteval run test_langchain_agent.py
```

## Inspect failures and traces

When a run fails, agenteval reports which assertions broke. You can also run with `--show-traces` to see every tool call, argument, and result for each run.

To export results as JSON for CI/CD pipelines, use `--json-output`.

```bash
agenteval run test_langchain_agent.py --show-traces --json-output results.json
```

## Run directly in Python (optional)

You can also invoke the test programmatically instead of using the CLI. This is useful for integrating agenteval into existing test suites or custom workflows.

```python
import asyncio
import agenteval

result = agenteval.run(
    test_langchain_refund_agent,
    n=10,
    concurrency=4,
    threshold=0.8,
)

print(f"Pass rate: {result.pass_rate:.0%}")
print(f"Passed: {result.n_passed}/{result.n_runs}")
print(f"Average steps: {result.avg_steps:.1f}")

if not result.met_threshold:
    print("Test failed to meet threshold.")
    for trace in result.failed_traces:
        print(f"Run {trace.run_id}: {trace.assertion_errors}")

# Export to JSON
import json
with open("results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

## Expected result

When the test passes, you'll see output like:

```
test_langchain_refund_agent    8/10  ✅ 80%   avg 2.3s   3.2 steps
```

This means 8 out of 10 runs passed all assertions, meeting the 80% threshold. The agent averaged 3.2 tool calls per run and took about 2.3 seconds per invocation.

If a run fails, you'll see which assertions broke—e.g., "Expected tool 'lookup_order' to be called, but it was never invoked" or "Response does not match schema RefundReply: field 'evidence' has fewer than 2 items."

## FAQ

### How do I wrap LangChain tools with agenteval?

Wrap the underlying async function with tracer.wrap(fn), then recreate the LangChain StructuredTool from the wrapped version using StructuredTool.from_function(coroutine=wrapped_fn, name=..., description=...). Pass the new tool list to your agent executor.

### Can I use agenteval with LangChain's built-in @tool decorator?

Yes. If you defined tools with @tool, extract the underlying coroutine, wrap it with tracer.wrap(), and rebuild the tool. For example: raw_fn = my_tool.func; wrapped = tracer.wrap(raw_fn); new_tool = StructuredTool.from_function(coroutine=wrapped, name=my_tool.name, description=my_tool.description).

### What does threshold=0.8 mean?

The test runs n times (e.g. 10 runs). At least 80% of those runs must pass all assertions for the test to succeed. This accounts for LLM non-determinism—your agent doesn't need to be perfect every time, just reliable enough.

### How do I validate the agent's response schema?

Define a Pydantic model (e.g. class RefundReply(BaseModel): eligible: bool; ...) and call tracer.assert_that().response_matches_schema(RefundReply). agenteval parses the agent's output as JSON and validates it against the model.

### Can I test tool call ordering with agenteval?

Yes. Use tracer.assert_that().tool_called_before('tool_a', 'tool_b') to assert that tool_a was called before tool_b in the trace. This is useful for enforcing workflows like 'look up order before fetching policy'.
