"""Evaluate a LangChain customer support agent with agenteval.

Run:
    OPENAI_API_KEY=sk-... agenteval run test_langchain_agent.py

Or:
    OPENAI_API_KEY=sk-... python test_langchain_agent.py
"""

from __future__ import annotations

import os
from pydantic import BaseModel, Field

import agenteval
from agenteval import Tracer

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Define the expected response schema
class RefundReply(BaseModel):
    eligible: bool
    policy: str
    ticket_id: str
    evidence: list[str] = Field(min_length=2)


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


@agenteval.test(n=10, threshold=0.8, tags=["support", "langchain"])
async def test_langchain_refund_agent(tracer: Tracer) -> None:
    """Test that the LangChain agent calls the right tools in the right order."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this test.")

    # Wrap the raw async tool functions with the tracer
    wrapped_lookup = tracer.wrap(lookup_order)
    wrapped_policy = tracer.wrap(fetch_refund_policy)
    wrapped_ticket = tracer.wrap(create_support_ticket)

    # Recreate LangChain tools from the wrapped functions
    instrumented_tools = [
        StructuredTool.from_function(
            coroutine=wrapped_lookup,
            name="lookup_order",
            description="Look up order details by order ID",
        ),
        StructuredTool.from_function(
            coroutine=wrapped_policy,
            name="fetch_refund_policy",
            description="Fetch refund policy for a country and item",
        ),
        StructuredTool.from_function(
            coroutine=wrapped_ticket,
            name="create_support_ticket",
            description="Create a support ticket with order_id, reason, and priority",
        ),
    ]

    # Build the agent executor with instrumented tools
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful customer support assistant. Use tools to gather information before responding. "
                "Return a JSON object with keys: eligible (bool), policy (str), ticket_id (str), evidence (list of strings with at least 2 items).",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

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
        .tool_called_before("lookup_order", "fetch_refund_policy")  # Order before policy
        .tool_called_before("fetch_refund_policy", "create_support_ticket")  # Policy before ticket
        .tool_called_with_args(
            "create_support_ticket", {"priority": "normal"}
        )  # Ticket priority must be normal
        .response_matches_schema(RefundReply)  # Response must match the schema
        .response_contains("30-day", case_sensitive=False)  # Must mention the policy
        .completed_within_steps(5)  # Should finish in 5 or fewer tool calls
        .no_errors()  # No unhandled exceptions
        .check()
    )


if __name__ == "__main__":
    result = agenteval.run(
        test_langchain_refund_agent,
        n=10,
        concurrency=4,
        threshold=0.8,
    )

    print(f"\nPass rate: {result.pass_rate:.0%}")
    print(f"Passed: {result.n_passed}/{result.n_runs}")
    print(f"Average steps: {result.avg_steps:.1f}")
    print(f"Met threshold: {result.met_threshold}")

    if not result.met_threshold:
        print("\nTest failed to meet threshold. Failures:")
        for trace in result.failed_traces:
            print(f"  Run {trace.run_id}: {trace.assertion_errors}")
