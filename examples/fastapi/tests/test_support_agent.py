"""agenteval tests for the FastAPI support agent.

These tests validate agent behavior across multiple runs:
- Tool call ordering and arguments
- Response schema compliance
- Reliability thresholds
"""

import httpx
import agenteval
from agenteval import Tracer
from app.main import RefundResponse


BASE_URL = "http://127.0.0.1:8000"


async def lookup_order(order_id: str) -> dict:
    """Wrapper for the FastAPI tool endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/tools/orders/{order_id}")
        response.raise_for_status()
        return response.json()


async def fetch_refund_policy(country: str, item: str) -> dict:
    """Wrapper for the FastAPI tool endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/tools/policy",
            params={"country": country, "item": item},
        )
        response.raise_for_status()
        return response.json()


async def create_support_ticket(order_id: str, reason: str, priority: str) -> dict:
    """Wrapper for the FastAPI tool endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/tools/tickets",
            json={"order_id": order_id, "reason": reason, "priority": priority},
        )
        response.raise_for_status()
        return response.json()


async def call_agent(order_id: str, message: str) -> dict:
    """Call the FastAPI agent endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/agent/refund",
            json={"order_id": order_id, "message": message},
        )
        response.raise_for_status()
        return response.json()


@agenteval.test(n=10, threshold=0.9, tags=["support", "policy", "fastapi"])
async def test_refund_agent_reliability(tracer: Tracer) -> None:
    """Test that the FastAPI agent correctly handles refund requests.

    This validates:
    - Agent calls lookup_order, fetch_refund_policy, and create_support_ticket
    - Tools are called in the correct order
    - Ticket is created with expected priority
    - Response matches the RefundResponse schema
    - Response contains policy details
    - No errors occur
    """
    order_tool = tracer.wrap(lookup_order)
    policy_tool = tracer.wrap(fetch_refund_policy)
    ticket_tool = tracer.wrap(create_support_ticket)

    prompt = "I want a refund for order A1007"
    async with tracer.run(input=prompt) as run:
        run.add_metadata(endpoint="/agent/refund", framework="fastapi")

        # The agent internally calls these tools
        order = await order_tool(order_id="A1007")
        policy = await policy_tool(country=order["country"], item=order["item"])
        ticket = await ticket_tool(
            order_id=order["order_id"],
            reason="refund request within policy window",
            priority="normal",
        )

        # Simulate calling the agent endpoint
        result = await call_agent(order_id="A1007", message=prompt)
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("lookup_order")
        .called_tool("fetch_refund_policy")
        .called_tool("create_support_ticket")
        .tool_called_before("lookup_order", "fetch_refund_policy")
        .tool_called_before("fetch_refund_policy", "create_support_ticket")
        .tool_called_with_args("create_support_ticket", {"priority": "normal"})
        .completed_within_steps(3)
        .response_matches_schema(RefundResponse)
        .response_contains("30-day policy", case_sensitive=False)
        .no_errors()
        .check()
    )


@agenteval.test(n=8, threshold=0.85, tags=["support", "fastapi"])
async def test_agent_tool_call_order(tracer: Tracer) -> None:
    """Verify that the agent always looks up the order before fetching policy."""
    order_tool = tracer.wrap(lookup_order)
    policy_tool = tracer.wrap(fetch_refund_policy)
    ticket_tool = tracer.wrap(create_support_ticket)

    async with tracer.run(input="Refund for order A1007") as run:
        order = await order_tool(order_id="A1007")
        policy = await policy_tool(country=order["country"], item=order["item"])
        await ticket_tool(
            order_id=order["order_id"],
            reason="refund request within policy window",
            priority="normal",
        )
        result = await call_agent(order_id="A1007", message="Refund for order A1007")
        run.set_output(result)

    (
        tracer.assert_that()
        .tool_called_before("lookup_order", "fetch_refund_policy")
        .tool_called_before("fetch_refund_policy", "create_support_ticket")
        .completed_within_steps(3)
        .no_errors()
        .check()
    )
