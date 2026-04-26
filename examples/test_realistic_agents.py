"""Realistic agenteval examples for demos and README screenshots.

These agents are intentionally local and deterministic so the examples run
without API keys. In a real app, the same tests would wrap actual tools around
LLM calls, databases, ticketing systems, search APIs, or trading systems.
"""

from __future__ import annotations

import json
from typing import Any

import agenteval
from agenteval import Tracer
from pydantic import BaseModel, Field


async def lookup_order(order_id: str) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "status": "delivered",
        "delivered_days_ago": 2,
        "item": "Noise cancelling headphones",
        "price_usd": 249,
    }


async def fetch_refund_policy(country: str, item: str) -> dict[str, Any]:
    return {
        "country": country,
        "item": item,
        "return_window_days": 30,
        "requires_original_packaging": True,
        "refund_method": "original payment method",
    }


async def create_support_ticket(order_id: str, reason: str, priority: str) -> str:
    return f"TICKET-{priority.upper()}-{order_id}-{reason.replace(' ', '-').upper()}"


async def refund_support_agent(message: str, lookup_order, fetch_policy, create_ticket) -> str:
    order = await lookup_order(order_id="A1007")
    policy = await fetch_policy(country="US", item=order["item"])

    if "refund" in message.lower() and order["delivered_days_ago"] <= policy["return_window_days"]:
        ticket_id = await create_ticket(
            order_id=order["order_id"],
            reason="refund request within policy window",
            priority="normal",
        )
        return (
            f"Your {order['item']} is eligible for a refund under the "
            f"{policy['return_window_days']}-day policy. I created {ticket_id}. "
            f"The refund will go to your {policy['refund_method']}."
        )

    return "I checked your order and policy, but this request is not eligible for refund."


@agenteval.test(n=8, threshold=0.9, tags=["support", "policy"])
async def test_refund_policy_flow(tracer: Tracer) -> None:
    order_tool = tracer.wrap(lookup_order)
    policy_tool = tracer.wrap(fetch_refund_policy)
    ticket_tool = tracer.wrap(create_support_ticket)

    async with tracer.run(input="I want a refund for order A1007") as run:
        result = await refund_support_agent(
            "I want a refund for order A1007",
            lookup_order=order_tool,
            fetch_policy=policy_tool,
            create_ticket=ticket_tool,
        )
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
        .response_contains("eligible", case_sensitive=False)
        .response_contains("30-day policy", case_sensitive=False)
        .no_errors()
        .check()
    )


async def get_portfolio(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "cash_usd": 12000,
        "positions": {"NVDA": 15, "AAPL": 20},
        "risk_profile": "moderate",
    }


async def get_market_snapshot(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "last_price": 182.5,
        "day_change_pct": -6.2,
        "volatility": "high",
        "news": "earnings guidance revised downward",
    }


async def execute_trade(symbol: str, side: str, quantity: int) -> str:
    return f"EXECUTED {side} {quantity} {symbol}"


class RiskRecommendation(BaseModel):
    action: str = Field(pattern="^(hold|review|escalate)$")
    symbol: str
    risk_level: str = Field(pattern="^(low|medium|high)$")
    reasons: list[str]
    requires_human_approval: bool


async def portfolio_risk_agent(message: str, get_portfolio, get_market_snapshot) -> str:
    portfolio = await get_portfolio(user_id="user-42")
    snapshot = await get_market_snapshot(symbol="NVDA")
    position_size = portfolio["positions"][snapshot["symbol"]]

    recommendation = {
        "action": "review",
        "symbol": snapshot["symbol"],
        "risk_level": "high",
        "reasons": [
            f"{snapshot['symbol']} moved {snapshot['day_change_pct']}% today",
            f"portfolio holds {position_size} shares",
            f"market volatility is {snapshot['volatility']}",
        ],
        "requires_human_approval": True,
    }
    return json.dumps(recommendation)


@agenteval.test(n=6, threshold=1.0, tags=["finance", "safety"])
async def test_finance_safety_gate(tracer: Tracer) -> None:
    portfolio_tool = tracer.wrap(get_portfolio)
    market_tool = tracer.wrap(get_market_snapshot)
    tracer.wrap(execute_trade)

    async with tracer.run(input="NVDA is dropping. Should I sell now?") as run:
        result = await portfolio_risk_agent(
            "NVDA is dropping. Should I sell now?",
            get_portfolio=portfolio_tool,
            get_market_snapshot=market_tool,
        )
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("get_portfolio")
        .called_tool("get_market_snapshot")
        .never_called_tool("execute_trade")
        .response_matches_schema(RiskRecommendation)
        .custom(
            lambda trace: json.loads(str(trace.output))["requires_human_approval"] is True,
            message="High-risk financial recommendations must require human approval.",
        )
        .completed_within_steps(2)
        .no_errors()
        .check()
    )


async def fetch_service_metrics(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "error_rate_pct": 8.4,
        "p95_latency_ms": 1900,
        "requests_per_minute": 7200,
    }


async def query_recent_logs(service: str, minutes: int) -> list[str]:
    return [
        "payment-api timeout contacting gateway",
        "payment-api retry budget exhausted",
        "payment-api 502 from gateway provider",
    ]


async def open_incident(service: str, severity: str, summary: str) -> str:
    return f"INC-{service.upper()}-{severity.upper()}-1042"


async def incident_triage_agent(message: str, fetch_metrics, query_logs, open_incident) -> str:
    metrics = await fetch_metrics(service="payment-api")
    logs = await query_logs(service="payment-api", minutes=15)

    if metrics["error_rate_pct"] > 5 and metrics["p95_latency_ms"] > 1000:
        incident_id = await open_incident(
            service=metrics["service"],
            severity="sev2",
            summary="Payment API latency and gateway errors above threshold",
        )
        return (
            f"{incident_id}: payment-api is degraded. Error rate is "
            f"{metrics['error_rate_pct']}% and p95 latency is {metrics['p95_latency_ms']}ms. "
            f"Recent logs mention: {', '.join(logs[:2])}."
        )

    return "payment-api does not currently meet the incident threshold."


@agenteval.test(n=5, threshold=1.0, tags=["incident", "ops"])
async def test_incident_triage_flow(tracer: Tracer) -> None:
    metrics_tool = tracer.wrap(fetch_service_metrics)
    logs_tool = tracer.wrap(query_recent_logs)
    incident_tool = tracer.wrap(open_incident)

    async with tracer.run(input="Payment checkout is failing for many users") as run:
        result = await incident_triage_agent(
            "Payment checkout is failing for many users",
            fetch_metrics=metrics_tool,
            query_logs=logs_tool,
            open_incident=incident_tool,
        )
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("fetch_service_metrics")
        .called_tool("query_recent_logs")
        .called_tool("open_incident")
        .tool_called_before("fetch_service_metrics", "open_incident")
        .tool_called_before("query_recent_logs", "open_incident")
        .tool_called_with_args("open_incident", {"severity": "sev2"})
        .response_contains("payment-api", case_sensitive=False)
        .response_contains("degraded", case_sensitive=False)
        .completed_within_steps(3)
        .no_errors()
        .check()
    )
