from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from typing import Any

app = FastAPI(title="Support Agent API")


class RefundRequest(BaseModel):
    order_id: str
    message: str


class RefundResponse(BaseModel):
    eligible: bool
    policy: str
    ticket_id: str
    evidence: list[str] = Field(min_length=2)


class OrderInfo(BaseModel):
    order_id: str
    status: str
    delivered_days_ago: int
    country: str
    item: str


class PolicyInfo(BaseModel):
    country: str
    item: str
    return_window_days: int
    refund_method: str


# Internal tool endpoints that the agent uses
@app.get("/tools/orders/{order_id}", response_model=OrderInfo)
async def lookup_order(order_id: str) -> OrderInfo:
    """Tool endpoint: fetch order details."""
    return OrderInfo(
        order_id=order_id,
        status="delivered",
        delivered_days_ago=2,
        country="US",
        item="Noise cancelling headphones",
    )


@app.get("/tools/policy", response_model=PolicyInfo)
async def fetch_refund_policy(country: str, item: str) -> PolicyInfo:
    """Tool endpoint: fetch refund policy for a country and item."""
    return PolicyInfo(
        country=country,
        item=item,
        return_window_days=30,
        refund_method="original payment method",
    )


class TicketCreate(BaseModel):
    order_id: str
    reason: str
    priority: str


@app.post("/tools/tickets")
async def create_support_ticket(ticket: TicketCreate) -> dict[str, str]:
    """Tool endpoint: create a support ticket."""
    slug = ticket.reason.replace(" ", "-").upper()
    ticket_id = f"TICKET-{ticket.priority.upper()}-{ticket.order_id}-{slug}"
    return {"ticket_id": ticket_id}


# Agent endpoint that orchestrates tool calls
@app.post("/agent/refund", response_model=RefundResponse)
async def refund_support_agent(request: RefundRequest) -> RefundResponse:
    """Agent endpoint: handle refund requests by calling internal tools."""
    if "refund" not in request.message.lower():
        raise HTTPException(status_code=400, detail="Request does not mention refund")

    # Call internal tools (in a real agent, this would be LLM-driven)
    order = await lookup_order(order_id=request.order_id)
    policy = await fetch_refund_policy(country=order.country, item=order.item)

    if order.delivered_days_ago > policy.return_window_days:
        raise HTTPException(
            status_code=400,
            detail=f"Order delivered {order.delivered_days_ago} days ago, outside {policy.return_window_days}-day window",
        )

    ticket_result = await create_support_ticket(
        TicketCreate(
            order_id=order.order_id,
            reason="refund request within policy window",
            priority="normal",
        )
    )

    return RefundResponse(
        eligible=True,
        policy=f"{policy.return_window_days}-day policy",
        ticket_id=ticket_result["ticket_id"],
        evidence=[
            f"order delivered {order.delivered_days_ago} days ago",
            f"refunds allowed within {policy.return_window_days} days",
        ],
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
