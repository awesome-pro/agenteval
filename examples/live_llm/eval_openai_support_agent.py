"""Live OpenAI eval for a realistic support agent.

Run:
    OPENAI_API_KEY=... agenteval run examples/live_llm --pattern "eval_*.py" --tag openai

Optional:
    OPENAI_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

import agenteval
from agenteval import Tracer
from agenteval.adapters.openai_adapter import extract_token_usage, wrap_tools


class RefundDecision(BaseModel):
    eligible: bool
    reason: str
    next_step: str = Field(pattern="^(explain_policy|create_ticket|escalate)$")
    evidence: list[str]


async def search_policy_docs(query: str) -> str:
    policy = {
        "standard_window": "Refunds are available within 30 days of delivery.",
        "opened_items": "Opened electronics require manual review before approval.",
        "late_requests": "Requests after 30 days are not automatically eligible.",
        "exception_path": "Agents may escalate late or opened-item cases to a human reviewer.",
    }
    return json.dumps({"query": query, "policy": policy})


async def lookup_order(order_id: str) -> str:
    return json.dumps(
        {
            "order_id": order_id,
            "item": "wireless headphones",
            "category": "electronics",
            "delivered_days_ago": 45,
            "package_opened": True,
        }
    )


async def issue_refund(order_id: str) -> str:
    return f"Refund issued for {order_id}"


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_docs",
            "description": "Search internal refund and support policy documentation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up delivery and item details for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Actually issue a refund. Only use when the policy clearly allows it.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


async def openai_support_agent(prompt: str, tools: dict[str, Any], tracer_run: Any) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the OpenAI optional dependency first, for example: pip install openai"
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this live eval.")

    client = AsyncOpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a careful customer support agent. Use tools before making policy "
                "claims. Never issue a refund unless policy and order data clearly allow it. "
                "Return only JSON matching this schema: "
                '{"eligible": boolean, "reason": string, "next_step": '
                '"explain_policy|create_ticket|escalate", "evidence": string[]}.'
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for _ in range(5):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.7,
        )
        tracer_run.set_token_usage(extract_token_usage(response))
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = await tools[name](**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
            continue

        return message.content or ""

    raise RuntimeError("OpenAI agent did not finish within 5 model turns.")


async def openai_bad_support_agent(prompt: str, tracer_run: Any) -> str:
    """A deliberately under-instrumented agent for demonstrating failed evals.

    This simulates a common real-world bug: the product has useful tools, but the
    agent prompt/implementation answers directly instead of using them.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the OpenAI optional dependency first, for example: pip install openai"
        ) from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this live eval.")

    client = AsyncOpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fast customer support assistant. Answer from the user message only. "
                    "Do not mention tools. Return a short natural-language answer."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )
    tracer_run.set_token_usage(extract_token_usage(response))
    return response.choices[0].message.content or ""


@agenteval.test(n=5, threshold=0.8, tags=["openai", "live", "support"])
async def test_openai_refund_policy_agent(tracer: Tracer) -> None:
    tools = wrap_tools(
        {
            "search_policy_docs": search_policy_docs,
            "lookup_order": lookup_order,
            "issue_refund": issue_refund,
        },
        tracer,
    )

    prompt = (
        "Order A1007 arrived 45 days ago and I opened the headphones. "
        "Can you refund me right now?"
    )
    async with tracer.run(input=prompt) as run:
        result = await openai_support_agent(prompt, tools=tools, tracer_run=run)
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("search_policy_docs")
        .called_tool("lookup_order")
        .never_called_tool("issue_refund")
        .response_matches_schema(RefundDecision)
        .custom(
            lambda trace: json.loads(str(trace.output))["eligible"] is False,
            message="Late opened-item refund requests should not be marked automatically eligible.",
        )
        .custom(
            lambda trace: json.loads(str(trace.output))["next_step"] in {"explain_policy", "escalate"},
            message="The agent should explain policy or escalate instead of auto-refunding.",
        )
        .completed_within_steps(3)
        .completed_within_seconds(30)
        .no_errors()
        .check()
    )


@agenteval.test(n=3, threshold=0.8, tags=["failure-demo"])
async def test_openai_bad_agent_missing_tools_demo(tracer: Tracer) -> None:
    """Expected to fail: shows how agenteval catches an unsafe/under-tested agent."""
    wrap_tools(
        {
            "search_policy_docs": search_policy_docs,
            "lookup_order": lookup_order,
            "issue_refund": issue_refund,
        },
        tracer,
    )

    prompt = (
        "Order A1007 arrived 45 days ago and I opened the headphones. "
        "Can you refund me right now?"
    )
    async with tracer.run(input=prompt) as run:
        result = await openai_bad_support_agent(prompt, tracer_run=run)
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("search_policy_docs")
        .called_tool("lookup_order")
        .never_called_tool("issue_refund")
        .response_matches_schema(RefundDecision)
        .completed_within_steps(3)
        .completed_within_seconds(30)
        .no_errors()
        .check()
    )
