from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from itertools import count
from typing import Any

from pydantic import BaseModel, Field

import agenteval
from agenteval import RichReporter, Tracer


class RefundReply(BaseModel):
    eligible: bool
    policy: str
    ticket_id: str
    evidence: list[str] = Field(min_length=2)


async def lookup_order(order_id: str) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "status": "delivered",
        "delivered_days_ago": 2,
        "country": "US",
        "item": "Noise cancelling headphones",
    }


async def fetch_refund_policy(country: str, item: str) -> dict[str, Any]:
    return {
        "country": country,
        "item": item,
        "return_window_days": 30,
        "refund_method": "original payment method",
    }


async def create_support_ticket(order_id: str, reason: str, priority: str) -> str:
    slug = reason.replace(" ", "-").upper()
    return f"TICKET-{priority.upper()}-{order_id}-{slug}"


Tool = Callable[..., Awaitable[Any]]


async def refund_support_agent(
    message: str,
    *,
    lookup_order_tool: Tool,
    fetch_policy_tool: Tool,
    create_ticket_tool: Tool,
    variant: str,
    run_index: int,
) -> str:
    order = await lookup_order_tool(order_id="A1007")

    # Simulate a release regression that appears only on some runs.
    # This is the exact class of issue that one-off manual demos miss.
    if variant == "regression" and run_index in {1, 4}:
        await create_ticket_tool(
            order_id=order["order_id"],
            reason="refund request",
            priority="high",
        )
        return "I opened a ticket. The team will review this."

    policy = await fetch_policy_tool(country=order["country"], item=order["item"])
    ticket_id = await create_ticket_tool(
        order_id=order["order_id"],
        reason="refund request within policy window",
        priority="normal",
    )
    return json.dumps(
        {
            "eligible": order["delivered_days_ago"] <= policy["return_window_days"],
            "policy": f"{policy['return_window_days']}-day policy",
            "ticket_id": ticket_id,
            "evidence": [
                f"order delivered {order['delivered_days_ago']} days ago",
                f"refunds allowed within {policy['return_window_days']} days",
            ],
        }
    )


def make_refund_eval(*, variant: str) -> Callable[[Tracer], Awaitable[None]]:
    run_numbers = count()

    async def test_refund_support_reliability(tracer: Tracer) -> None:
        run_index = next(run_numbers)
        order_tool = tracer.wrap(lookup_order)
        policy_tool = tracer.wrap(fetch_refund_policy)
        ticket_tool = tracer.wrap(create_support_ticket)

        prompt = "I want a refund for order A1007"
        async with tracer.run(input=prompt, variant=variant) as run:
            run.add_metadata(
                model="demo/local-agent",
                release="healthy" if variant == "healthy" else "candidate-regression",
                run_index=run_index,
            )
            result = await refund_support_agent(
                prompt,
                lookup_order_tool=order_tool,
                fetch_policy_tool=policy_tool,
                create_ticket_tool=ticket_tool,
                variant=variant,
                run_index=run_index,
            )
            run.set_output(result)
            run.set_token_usage({"input_tokens": 310, "output_tokens": 96})

        (
            tracer.assert_that()
            .called_tool("lookup_order")
            .called_tool("fetch_refund_policy")
            .called_tool("create_support_ticket")
            .tool_called_before("lookup_order", "fetch_refund_policy")
            .tool_called_before("fetch_refund_policy", "create_support_ticket")
            .tool_called_with_args("create_support_ticket", {"priority": "normal"})
            .completed_within_steps(3)
            .response_matches_schema(RefundReply)
            .response_contains("30-day policy", case_sensitive=False)
            .no_errors()
            .check()
        )

    test_refund_support_reliability.__name__ = (
        f"test_refund_support_reliability_{variant}"
    )
    return test_refund_support_reliability


def print_trace_summary(result: agenteval.TestResult) -> None:
    print("\nTrace inspection:")
    for index, trace in enumerate(result.traces, 1):
        status = "passed" if trace.passed else "failed"
        tools = " -> ".join(call.name for call in trace.tool_calls) or "none"
        print(
            f"  run {index}: {status} | "
            f"{trace.effective_steps} steps | tools: {tools}"
        )
        if trace.assertion_errors:
            print(f"    failure: {trace.assertion_errors[0].splitlines()[0]}")


def print_gate_summary(result: agenteval.TestResult) -> None:
    exit_code = 0 if result.met_threshold else 1
    print(
        "\nGate summary: "
        f"{result.n_passed}/{result.n_runs} passed "
        f"({result.pass_rate:.0%}), threshold={result.threshold:.0%}, "
        f"exit_code={exit_code}, avg_steps={result.avg_steps:.1f}"
    )


def run_case(*, variant: str, threshold: float) -> agenteval.TestResult:
    result = agenteval.run(
        make_refund_eval(variant=variant),
        n=6,
        concurrency=1,
        threshold=threshold,
        tags=["support", "policy", "video-demo"],
    )
    reporter = RichReporter(show_traces=True, show_failures=True)
    reporter.render_result(result)
    print_gate_summary(result)
    print_trace_summary(result)
    return result


def main() -> None:
    print("\n=== AgentEval video demo: healthy release ===")
    healthy = run_case(variant="healthy", threshold=0.8)

    print("\n=== AgentEval video demo: candidate regression ===")
    regression = run_case(variant="regression", threshold=0.8)

    print("\nJSON report excerpt:")
    payload = {
        "healthy": healthy.model_dump(mode="json"),
        "regression": regression.model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2)[:1400])
    print("...")


if __name__ == "__main__":
    main()
