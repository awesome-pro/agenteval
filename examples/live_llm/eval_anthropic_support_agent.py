"""Live Anthropic eval for a realistic support agent.

Run:
    ANTHROPIC_API_KEY=... agenteval run examples/live_llm --pattern "eval_*.py" --tag anthropic

Optional:
    ANTHROPIC_MODEL=claude-sonnet-4-6
"""

from __future__ import annotations

import json
import os
import random
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

import agenteval
from agenteval import Tracer
from agenteval.adapters.anthropic_adapter import extract_token_usage, wrap_tools

if TYPE_CHECKING:
    from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam, ToolUseBlock


class TriageDecision(BaseModel):
    severity: str = Field(min_length=1)
    action: str = Field(min_length=1)
    evidence: list[str]
    customer_message: str


@dataclass(frozen=True)
class IncidentScenario:
    service: str
    user_prompt: str
    error_rate_pct: float
    p95_latency_ms: int
    log_events: list[str]
    expected_severity: str
    expected_action: str
    should_page: bool


DEFAULT_SCENARIO = IncidentScenario(
    service="payment-api",
    user_prompt="Checkout is timing out for customers in us-east. Should we escalate?",
    error_rate_pct=7.8,
    p95_latency_ms=2200,
    log_events=[
        "checkout timeout connecting to payment gateway",
        "payment-api 502 response spike",
        "retry queue depth above threshold",
    ],
    expected_severity="high",
    expected_action="escalate",
    should_page=True,
)


RANDOM_SCENARIOS = [
    DEFAULT_SCENARIO,
    IncidentScenario(
        service="catalog-api",
        user_prompt="Product pages feel a little slow for some users. Is this an incident?",
        error_rate_pct=0.7,
        p95_latency_ms=420,
        log_events=[
            "catalog cache refresh completed",
            "normal background index update",
            "no 5xx spike detected",
        ],
        expected_severity="low",
        expected_action="monitor",
        should_page=False,
    ),
    IncidentScenario(
        service="image-api",
        user_prompt="Image uploads are slightly slower after deploy, but users are not blocked.",
        error_rate_pct=1.1,
        p95_latency_ms=760,
        log_events=[
            "image-api thumbnail worker backlog cleared",
            "temporary object storage latency warning",
            "no customer-facing upload failures detected",
        ],
        expected_severity="low",
        expected_action="monitor",
        should_page=False,
    ),
    IncidentScenario(
        service="email-worker",
        user_prompt="Password reset emails are delayed by several minutes. Should we wake someone up?",
        error_rate_pct=1.8,
        p95_latency_ms=1350,
        log_events=[
            "email-worker queue age above warning threshold",
            "smtp provider rate limit retrying",
            "messages still draining successfully",
        ],
        expected_severity="medium",
        expected_action="open_incident",
        should_page=False,
    ),
    IncidentScenario(
        service="checkout-api",
        user_prompt="A few customers report checkout delays, but orders are still going through.",
        error_rate_pct=2.9,
        p95_latency_ms=980,
        log_events=[
            "checkout-api elevated latency warning",
            "payment provider retries recovered",
            "no sustained 5xx spike",
        ],
        expected_severity="medium",
        expected_action="open_incident",
        should_page=False,
    ),
    IncidentScenario(
        service="billing-api",
        user_prompt="Some invoice downloads are failing for enterprise admins.",
        error_rate_pct=3.4,
        p95_latency_ms=1120,
        log_events=[
            "billing-api pdf renderer timeout warning",
            "invoice download 504 responses elevated",
            "payment collection endpoints healthy",
        ],
        expected_severity="medium",
        expected_action="open_incident",
        should_page=False,
    ),
    IncidentScenario(
        service="search-api",
        user_prompt="Search results intermittently return empty pages during a marketing launch.",
        error_rate_pct=4.9,
        p95_latency_ms=1450,
        log_events=[
            "search-api shard timeout warnings",
            "empty result responses above baseline",
            "traffic spike from campaign landing pages",
        ],
        expected_severity="medium",
        expected_action="open_incident",
        should_page=False,
    ),
    IncidentScenario(
        service="auth-api",
        user_prompt="Login success rate dropped sharply and support tickets are spiking.",
        error_rate_pct=9.4,
        p95_latency_ms=2600,
        log_events=[
            "auth-api database timeout spike",
            "login 503 responses above threshold",
            "session creation retry budget exhausted",
        ],
        expected_severity="high",
        expected_action="escalate",
        should_page=True,
    ),
    IncidentScenario(
        service="orders-api",
        user_prompt="Order placement is failing globally after the latest release.",
        error_rate_pct=12.6,
        p95_latency_ms=3100,
        log_events=[
            "orders-api database connection pool exhausted",
            "order creation 500 responses above critical threshold",
            "rollback health check failed",
        ],
        expected_severity="high",
        expected_action="escalate",
        should_page=True,
    ),
    IncidentScenario(
        service="webhooks-api",
        user_prompt="Webhook delivery failures are causing merchants to miss paid order events.",
        error_rate_pct=8.1,
        p95_latency_ms=2400,
        log_events=[
            "webhooks-api retry queue depth critical",
            "merchant delivery 5xx responses spiking",
            "dead-letter queue growth above threshold",
        ],
        expected_severity="high",
        expected_action="escalate",
        should_page=True,
    ),
    IncidentScenario(
        service="notifications-api",
        user_prompt="Push notifications are delayed, but core checkout and login are healthy.",
        error_rate_pct=2.2,
        p95_latency_ms=890,
        log_events=[
            "notifications fanout lag warning",
            "push provider accepted retries",
            "checkout and auth dependency checks healthy",
        ],
        expected_severity="medium",
        expected_action="open_incident",
        should_page=False,
    ),
    IncidentScenario(
        service="profile-api",
        user_prompt="A small number of users cannot update profile photos.",
        error_rate_pct=0.9,
        p95_latency_ms=510,
        log_events=[
            "profile image resize warnings below threshold",
            "metadata update endpoint healthy",
            "no sustained 5xx increase",
        ],
        expected_severity="low",
        expected_action="monitor",
        should_page=False,
    ),
    IncidentScenario(
        service="warehouse-sync",
        user_prompt="Inventory sync is behind and sellers may oversell popular items.",
        error_rate_pct=5.6,
        p95_latency_ms=1800,
        log_events=[
            "warehouse-sync job lag above threshold",
            "inventory update failures elevated",
            "manual reconciliation queue growing",
        ],
        expected_severity="high",
        expected_action="escalate",
        should_page=True,
    ),
    IncidentScenario(
        service="analytics-api",
        user_prompt="Dashboard charts are stale for the last hour. Customers are asking support.",
        error_rate_pct=2.7,
        p95_latency_ms=1050,
        log_events=[
            "analytics materialization lag above warning threshold",
            "dashboard reads serving stale snapshots",
            "core transaction APIs unaffected",
        ],
        expected_severity="medium",
        expected_action="open_incident",
        should_page=False,
    ),
]


CURRENT_SCENARIO: ContextVar[IncidentScenario] = ContextVar(
    "CURRENT_SCENARIO", default=DEFAULT_SCENARIO
)


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def _decision_from_trace_output(output: Any) -> dict[str, Any]:
    return json.loads(_extract_json_object(str(output)))


def _normalize_severity(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"critical", "sev1", "sev_1", "p0", "p1"}:
        return "high"
    if normalized in {"warning", "warn", "moderate", "investigate"}:
        return "medium"
    return normalized


def _normalize_action(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    if "page" in normalized or "on_call" in normalized or "escalat" in normalized:
        return "escalate"
    if "incident" in normalized or "investigate" in normalized or "ticket" in normalized:
        return "open_incident"
    if "monitor" in normalized or "watch" in normalized:
        return "monitor"
    return normalized


def _service_was_queried(trace: Any) -> bool:
    calls = [call for call in trace.tool_calls if call.name in {"fetch_metrics", "query_logs"}]
    return bool(calls) and all(str(call.arguments.get("service", "")).strip() for call in calls)


async def fetch_metrics(service: str) -> str:
    scenario = CURRENT_SCENARIO.get()
    return json.dumps(
        {
            "service": service,
            "error_rate_pct": scenario.error_rate_pct,
            "p95_latency_ms": scenario.p95_latency_ms,
            "affected_region": "us-east",
        }
    )


async def query_logs(service: str, minutes: int) -> str:
    scenario = CURRENT_SCENARIO.get()
    return json.dumps(
        {
            "service": service,
            "minutes": minutes,
            "events": scenario.log_events,
        }
    )


async def page_oncall(service: str, severity: str) -> str:
    return f"Paged on-call for {service} at {severity}"


ANTHROPIC_TOOLS: list[ToolParam] = [
    {
        "name": "fetch_metrics",
        "description": "Fetch service reliability metrics.",
        "input_schema": {
            "type": "object",
            "properties": {"service": {"type": "string"}},
            "required": ["service"],
        },
    },
    {
        "name": "query_logs",
        "description": "Query recent service logs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "minutes": {"type": "integer"},
            },
            "required": ["service", "minutes"],
        },
    },
    {
        "name": "page_oncall",
        "description": "Page the on-call engineer. Only use when evidence supports escalation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["service", "severity"],
        },
    },
]


def _content_to_text(content: list[Any]) -> str:
    return "".join(
        getattr(block, "text", "") for block in content if getattr(block, "type", "") == "text"
    )


async def anthropic_incident_agent(prompt: str, tools: dict[str, Any], tracer_run: Any) -> str:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "Install the Anthropic optional dependency first, for example: pip install anthropic"
        ) from exc

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY before running this live eval.")

    client = AsyncAnthropic()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    system = (
        "You are an incident triage agent. Gather metrics and logs before deciding. "
        "Page on-call only if service evidence clearly indicates a high-severity incident. "
        "Return only JSON matching this schema: "
        '{"severity": "low|medium|high", "action": "monitor|open_incident|escalate", '
        '"evidence": string[], "customer_message": string}.'
    )

    for _ in range(5):
        response = await client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0.7,
            system=system,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        tracer_run.set_token_usage(extract_token_usage(response))

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[ToolResultBlockParam] = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                tool_use = cast("ToolUseBlock", block)
                result = await tools[tool_use.name](**tool_use.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        return _extract_json_object(_content_to_text(response.content))

    raise RuntimeError("Anthropic agent did not finish within 5 model turns.")


async def anthropic_fast_triage_agent(prompt: str, tools: dict[str, Any], tracer_run: Any) -> str:
    """A faster, less constrained live agent for realistic reliability demos."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "Install the Anthropic optional dependency first, for example: pip install anthropic"
        ) from exc

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY before running this live eval.")

    client = AsyncAnthropic()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]
    system = (
        "You are a fast incident triage assistant. Use metrics and logs when helpful. "
        "Use your judgment for borderline cases. Page on-call only for clear severe incidents. "
        "Return JSON with severity, action, evidence, and customer_message. "
        "Do not include extra prose."
    )

    for _ in range(4):
        response = await client.messages.create(
            model=model,
            max_tokens=700,
            temperature=1.0,
            system=system,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        tracer_run.set_token_usage(extract_token_usage(response))

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[ToolResultBlockParam] = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                tool_use = cast("ToolUseBlock", block)
                result = await tools[tool_use.name](**tool_use.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
            continue

        return _extract_json_object(_content_to_text(response.content))

    raise RuntimeError("Anthropic fast triage agent did not finish within 4 model turns.")


@agenteval.test(n=5, threshold=0.8, tags=["anthropic-smoke", "live", "incident"])
async def test_anthropic_incident_triage_agent(tracer: Tracer) -> None:
    tools = wrap_tools(
        {
            "fetch_metrics": fetch_metrics,
            "query_logs": query_logs,
            "page_oncall": page_oncall,
        },
        tracer,
    )

    prompt = "Checkout is timing out for customers in us-east. Should we escalate?"
    async with tracer.run(input=prompt) as run:
        result = await anthropic_incident_agent(prompt, tools=tools, tracer_run=run)
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("fetch_metrics")
        .called_tool("query_logs")
        .called_tool("page_oncall")
        .tool_called_before("fetch_metrics", "page_oncall")
        .tool_called_before("query_logs", "page_oncall")
        .response_matches_schema(TriageDecision)
        .custom(
            lambda trace: json.loads(str(trace.output))["severity"] == "high",
            message="Severe checkout degradation should be classified as high severity.",
        )
        .completed_within_steps(3)
        .completed_within_seconds(30)
        .no_errors()
        .check()
    )


@agenteval.test(n=15, threshold=0.65, tags=["anthropic", "anthropic-random", "live", "incident"])
async def test_anthropic_randomized_incident_reliability(tracer: Tracer) -> None:
    scenario = random.choice(RANDOM_SCENARIOS)
    token = CURRENT_SCENARIO.set(scenario)
    tools = wrap_tools(
        {
            "fetch_metrics": fetch_metrics,
            "query_logs": query_logs,
            "page_oncall": page_oncall,
        },
        tracer,
    )

    try:
        async with tracer.run(input=scenario.user_prompt) as run:
            run.add_metadata(
                service=scenario.service,
                expected_severity=scenario.expected_severity,
                expected_action=scenario.expected_action,
                should_page=scenario.should_page,
            )
            result = await anthropic_fast_triage_agent(
                scenario.user_prompt,
                tools=tools,
                tracer_run=run,
            )
            run.set_output(result)
    finally:
        CURRENT_SCENARIO.reset(token)

    assertions = (
        tracer.assert_that()
        .called_tool("fetch_metrics")
        .called_tool("query_logs")
        .response_matches_schema(TriageDecision)
        .custom(
            _service_was_queried,
            message="Agent should query metrics/logs for a concrete service name.",
        )
        .custom(
            lambda trace: _normalize_severity(
                str(_decision_from_trace_output(trace.output)["severity"])
            )
            == scenario.expected_severity,
            message=f"Expected severity {scenario.expected_severity} for {scenario.service}.",
        )
        .custom(
            lambda trace: _normalize_action(str(_decision_from_trace_output(trace.output)["action"]))
            == scenario.expected_action,
            message=f"Expected action {scenario.expected_action} for {scenario.service}.",
        )
        .completed_within_steps(3)
        .completed_within_seconds(30)
        .no_errors()
    )

    if scenario.should_page:
        assertions.called_tool("page_oncall")
    else:
        assertions.never_called_tool("page_oncall")

    assertions.check()
