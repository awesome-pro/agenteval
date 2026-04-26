# agenteval Examples

These examples show how to use agenteval to test agent behavior, not exact text output.

Run all examples:

```bash
agenteval run examples/
```

Run one domain:

```bash
agenteval run examples/ --tag support
agenteval run examples/ --tag finance
agenteval run examples/ --tag incident
```

## What The Examples Prove

### Customer Support

`test_agent_checks_order_status` is the smallest smoke test. It proves that an
agent can call an order lookup tool, produce an answer, and pass repeated runs.

`test_refund_policy_flow` is closer to a production support workflow. It checks
that the agent:

- looks up the order
- fetches the refund policy
- opens a support ticket only after gathering evidence
- creates the ticket with the expected priority
- explains the policy in the final answer

### Finance Safety

`test_finance_safety_gate` demonstrates a high-stakes safety eval. The agent can
read portfolio and market data, but the test verifies that it never calls the
`execute_trade` tool. It also validates the final response against a Pydantic
schema and requires human approval for high-risk advice.

### Incident Triage

`test_incident_triage_flow` models an ops/SRE agent. It checks that the agent
collects metrics and logs before opening an incident, assigns the expected
severity, and includes useful evidence in the response.

## Why This Is Useful

LLM agents are not always deterministic. Instead of asserting one exact output,
agenteval checks behavior over repeated runs:

- which tools were called
- which tools were never called
- whether tools were called in the right order
- whether the response matches a schema
- whether the run stayed within step and time limits
- whether the pass rate met a reliability threshold

