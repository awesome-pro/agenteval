"""agenteval — evaluation toolkit for LLM agents.

Quick start::

    import agenteval

    @agenteval.test(n=20, threshold=0.8)
    async def test_my_agent(tracer: agenteval.Tracer) -> None:
        search = tracer.wrap(my_search_tool)

        async with tracer.run(input="find Python tutorials") as run:
            result = await my_agent("find Python tutorials", search=search)
            run.set_output(result)

        tracer.assert_that().called_tool("my_search_tool").no_errors().check()

    # Run a single test directly:
    result = agenteval.run(test_my_agent, n=10)

    # Discover and run all @agenteval.test functions in a directory:
    suite = agenteval.run_suite("tests/")
"""

from agenteval import adapters
from agenteval.assertions import AssertionSet
from agenteval.models import AgentTrace, SuiteResult, TestResult, ToolCall
from agenteval.registry import test
from agenteval.reporter import RichReporter
from agenteval.runner import run
from agenteval.suite import run_suite
from agenteval.tracer import Tracer

__version__ = "0.1.0"
__all__ = [
    "AgentTrace",
    "AssertionSet",
    "RichReporter",
    "SuiteResult",
    "TestResult",
    "ToolCall",
    "Tracer",
    "adapters",
    "run",
    "run_suite",
    "test",
]
