"""Tests for the Anthropic adapter."""

from __future__ import annotations

from agenteval.adapters.anthropic_adapter import extract_token_usage, wrap_tools
from agenteval.tracer import Tracer


class TestWrapTools:
    def test_wraps_and_records(self) -> None:
        tracer = Tracer()

        async def web_search(query: str) -> str:
            return f"found: {query}"

        import asyncio

        wrapped = wrap_tools({"web_search": web_search}, tracer)
        asyncio.run(wrapped["web_search"](query="llm news"))

        assert len(tracer._tool_calls) == 1
        assert tracer._tool_calls[0].name == "web_search"
        assert tracer._tool_calls[0].arguments == {"query": "llm news"}


class TestExtractTokenUsage:
    def test_extracts_input_output(self) -> None:
        class MockUsage:
            input_tokens = 200
            output_tokens = 80

        class MockResponse:
            usage = MockUsage()

        result = extract_token_usage(MockResponse())
        assert result == {"input_tokens": 200, "output_tokens": 80}

    def test_includes_cache_tokens(self) -> None:
        class MockUsage:
            input_tokens = 200
            output_tokens = 80
            cache_read_input_tokens = 50
            cache_creation_input_tokens = 10

        class MockResponse:
            usage = MockUsage()

        result = extract_token_usage(MockResponse())
        assert result is not None
        assert result["cache_read_input_tokens"] == 50
        assert result["cache_creation_input_tokens"] == 10

    def test_returns_none_when_no_usage(self) -> None:
        class MockResponse:
            usage = None

        assert extract_token_usage(MockResponse()) is None
