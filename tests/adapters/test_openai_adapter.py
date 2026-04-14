"""Tests for the OpenAI adapter."""

from __future__ import annotations

from agenteval.adapters.openai_adapter import extract_token_usage, wrap_tools
from agenteval.tracer import Tracer


class TestWrapTools:
    def test_wraps_all_tools(self) -> None:
        tracer = Tracer()

        def search(q: str) -> str:
            return f"results: {q}"

        def calculator(x: int, y: int) -> int:
            return x + y

        wrapped = wrap_tools({"search": search, "calculator": calculator}, tracer)

        assert set(wrapped.keys()) == {"search", "calculator"}
        assert wrapped["search"](q="python") == "results: python"
        assert len(tracer._tool_calls) == 1
        assert tracer._tool_calls[0].name == "search"

        wrapped["calculator"](x=2, y=3)
        assert len(tracer._tool_calls) == 2
        assert tracer._tool_calls[1].name == "calculator"

    def test_wrapped_tool_records_error(self) -> None:
        tracer = Tracer()

        def broken(x: int) -> int:
            raise ValueError("nope")

        wrapped = wrap_tools({"broken": broken}, tracer)
        try:
            wrapped["broken"](x=1)
        except ValueError:
            pass

        assert tracer._tool_calls[0].error is not None


class TestExtractTokenUsage:
    def test_extracts_from_mock_response(self) -> None:
        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            usage = MockUsage()

        result = extract_token_usage(MockResponse())
        assert result == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    def test_returns_none_when_no_usage(self) -> None:
        class MockResponse:
            usage = None

        assert extract_token_usage(MockResponse()) is None

    def test_returns_none_for_plain_object(self) -> None:
        assert extract_token_usage(object()) is None
