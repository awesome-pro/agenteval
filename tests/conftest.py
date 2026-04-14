"""Shared pytest fixtures and configuration."""

import pytest

from agenteval.registry import TestRegistry


@pytest.fixture(autouse=False)
def clean_registry() -> None:
    """Reset the global TestRegistry before a test. Use when testing registry/suite behavior."""
    TestRegistry.reset()
    yield
    TestRegistry.reset()
