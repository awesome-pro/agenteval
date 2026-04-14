"""Custom exception hierarchy for agenteval."""


class AgentEvalError(Exception):
    """Base exception for all agenteval errors."""


class AssertionFailure(AgentEvalError):
    """Raised when one or more trace assertions fail."""


class DiscoveryError(AgentEvalError):
    """Raised when test file discovery or import fails."""


class TracerError(AgentEvalError):
    """Raised for invalid tracer usage (e.g., accessing trace before run completes)."""
