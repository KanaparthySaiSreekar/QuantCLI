"""
Custom exceptions for QuantCLI.
"""


class QuantCLIError(Exception):
    """Base exception for all QuantCLI errors."""
    pass


class DataError(QuantCLIError):
    """Raised when there's an issue with data acquisition or processing."""
    pass


class ModelError(QuantCLIError):
    """Raised when there's an issue with model training or inference."""
    pass


class ExecutionError(QuantCLIError):
    """Raised when there's an issue with order execution."""
    pass


class RiskError(QuantCLIError):
    """Raised when risk limits are breached."""
    pass


class ValidationError(QuantCLIError):
    """Raised when validation fails."""
    pass


class ConfigurationError(QuantCLIError):
    """Raised when there's a configuration issue."""
    pass


class ConnectionError(QuantCLIError):
    """Raised when connection to external service fails."""
    pass


class RateLimitError(DataError):
    """Raised when API rate limit is exceeded."""
    pass


class KillSwitchError(RiskError):
    """Raised when kill switch is triggered."""
    pass


class PositionLimitError(RiskError):
    """Raised when position limits are exceeded."""
    pass


class PreTradeCheckError(RiskError):
    """Raised when pre-trade risk check fails."""
    pass
