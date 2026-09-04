"""Public package exports for deki_smpc."""

from .clients import FedAvgClient, PreparedRound
from .protocol.errors import (
    AggregateIntegrityError,
    ArtifactValidationError,
    AuthenticationError,
    ConfigurationError,
    DekiSMPCError,
    ProtocolVersionError,
    RoundConflictError,
    RoundExpiredError,
    RoundFailedError,
    RoundTimeoutError,
    TransportError,
)

__all__: list[str] = [
    "AggregateIntegrityError",
    "ArtifactValidationError",
    "AuthenticationError",
    "ConfigurationError",
    "DekiSMPCError",
    "FedAvgClient",
    "PreparedRound",
    "ProtocolVersionError",
    "RoundConflictError",
    "RoundExpiredError",
    "RoundFailedError",
    "RoundTimeoutError",
    "TransportError",
]
