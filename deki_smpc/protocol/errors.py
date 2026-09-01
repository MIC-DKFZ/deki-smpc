"""Typed errors exposed by the v1 client."""

from __future__ import annotations


class DekiSMPCError(Exception):
    """Base class for all client errors."""

    def __init__(self, message: str, *, round_id: str | None = None) -> None:
        super().__init__(message)
        self.round_id = round_id


class ConfigurationError(DekiSMPCError):
    pass


class AuthenticationError(DekiSMPCError):
    pass


class ProtocolVersionError(DekiSMPCError):
    pass


class RoundConflictError(DekiSMPCError):
    pass


class RoundExpiredError(DekiSMPCError):
    pass


class RoundFailedError(DekiSMPCError):
    pass


class ArtifactValidationError(DekiSMPCError):
    pass


class AggregateIntegrityError(DekiSMPCError):
    pass


class TransportError(DekiSMPCError):
    pass


class RoundTimeoutError(TransportError):
    pass


ERROR_TYPES = {
    "AUTHENTICATION_FAILED": AuthenticationError,
    "AUTHORIZATION_FAILED": AuthenticationError,
    "PROTOCOL_VERSION_UNSUPPORTED": ProtocolVersionError,
    "ROUND_CONFLICT": RoundConflictError,
    "IDEMPOTENCY_CONFLICT": RoundConflictError,
    "ROUND_EXPIRED": RoundExpiredError,
    "ROUND_FAILED": RoundFailedError,
    "ARTIFACT_INVALID": ArtifactValidationError,
    "SCHEMA_MISMATCH": ArtifactValidationError,
}
