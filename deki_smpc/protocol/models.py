"""Wire-visible v1 enums and immutable context."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

PROTOCOL_VERSION = "1.0"
# Pairwise masking only hides an update when at least three sites take part:
# a single site masks with zero, and with two sites each can subtract its own
# update from the aggregate to recover the other.
MINIMUM_PARTICIPANTS = 3


class RoundState(StrEnum):
    CREATED = "CREATED"
    REGISTRATION_OPEN = "REGISTRATION_OPEN"
    KEY_SETUP = "KEY_SETUP"
    UPDATE_COLLECTION = "UPDATE_COLLECTION"
    AGGREGATING = "AGGREGATING"
    RESULT_READY = "RESULT_READY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    EXPIRED = "EXPIRED"


class AggregationPolicy(StrEnum):
    EQUAL_WEIGHTED = "EQUAL_WEIGHTED"


@dataclass(frozen=True)
class RoundContext:
    federation_id: str
    round_id: str
    client_id: str
    protocol_version: str
    model_schema_hash: str
    participant_manifest_hash: str
    aggregation_policy: str
    precision_bits: int

    def domain(self, purpose: str) -> bytes:
        fields = (
            purpose,
            self.protocol_version,
            self.federation_id,
            self.round_id,
            self.model_schema_hash,
            self.participant_manifest_hash,
            self.aggregation_policy,
            str(self.precision_bits),
        )
        return ("\x00".join(fields)).encode("utf-8")
