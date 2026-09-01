"""Prime-field aggregate integrity tags for protocol v1."""

from __future__ import annotations

import hmac
from collections.abc import Mapping

import torch

from .errors import AggregateIntegrityError
from .models import RoundContext
from .prf import pseudorandom_chunks
from .schema import ModelSchema

FIELD_PRIME = 2**127 - 1
TAG_BYTES = 16


def digest_tensors(
    tensors: Mapping[str, torch.Tensor],
    schema: ModelSchema,
    seed: bytes,
    context: RoundContext,
) -> int:
    element_count = sum(tensors[entry.name].numel() for entry in schema.uploaded_entries)
    raw_coefficients = pseudorandom_chunks(
        seed,
        context.domain("aggregate-integrity"),
        element_count * 16,
        purpose=b"aggregate-integrity",
    )
    coeffs = (
        int.from_bytes(chunk[offset : offset + 16], "big") % FIELD_PRIME
        for chunk in raw_coefficients
        for offset in range(0, len(chunk), 16)
    )
    total = 0
    for entry in schema.uploaded_entries:
        tensor = tensors[entry.name].detach().to("cpu", torch.int64).contiguous().view(-1)
        for value in tensor.tolist():
            total = (total + next(coeffs) * (int(value) % FIELD_PRIME)) % FIELD_PRIME
    return total


def encode_tag(tag: int) -> str:
    if not 0 <= tag < FIELD_PRIME:
        raise ValueError("tag is outside the integrity field")
    return tag.to_bytes(TAG_BYTES, "big").hex()


def decode_tag(encoded: str) -> int:
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as exc:
        raise AggregateIntegrityError("malformed aggregate integrity tag") from exc
    if len(raw) != TAG_BYTES:
        raise AggregateIntegrityError("malformed aggregate integrity tag")
    value = int.from_bytes(raw, "big")
    if value >= FIELD_PRIME:
        raise AggregateIntegrityError("aggregate integrity tag is outside the field")
    return value


def verify_tag(actual: int, expected: int) -> None:
    if not hmac.compare_digest(actual.to_bytes(TAG_BYTES, "big"), expected.to_bytes(TAG_BYTES, "big")):
        raise AggregateIntegrityError("aggregate integrity verification failed")
