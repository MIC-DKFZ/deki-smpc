"""Canonical model schema and aggregation-policy selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import torch

from .errors import ArtifactValidationError

MAX_TENSORS = 4096
MAX_RANK = 8
MAX_DIMENSION = 1_000_000_000
RESERVED_PREFIX = "__deki_"


class TensorPolicy(StrEnum):
    MEAN = "MEAN"
    SUM = "SUM"
    KEEP_LOCAL = "KEEP_LOCAL"


SUPPORTED_FLOATS = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
SUPPORTED_INTS = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class TensorSchema:
    name: str
    shape: tuple[int, ...]
    dtype: str
    policy: TensorPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "name": self.name,
            "policy": self.policy.value,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class ModelSchema:
    entries: tuple[TensorSchema, ...]
    precision_bits: int
    aggregation_policy: str = "EQUAL_WEIGHTED"

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        precision_bits: int = 24,
        policies: Mapping[str, TensorPolicy | str] | None = None,
    ) -> ModelSchema:
        if len(state_dict) > MAX_TENSORS:
            raise ArtifactValidationError("model contains too many tensors")
        entries: list[TensorSchema] = []
        for name in sorted(state_dict):
            tensor = state_dict[name]
            if not name or name.startswith(RESERVED_PREFIX) or "\x00" in name:
                raise ArtifactValidationError(f"invalid or reserved tensor name: {name!r}")
            if not torch.is_tensor(tensor) or tensor.is_sparse or tensor.is_quantized:
                raise ArtifactValidationError(f"unsupported tensor {name!r}")
            if tensor.is_complex() or tensor.dtype not in SUPPORTED_FLOATS | SUPPORTED_INTS:
                raise ArtifactValidationError(f"unsupported dtype for tensor {name!r}")
            shape = tuple(int(v) for v in tensor.shape)
            if len(shape) > MAX_RANK or any(v < 0 or v > MAX_DIMENSION for v in shape):
                raise ArtifactValidationError(f"invalid shape for tensor {name!r}")
            raw_policy = (policies or {}).get(name)
            if raw_policy is None:
                policy = TensorPolicy.MEAN if tensor.dtype in SUPPORTED_FLOATS else TensorPolicy.KEEP_LOCAL
            else:
                policy = TensorPolicy(raw_policy)
            if policy in {TensorPolicy.MEAN, TensorPolicy.SUM} and tensor.dtype not in SUPPORTED_FLOATS:
                raise ArtifactValidationError(f"{policy.value} requires a floating tensor: {name!r}")
            entries.append(TensorSchema(name, shape, str(tensor.dtype).removeprefix("torch."), policy))
        return cls(tuple(entries), precision_bits)

    def as_dict(self) -> dict[str, object]:
        return {
            "aggregation_policy": self.aggregation_policy,
            "entries": [entry.as_dict() for entry in self.entries],
            "precision_bits": self.precision_bits,
        }

    @property
    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()

    @property
    def uploaded_entries(self) -> tuple[TensorSchema, ...]:
        return tuple(entry for entry in self.entries if entry.policy in {TensorPolicy.MEAN, TensorPolicy.SUM})

    def validate_encoded(self, tensors: Mapping[str, torch.Tensor]) -> None:
        expected = {entry.name: entry for entry in self.uploaded_entries}
        if set(tensors) != set(expected):
            raise ArtifactValidationError("tensor names do not match committed schema")
        for name, entry in expected.items():
            tensor = tensors[name]
            if tensor.dtype != torch.int64 or tuple(tensor.shape) != entry.shape:
                raise ArtifactValidationError(f"encoded tensor {name!r} has an invalid dtype or shape")
