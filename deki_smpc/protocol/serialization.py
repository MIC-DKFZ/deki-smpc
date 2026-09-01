"""Non-executable tensor serialization."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from safetensors.torch import load, save

from .errors import ArtifactValidationError
from .schema import ModelSchema

MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def serialize_tensors(tensors: Mapping[str, torch.Tensor]) -> bytes:
    normalized = {
        name: tensor.detach().to(device="cpu", dtype=torch.int64).contiguous() for name, tensor in tensors.items()
    }
    data = save(normalized)
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("tensor artifact exceeds the configured limit")
    return data


def deserialize_tensors(data: bytes, schema: ModelSchema) -> dict[str, torch.Tensor]:
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ArtifactValidationError("tensor artifact exceeds the configured limit")
    try:
        tensors = load(data)
    except Exception as exc:
        raise ArtifactValidationError("invalid safetensors artifact") from exc
    schema.validate_encoded(tensors)
    return tensors
