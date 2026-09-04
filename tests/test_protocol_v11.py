from __future__ import annotations

import base64
import json
import math
import pickle
from pathlib import Path

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from deki_smpc import PreparedRound
from deki_smpc.protocol.errors import ArtifactValidationError
from deki_smpc.protocol.models import RoundContext
from deki_smpc.protocol.schema import ModelSchema
from deki_smpc.protocol.tree import (
    decrypt_final_key,
    decrypt_task_artifact,
    derive_tree_plan,
    encrypt_final_key,
    encrypt_task_artifact,
    tree_plan_hash,
)


def _context() -> RoundContext:
    return RoundContext("federation", "round", "client-0", "1.1", "a" * 64, "b" * 64, "EQUAL_WEIGHTED", 24)


def _manifest(count: int) -> list[dict[str, str]]:
    return [
        {"client_id": f"client-{index}", "public_key": f"key-{index}", "signature": f"signature-{index}"}
        for index in range(count)
    ]


@pytest.mark.parametrize("count", [3, 4, 5, 6, 7, 8, 12, 13, 20, 65])
def test_tree_topology_is_deterministic_balanced_and_complete(count: int) -> None:
    first = derive_tree_plan(_context(), _manifest(count))
    second = derive_tree_plan(_context(), _manifest(count))

    assert first == second
    assert sorted(first["participants"]) == sorted(f"client-{index}" for index in range(count))
    sizes = [len(group["members"]) for group in first["groups"]]
    assert sizes == [5] if count == 5 else all(size in {3, 4} for size in sizes)
    assert first["tree_levels"] == math.ceil(math.log2(len(first["groups"])))
    starts = [task for task in first["tasks"] if task["action"] == "GROUP_START"]
    assert len(starts) == len(first["groups"])
    assert all(not task["dependencies"] for task in starts)
    positions = {task["task_id"]: index for index, task in enumerate(first["tasks"])}
    assert all(
        positions[dependency] < positions[task["task_id"]]
        for task in first["tasks"]
        for dependency in task["dependencies"]
    )


def test_protocol_11_tree_fixture_is_canonical() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "protocol-v1.1-tree.json").read_text())
    values = fixture["context"]
    context = RoundContext(
        values["federation_id"],
        values["round_id"],
        "client-0",
        fixture["protocol_version"],
        values["model_schema_hash"],
        values["participant_manifest_hash"],
        values["aggregation_policy"],
        values["precision_bits"],
    )
    plan = derive_tree_plan(context, fixture["ephemeral_manifest"])
    assert plan == fixture["plan"]
    assert tree_plan_hash(plan) == fixture["plan_hash"]


def test_tree_artifact_binds_context_and_detects_tampering() -> None:
    sender_x, receiver_x = X25519PrivateKey.generate(), X25519PrivateKey.generate()
    sender_identity = Ed25519PrivateKey.generate()
    schema = ModelSchema.from_state_dict({"weight": torch.tensor([1.0, 2.0])})
    context = _context()
    task = derive_tree_plan(context, _manifest(3))["tasks"][0]
    plan_hash = tree_plan_hash(derive_tree_plan(context, _manifest(3)))
    tensors = {"weight": torch.tensor([torch.iinfo(torch.int64).max, 7], dtype=torch.int64)}

    ciphertext, value, signature = encrypt_task_artifact(
        tensors,
        task,
        plan_hash,
        schema,
        context,
        sender_x,
        receiver_x.public_key(),
        sender_identity,
    )
    clear = decrypt_task_artifact(
        ciphertext,
        value,
        signature,
        task,
        plan_hash,
        schema,
        context,
        receiver_x,
        sender_x.public_key(),
        sender_identity.public_key(),
    )
    assert torch.equal(clear["weight"], tensors["weight"])

    altered = bytearray(ciphertext)
    altered[0] ^= 1
    with pytest.raises(ArtifactValidationError, match="authentication"):
        decrypt_task_artifact(
            bytes(altered),
            value,
            signature,
            task,
            plan_hash,
            schema,
            context,
            receiver_x,
            sender_x.public_key(),
            sender_identity.public_key(),
        )


def test_final_key_uses_shared_group_domain_and_rejects_wrong_seed() -> None:
    identity = Ed25519PrivateKey.generate()
    schema = ModelSchema.from_state_dict({"weight": torch.tensor([1.0, 2.0])})
    context = RoundContext("federation", "round", "root", "1.1", schema.hash, "b" * 64, "EQUAL_WEIGHTED", 24)
    tensors = {"weight": torch.tensor([torch.iinfo(torch.int64).min, 19], dtype=torch.int64)}
    seed = bytes(range(32))
    ciphertext, value, signature = encrypt_final_key(tensors, seed, context, "c" * 64, "root", identity)

    clear = decrypt_final_key(
        ciphertext,
        value,
        signature,
        seed,
        context,
        "c" * 64,
        "root",
        identity.public_key(),
        schema,
    )
    assert torch.equal(clear["weight"], tensors["weight"])
    with pytest.raises(ArtifactValidationError, match="authentication"):
        decrypt_final_key(
            ciphertext,
            value,
            signature,
            b"wrong" * 8,
            context,
            "c" * 64,
            "root",
            identity.public_key(),
            schema,
        )


def test_prepared_round_is_nonserializable() -> None:
    identity = Ed25519PrivateKey.generate()
    raw = base64.b64encode(
        identity.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
    ).decode()
    # Constructing without __init__ lets this test cover the opaque-handle
    # serialization contract without requiring a live multi-client barrier.
    handle = object.__new__(PreparedRound)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(handle)
    assert raw
