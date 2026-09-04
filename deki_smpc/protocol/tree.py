"""Canonical protocol-1.1 topology and encrypted key artifacts."""

from __future__ import annotations

import base64
import hashlib
import math
import secrets
from collections.abc import Mapping
from typing import Any

import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .errors import ArtifactValidationError
from .models import RoundContext
from .prf import pseudorandom_chunks
from .schema import ModelSchema, canonical_json
from .serialization import deserialize_tensors, serialize_tensors


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _shuffle(participants: list[str], seed: bytes) -> list[str]:
    """Fisher-Yates with a specified SHA-256 stream and rejection sampling."""
    shuffled = sorted(participants)
    counter = 0
    for upper in range(len(shuffled) - 1, 0, -1):
        modulus = upper + 1
        limit = 1 << 256
        cutoff = limit - (limit % modulus)
        while True:
            value = int.from_bytes(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest(), "big")
            counter += 1
            if value < cutoff:
                break
        selected = value % modulus
        shuffled[upper], shuffled[selected] = shuffled[selected], shuffled[upper]
    return shuffled


def _group_sizes(participant_count: int) -> list[int]:
    if participant_count < 3:
        raise ValueError("protocol 1.1 requires at least three participants")
    if participant_count == 5:
        return [5]
    group_count = math.ceil(participant_count / 4)
    base, remainder = divmod(participant_count, group_count)
    sizes = [base + (index < remainder) for index in range(group_count)]
    if any(size not in {3, 4} for size in sizes):
        raise ValueError("could not form balanced groups")
    return sizes


def signed_ephemeral_manifest(records: Mapping[str, Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "client_id": client_id,
            "public_key": records[client_id]["public_key"],
            "signature": records[client_id]["signature"],
        }
        for client_id in sorted(records)
    ]


def derive_tree_plan(context: RoundContext, manifest: list[dict[str, str]]) -> dict[str, Any]:
    """Derive the complete canonical task graph from signed ephemeral keys."""
    participants = [record["client_id"] for record in manifest]
    if len(set(participants)) != len(participants):
        raise ValueError("ephemeral manifest contains duplicate participants")
    topology_seed = hashlib.sha256(context.domain("tree-topology") + canonical_json(manifest)).digest()
    shuffled = _shuffle(participants, topology_seed)
    sizes = _group_sizes(len(shuffled))
    groups: list[dict[str, Any]] = []
    offset = 0
    for index, size in enumerate(sizes):
        members = shuffled[offset : offset + size]
        offset += size
        groups.append({"group_id": index, "members": members, "coordinator": members[0]})

    # Nodes are deliberately plain dictionaries so the exact serialized plan
    # is straightforward to reproduce in the server implementation.
    active: list[dict[str, Any]] = [
        {"operator": group["coordinator"], "group": group["group_id"], "children": [], "level": -1} for group in groups
    ]
    tree_nodes: list[dict[str, Any]] = []
    level = 0
    while len(active) > 1:
        next_level: list[dict[str, Any]] = []
        for index in range(0, len(active), 2):
            children = active[index : index + 2]
            action = "TREE_COMBINE" if len(children) == 2 else "TREE_CARRY"
            node = {
                "operator": children[0]["operator"],
                "children": children,
                "level": level,
                "action": action,
                "index": index // 2,
            }
            tree_nodes.append(node)
            next_level.append(node)
        active = next_level
        level += 1
    root = active[0]

    def assign_targets(node: dict[str, Any], target: str) -> None:
        node["target"] = target
        for child in node["children"]:
            assign_targets(child, str(node["operator"]))

    assign_targets(root, str(root["operator"]))
    tasks: list[dict[str, Any]] = []
    for group in groups:
        group_id = int(group["group_id"])
        members = list(group["members"])
        previous = f"g{group_id}-start"
        tasks.append(
            {
                "task_id": previous,
                "action": "GROUP_START",
                "stage": "GROUP",
                "level": 0,
                "sender": members[0],
                "receiver": members[1],
                "dependencies": [],
            }
        )
        for position in range(1, len(members)):
            task_id = f"g{group_id}-add-{position}"
            tasks.append(
                {
                    "task_id": task_id,
                    "action": "GROUP_ADD",
                    "stage": "GROUP",
                    "level": 0,
                    "sender": members[position],
                    "receiver": members[position + 1] if position + 1 < len(members) else members[0],
                    "dependencies": [previous],
                }
            )
            previous = task_id
        leaf = next(node for node in _walk_nodes(root) if node.get("group") == group_id)
        task_id = f"g{group_id}-unblind"
        tasks.append(
            {
                "task_id": task_id,
                "action": "GROUP_UNBLIND",
                "stage": "GROUP",
                "level": 0,
                "sender": members[0],
                "receiver": leaf["target"],
                "dependencies": [previous],
            }
        )
        leaf["task_id"] = task_id

    for node in sorted(tree_nodes, key=lambda item: (int(item["level"]), int(item["index"]))):
        task_id = f"t{node['level']}-{node['index']}"
        dependencies = [str(child["task_id"]) for child in node["children"]]
        tasks.append(
            {
                "task_id": task_id,
                "action": node["action"],
                "stage": "TREE",
                "level": int(node["level"]),
                "sender": node["operator"],
                "receiver": node["target"],
                "dependencies": dependencies,
            }
        )
        node["task_id"] = task_id

    return {
        "protocol_version": "1.1",
        "participants": shuffled,
        "groups": groups,
        "tasks": tasks,
        "root_client_id": root["operator"],
        "root_task_id": root["task_id"],
        "tree_levels": math.ceil(math.log2(len(groups))) if len(groups) > 1 else 0,
    }


def _walk_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    return [node] + [descendant for child in node["children"] for descendant in _walk_nodes(child)]


def tree_plan_hash(plan: Mapping[str, Any]) -> str:
    return _sha256(canonical_json(plan))


def artifact_value(
    task: Mapping[str, Any], plan_hash: str, schema_hash: str, nonce: str, digest: str, size: int
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "action": task["action"],
        "stage": task["stage"],
        "level": task["level"],
        "sender": task["sender"],
        "receiver": task["receiver"],
        "dependencies": task["dependencies"],
        "plan_hash": plan_hash,
        "model_schema_hash": schema_hash,
        "nonce": nonce,
        "ciphertext_digest": digest,
        "ciphertext_size": size,
    }


def final_key_value(
    context: RoundContext, plan_hash: str, root: str, nonce: str, digest: str, size: int
) -> dict[str, Any]:
    return {
        "task_id": "final-key",
        "stage": "FINAL_DISTRIBUTION",
        "level": -1,
        "sender": root,
        "receiver": "ALL_PARTICIPANTS",
        "dependencies": [],
        "round_id": context.round_id,
        "plan_hash": plan_hash,
        "model_schema_hash": context.model_schema_hash,
        "nonce": nonce,
        "ciphertext_digest": digest,
        "ciphertext_size": size,
    }


def _derive_edge_key(
    private: X25519PrivateKey, peer: X25519PublicKey, context: RoundContext, task_id: str, plan_hash: str
) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context.domain(f"tree-edge:{plan_hash}:{task_id}"),
    ).derive(private.exchange(peer))


def _encrypt_with_digest(key: bytes, clear: bytes, nonce: bytes, value_factory: Any) -> tuple[bytes, dict[str, Any]]:
    preliminary = AESGCM(key).encrypt(nonce, clear, b"")
    digest = _sha256(nonce + preliminary[:-16])
    ciphertext_size = len(preliminary)
    preliminary = b""
    value = value_factory(base64.b64encode(nonce).decode("ascii"), digest, ciphertext_size)
    ciphertext = AESGCM(key).encrypt(nonce, clear, canonical_json(value))
    if not secrets.compare_digest(digest, _sha256(nonce + ciphertext[:-16])):
        raise RuntimeError("AES-GCM ciphertext changed while binding associated data")
    return ciphertext, value


def encrypt_task_artifact(
    tensors: Mapping[str, torch.Tensor],
    task: Mapping[str, Any],
    plan_hash: str,
    schema: ModelSchema,
    context: RoundContext,
    private: X25519PrivateKey,
    receiver: X25519PublicKey,
    identity: Ed25519PrivateKey,
) -> tuple[bytes, dict[str, Any], str]:
    clear = serialize_tensors(tensors)
    nonce = secrets.token_bytes(12)
    key = _derive_edge_key(private, receiver, context, str(task["task_id"]), plan_hash)
    ciphertext, value = _encrypt_with_digest(
        key,
        clear,
        nonce,
        lambda encoded_nonce, digest, size: artifact_value(task, plan_hash, schema.hash, encoded_nonce, digest, size),
    )
    signature = base64.b64encode(identity.sign(context.domain("tree-artifact") + canonical_json(value))).decode()
    return ciphertext, value, signature


def decrypt_task_artifact(
    ciphertext: bytes,
    value: Mapping[str, Any],
    signature: str,
    task: Mapping[str, Any],
    plan_hash: str,
    schema: ModelSchema,
    context: RoundContext,
    private: X25519PrivateKey,
    sender: X25519PublicKey,
    identity: Ed25519PublicKey,
) -> dict[str, torch.Tensor]:
    expected = artifact_value(
        task,
        plan_hash,
        schema.hash,
        str(value.get("nonce")),
        str(value.get("ciphertext_digest")),
        len(ciphertext),
    )
    if dict(value) != expected:
        raise ArtifactValidationError("tree artifact metadata is not authoritative")
    try:
        nonce = base64.b64decode(str(value["nonce"]), validate=True)
        if len(nonce) != 12 or not secrets.compare_digest(
            str(value["ciphertext_digest"]), _sha256(nonce + ciphertext[:-16])
        ):
            raise ValueError
        identity.verify(
            base64.b64decode(signature, validate=True),
            context.domain("tree-artifact") + canonical_json(expected),
        )
        key = _derive_edge_key(private, sender, context, str(task["task_id"]), plan_hash)
        clear = AESGCM(key).decrypt(nonce, ciphertext, canonical_json(expected))
    except Exception as exc:
        raise ArtifactValidationError("tree artifact authentication failed") from exc
    return deserialize_tensors(clear, schema)


def final_group_key(seed: bytes, context: RoundContext, plan_hash: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=context.domain(f"final-group-key:{plan_hash}"),
    ).derive(seed)


def encrypt_final_key(
    tensors: Mapping[str, torch.Tensor],
    seed: bytes,
    context: RoundContext,
    plan_hash: str,
    root: str,
    identity: Ed25519PrivateKey,
) -> tuple[bytes, dict[str, Any], str]:
    nonce = secrets.token_bytes(12)
    ciphertext, value = _encrypt_with_digest(
        final_group_key(seed, context, plan_hash),
        serialize_tensors(tensors),
        nonce,
        lambda encoded_nonce, digest, size: final_key_value(context, plan_hash, root, encoded_nonce, digest, size),
    )
    signature = base64.b64encode(identity.sign(context.domain("final-key") + canonical_json(value))).decode()
    return ciphertext, value, signature


def decrypt_final_key(
    ciphertext: bytes,
    value: Mapping[str, Any],
    signature: str,
    seed: bytes,
    context: RoundContext,
    plan_hash: str,
    root: str,
    identity: Ed25519PublicKey,
    schema: ModelSchema,
) -> dict[str, torch.Tensor]:
    expected = final_key_value(
        context,
        plan_hash,
        root,
        str(value.get("nonce")),
        str(value.get("ciphertext_digest")),
        len(ciphertext),
    )
    if dict(value) != expected:
        raise ArtifactValidationError("final key metadata is not authoritative")
    try:
        nonce = base64.b64decode(str(value["nonce"]), validate=True)
        if len(nonce) != 12 or not secrets.compare_digest(
            str(value["ciphertext_digest"]), _sha256(nonce + ciphertext[:-16])
        ):
            raise ValueError
        identity.verify(
            base64.b64decode(signature, validate=True),
            context.domain("final-key") + canonical_json(expected),
        )
        clear = AESGCM(final_group_key(seed, context, plan_hash)).decrypt(nonce, ciphertext, canonical_json(expected))
    except Exception as exc:
        raise ArtifactValidationError("final key authentication failed") from exc
    return deserialize_tensors(clear, schema)


def seed_tensors(seed: bytes, schema: ModelSchema, context: RoundContext, purpose: str) -> dict[str, torch.Tensor]:
    """Expand an independent seed into one int64-ring key per uploaded tensor."""
    result: dict[str, torch.Tensor] = {}
    for entry in schema.uploaded_entries:
        domain = context.domain(f"{purpose}:{entry.name}")
        tensor = torch.empty(entry.shape, dtype=torch.int64)
        view = tensor.view(torch.uint8).view(-1)
        offset = 0
        for chunk in pseudorandom_chunks(seed, domain, math.prod(entry.shape) * 8, purpose=purpose.encode()):
            source = torch.frombuffer(bytearray(chunk), dtype=torch.uint8)
            view[offset : offset + len(chunk)].copy_(source)
            offset += len(chunk)
        result[entry.name] = tensor
    return result
