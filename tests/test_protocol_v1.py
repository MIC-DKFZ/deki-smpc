import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from deki_smpc.protocol.errors import AggregateIntegrityError, ArtifactValidationError, ConfigurationError
from deki_smpc.protocol.integrity import FIELD_PRIME, digest_tensors, verify_tag
from deki_smpc.protocol.key_setup import RoundKeyMaterial
from deki_smpc.protocol.models import RoundContext
from deki_smpc.protocol.schema import ModelSchema, canonical_json
from deki_smpc.protocol.serialization import deserialize_tensors, serialize_tensors
from deki_smpc.round_session import AggregationRoundSession
from deki_smpc.transport import V1Transport


def _context(client_id: str, schema: ModelSchema, round_id: str = "round-1") -> RoundContext:
    return RoundContext(
        "federation", round_id, client_id, "1.0", schema.hash, "manifest", "EQUAL_WEIGHTED", schema.precision_bits
    )


def _materials(count: int = 3) -> tuple[ModelSchema, list[RoundKeyMaterial]]:
    schema = ModelSchema.from_state_dict({"weight": torch.ones(13)})
    materials = [RoundKeyMaterial(_context(f"client-{index}", schema)) for index in range(count)]
    public = {item.context.client_id: item.public_key for item in materials}
    for item in materials:
        item.set_peer_keys(public)
    bundles = {item.context.client_id: item.encrypted_shares() for item in materials}
    for item in materials:
        incoming = {
            sender: messages[item.context.client_id]
            for sender, messages in bundles.items()
            if sender != item.context.client_id
        }
        item.finalize(incoming)
    return schema, materials


def test_all_clients_derive_the_same_secret_and_pairwise_masks_cancel() -> None:
    schema, materials = _materials()
    assert len({item.integrity_seed for item in materials}) == 1
    assert len({item.aggregate_q for item in materials}) == 1
    masks = [item.mask(schema)["weight"] for item in materials]
    assert torch.equal(sum(masks[1:], masks[0]), torch.zeros(13, dtype=torch.int64))


def test_key_material_is_fresh_and_round_bound() -> None:
    schema, first = _materials()
    _, second = _materials()
    assert first[0].public_key != second[0].public_key
    first[0].consume()
    with pytest.raises(ArtifactValidationError, match="consumed"):
        first[0].mask(schema)


def test_replayed_encrypted_share_fails_in_another_round() -> None:
    schema, materials = _materials()
    replay = materials[0].encrypted_shares()[materials[1].context.client_id]
    victim = RoundKeyMaterial(_context(materials[1].context.client_id, schema, "round-2"))
    victim.set_peer_keys(
        {materials[0].context.client_id: materials[0].public_key, victim.context.client_id: victim.public_key}
    )
    with pytest.raises(ArtifactValidationError, match="authentication failed"):
        victim.finalize({materials[0].context.client_id: replay})


def test_prime_field_integrity_detects_weight_and_tag_tampering() -> None:
    schema = ModelSchema.from_state_dict({"weight": torch.ones(4)})
    context = _context("client-1", schema)
    seed = bytes(range(32))
    encoded = {"weight": torch.tensor([1, -2, 3, 4], dtype=torch.int64)}
    digest = digest_tensors(encoded, schema, seed, context)
    verify_tag(digest, digest)
    tampered = {"weight": encoded["weight"].clone()}
    tampered["weight"][2] ^= 1
    with pytest.raises(AggregateIntegrityError):
        verify_tag(digest, digest_tensors(tampered, schema, seed, context))
    with pytest.raises(AggregateIntegrityError):
        verify_tag((digest + 1) % FIELD_PRIME, digest)


def test_safe_serialization_rejects_pickle_and_schema_mismatch() -> None:
    schema = ModelSchema.from_state_dict({"weight": torch.ones(2)})
    with pytest.raises(ArtifactValidationError, match="invalid safetensors"):
        deserialize_tensors(b"cos\nS'system'\n(S'echo unsafe'\ntR.", schema)
    payload = serialize_tensors({"other": torch.ones(2, dtype=torch.int64)})
    with pytest.raises(ArtifactValidationError, match="names"):
        deserialize_tensors(payload, schema)


def test_schema_hash_is_canonical_and_integer_buffers_stay_local() -> None:
    first = ModelSchema.from_state_dict({"z": torch.ones(2), "counter": torch.tensor(4), "a": torch.zeros(1)})
    second = ModelSchema.from_state_dict({"a": torch.zeros(1), "counter": torch.tensor(4), "z": torch.ones(2)})
    assert first.hash == second.hash
    assert [entry.name for entry in first.entries] == ["a", "counter", "z"]
    assert next(entry for entry in first.entries if entry.name == "counter").policy == "KEEP_LOCAL"


def test_golden_protocol_fixture() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "protocol-v1.json").read_text())
    schema = ModelSchema.from_state_dict({"weight": torch.zeros(2)}, precision_bits=8)
    assert schema.as_dict() == fixture["schema"]
    assert schema.hash == fixture["model_schema_hash"]
    context = RoundContext(
        fixture["federation_id"],
        fixture["round_id"],
        "client-1",
        "1.0",
        schema.hash,
        fixture["participant_manifest_hash"],
        "EQUAL_WEIGHTED",
        8,
    )
    seed = bytes.fromhex(fixture["integrity_seed_hex"])
    aggregate_tag = 0
    for client in fixture["clients"]:
        encoded = {"weight": torch.tensor(client["encoded"], dtype=torch.int64)}
        tag = (digest_tensors(encoded, schema, seed, context) + client["q"]) % FIELD_PRIME
        assert tag.to_bytes(16, "big").hex() == client["tag"]
        aggregate_tag = (aggregate_tag + tag) % FIELD_PRIME
    assert aggregate_tag.to_bytes(16, "big").hex() == fixture["aggregate_tag"]


def _pinned_manifest(keys: dict[str, Ed25519PrivateKey]) -> tuple[dict[str, Ed25519PublicKey], str]:
    public = {name: key.public_key() for name, key in keys.items()}
    manifest = [
        {
            "client_id": name,
            "signing_public_key": base64.b64encode(
                public[name].public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            ).decode(),
        }
        for name in sorted(public)
    ]
    return public, hashlib.sha256(canonical_json(manifest)).hexdigest()


def _session(count: int, reported: int) -> AggregationRoundSession:
    """Build a session whose server reports ``reported`` participants."""
    model = torch.nn.Linear(2, 2)
    keys = {f"client-{index}": Ed25519PrivateKey.generate() for index in range(count)}
    public, manifest_hash = _pinned_manifest(keys)
    schema = ModelSchema.from_state_dict(model.state_dict(), precision_bits=24)
    status = {
        "federation_id": "federation",
        "state": "REGISTRATION_OPEN",
        "protocol_version": "1.0",
        "model_schema_hash": schema.hash,
        "participant_manifest_hash": manifest_hash,
        "aggregation_policy": "EQUAL_WEIGHTED",
        "precision_bits": 24,
        "expected_participant_count": reported,
    }

    class _Transport:
        def request(self, *_: object, **__: object) -> object:
            return SimpleNamespace(json=lambda: status)

    return AggregationRoundSession(
        cast(V1Transport, _Transport()),
        "federation",
        "client-0",
        "round-1",
        model,
        keys["client-0"],
        public,
        timeout_seconds=5.0,
    )


def test_forged_participant_count_is_rejected_against_the_pinned_manifest() -> None:
    # A server that inflates the count would silently scale every client's
    # aggregate, and the prime-field tag cannot detect it.
    with pytest.raises(ArtifactValidationError, match="participant count"):
        _session(3, 6).run()


def test_round_below_the_masking_minimum_is_refused() -> None:
    # Two participants can each subtract their own update from the aggregate.
    with pytest.raises(ConfigurationError, match="at least 3"):
        _session(2, 2).run()
