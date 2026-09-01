"""One bounded secure-aggregation round."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .protocol.errors import ArtifactValidationError, ConfigurationError, RoundExpiredError, RoundFailedError
from .protocol.integrity import FIELD_PRIME, decode_tag, digest_tensors, encode_tag, verify_tag
from .protocol.key_setup import RoundKeyMaterial
from .protocol.models import MINIMUM_PARTICIPANTS, PROTOCOL_VERSION, RoundContext, RoundState
from .protocol.schema import ModelSchema, TensorPolicy, canonical_json
from .protocol.serialization import deserialize_tensors, serialize_tensors
from .transport import V1Transport, stable_idempotency_key
from .utils import FixedPointConverter


def _sign(key: Ed25519PrivateKey, context: RoundContext, purpose: str, value: object) -> str:
    return base64.b64encode(key.sign(context.domain(purpose) + canonical_json(value))).decode()


def _verify(key: Ed25519PublicKey, context: RoundContext, purpose: str, value: object, signature: str) -> None:
    try:
        key.verify(base64.b64decode(signature, validate=True), context.domain(purpose) + canonical_json(value))
    except Exception as exc:
        raise ArtifactValidationError("identity signature verification failed") from exc


@dataclass
class AggregationRoundSession:
    transport: V1Transport
    federation_id: str
    client_id: str
    round_id: str
    model: torch.nn.Module
    identity_key: Ed25519PrivateKey
    trusted_signing_keys: Mapping[str, Ed25519PublicKey]
    timeout_seconds: float
    tensor_policies: Mapping[str, TensorPolicy | str] | None = None
    cancellation: Callable[[], bool] | None = None
    progress: Callable[[dict[str, object]], None] | None = None

    def run(self) -> dict[str, torch.Tensor]:
        deadline = time.monotonic() + self.timeout_seconds
        local = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        status = self._status(deadline)
        schema = ModelSchema.from_state_dict(
            local,
            precision_bits=int(str(status["precision_bits"])),
            policies=self.tensor_policies,
        )
        if schema.hash != status["model_schema_hash"]:
            raise ArtifactValidationError("local model does not match the round schema")
        context = RoundContext(
            self.federation_id,
            self.round_id,
            self.client_id,
            str(status["protocol_version"]),
            schema.hash,
            str(status["participant_manifest_hash"]),
            str(status["aggregation_policy"]),
            schema.precision_bits,
        )
        pinned_manifest = []
        for client_id in sorted(self.trusted_signing_keys):
            raw = self.trusted_signing_keys[client_id].public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
            pinned_manifest.append({"client_id": client_id, "signing_public_key": base64.b64encode(raw).decode()})
        if hashlib.sha256(canonical_json(pinned_manifest)).hexdigest() != context.participant_manifest_hash:
            raise ArtifactValidationError("server participant manifest differs from pinned identities")
        # The count drives both the fixed-point aggregation bound and the MEAN
        # divisor, so it is taken from the pinned manifest rather than from the
        # server response, which carries no signature over it.
        participants = len(pinned_manifest)
        if participants < MINIMUM_PARTICIPANTS:
            raise ConfigurationError(f"a round needs at least {MINIMUM_PARTICIPANTS} pinned participants")
        if int(str(status["expected_participant_count"])) != participants:
            raise ArtifactValidationError("round participant count differs from pinned identities")
        if context.protocol_version != PROTOCOL_VERSION:
            raise ConfigurationError("unsupported protocol version")
        keys = RoundKeyMaterial(context)
        self._join(keys, deadline)
        self._key_setup(keys, deadline)
        encoded = self._encode(local, schema, participants)
        if keys.integrity_seed is None or keys.aggregate_q is None:
            raise ArtifactValidationError("key setup did not produce verification material")
        client_tag = (digest_tensors(encoded, schema, keys.integrity_seed, context) + keys.q_share) % FIELD_PRIME
        masks = keys.mask(schema)
        masked = {name: tensor + masks[name] for name, tensor in encoded.items()}
        self._upload_update(masked, client_tag, schema, deadline)
        result_data, result_tag = self._await_result(deadline)
        aggregate = deserialize_tensors(result_data, schema)
        expected = digest_tensors(aggregate, schema, keys.integrity_seed, context)
        actual = (decode_tag(result_tag) - keys.aggregate_q) % FIELD_PRIME
        verify_tag(actual, expected)
        result = self._decode(aggregate, local, schema, participants)
        self._complete(deadline)
        keys.consume()
        self._emit("completed", state=RoundState.COMPLETED.value)
        return result

    def _status(self, deadline: float) -> dict[str, object]:
        response = self.transport.request(
            "GET", f"/v1/rounds/{self.round_id}", deadline=deadline, cancellation=self.cancellation
        )
        status = cast(dict[str, object], response.json())
        if status.get("federation_id") != self.federation_id:
            raise ArtifactValidationError("round belongs to another federation")
        if status.get("state") == RoundState.EXPIRED:
            raise RoundExpiredError("round expired", round_id=self.round_id)
        if status.get("state") in {RoundState.FAILED, RoundState.ABORTED}:
            raise RoundFailedError(str(status.get("failure_code") or status["state"]), round_id=self.round_id)
        return status

    def _wait_for(self, states: set[str], deadline: float) -> dict[str, object]:
        delay = 0.05
        while True:
            status = self._status(deadline)
            if status["state"] in states:
                return status
            self.transport._wait(delay, deadline, self.cancellation)
            delay = min(delay * 1.6, 1)

    def _join(self, keys: RoundKeyMaterial, deadline: float) -> None:
        value = {"public_key": keys.public_key}
        body = {**value, "signature": _sign(self.identity_key, keys.context, "ephemeral-public-key", value)}
        encoded = canonical_json(body)
        self.transport.request(
            "PUT",
            f"/v1/rounds/{self.round_id}/artifacts/public_key",
            content=encoded,
            headers={"Content-Type": "application/json"},
            deadline=deadline,
            cancellation=self.cancellation,
            idempotency_key=stable_idempotency_key("public_key", self.round_id, encoded),
        )
        self._emit("joined")

    def _key_setup(self, keys: RoundKeyMaterial, deadline: float) -> None:
        self._wait_for({RoundState.KEY_SETUP.value, RoundState.UPDATE_COLLECTION.value}, deadline)
        response = self.transport.request(
            "GET",
            f"/v1/rounds/{self.round_id}/artifacts/public_keys",
            deadline=deadline,
            cancellation=self.cancellation,
        )
        records = response.json()["public_keys"]
        if set(records) != set(self.trusted_signing_keys):
            raise ArtifactValidationError("participant identity manifest is not pinned locally")
        encoded_keys: dict[str, str] = {}
        for client_id, record in records.items():
            value = {"public_key": record["public_key"]}
            _verify(
                self.trusted_signing_keys[client_id], keys.context, "ephemeral-public-key", value, record["signature"]
            )
            encoded_keys[client_id] = record["public_key"]
        keys.set_peer_keys(encoded_keys)
        value = {"messages": keys.encrypted_shares()}
        body = {**value, "signature": _sign(self.identity_key, keys.context, "key-bundle", value)}
        encoded = canonical_json(body)
        self.transport.request(
            "PUT",
            f"/v1/rounds/{self.round_id}/artifacts/key_bundle",
            content=encoded,
            headers={"Content-Type": "application/json"},
            deadline=deadline,
            cancellation=self.cancellation,
            idempotency_key=stable_idempotency_key("key_bundle", self.round_id, encoded),
        )
        delay = 0.05
        while True:
            response = self.transport.request(
                "GET",
                f"/v1/rounds/{self.round_id}/artifacts/key_bundle",
                deadline=deadline,
                cancellation=self.cancellation,
            )
            if response.status_code != 204:
                records = response.json()["bundles"]
                break
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = max(delay, min(float(retry_after), 2.0))
                except ValueError:
                    pass
            self.transport._wait(delay, deadline, self.cancellation)
            delay = min(delay * 1.6, 1.0)
        incoming: dict[str, Mapping[str, str]] = {}
        for sender, record in records.items():
            value = {"messages": record["messages"]}
            _verify(self.trusted_signing_keys[sender], keys.context, "key-bundle", value, record["signature"])
            incoming[sender] = record["messages"][self.client_id]
        keys.finalize(incoming)
        integrity_seed = keys.integrity_seed
        if integrity_seed is None:
            raise ArtifactValidationError("key setup did not produce verification material")
        commitment = hashlib.sha256(integrity_seed + keys.context.domain("context-commitment")).hexdigest()
        body_bytes = canonical_json({"context_commitment": commitment})
        self.transport.request(
            "POST",
            f"/v1/rounds/{self.round_id}/key-setup/complete",
            content=body_bytes,
            headers={"Content-Type": "application/json"},
            deadline=deadline,
            cancellation=self.cancellation,
            idempotency_key=stable_idempotency_key("key_complete", self.round_id, body_bytes),
        )
        self._wait_for({RoundState.UPDATE_COLLECTION.value}, deadline)
        self._emit("key_setup_complete")

    @staticmethod
    def _encode(state: Mapping[str, torch.Tensor], schema: ModelSchema, participants: int) -> dict[str, torch.Tensor]:
        converter = FixedPointConverter(schema.precision_bits, max_aggregation_terms=participants)
        return {
            entry.name: converter.encode(state[entry.name], tensor_name=entry.name).cpu().contiguous()
            for entry in schema.uploaded_entries
        }

    def _upload_update(
        self, masked: Mapping[str, torch.Tensor], tag: int, schema: ModelSchema, deadline: float
    ) -> None:
        content = serialize_tensors(masked)
        tag_text = encode_tag(tag)
        self.transport.request(
            "PUT",
            f"/v1/rounds/{self.round_id}/artifacts/update",
            content=content,
            headers={
                "Content-Type": "application/vnd.safetensors",
                "X-Integrity-Tag": tag_text,
                "X-Model-Schema-Hash": schema.hash,
            },
            deadline=deadline,
            cancellation=self.cancellation,
            idempotency_key=stable_idempotency_key("update", self.round_id, content + tag_text.encode()),
        )
        self._emit("update_uploaded", bytes=len(content))

    def _await_result(self, deadline: float) -> tuple[bytes, str]:
        self._wait_for({RoundState.RESULT_READY.value, RoundState.COMPLETED.value}, deadline)
        response = self.transport.request(
            "GET", f"/v1/rounds/{self.round_id}/artifacts/result", deadline=deadline, cancellation=self.cancellation
        )
        content_digest = response.headers.get("X-Content-SHA256")
        integrity_tag = response.headers.get("X-Integrity-Tag")
        if (
            content_digest is None
            or not hmac.compare_digest(hashlib.sha256(response.content).hexdigest(), content_digest)
            or integrity_tag is None
        ):
            raise ArtifactValidationError("aggregate artifact metadata is invalid")
        return response.content, integrity_tag

    @staticmethod
    def _decode(
        aggregate: Mapping[str, torch.Tensor], local: Mapping[str, torch.Tensor], schema: ModelSchema, participants: int
    ) -> dict[str, torch.Tensor]:
        converter = FixedPointConverter(schema.precision_bits, max_aggregation_terms=participants)
        result = {name: value.detach().clone() for name, value in local.items()}
        for entry in schema.entries:
            if entry.policy == TensorPolicy.KEEP_LOCAL:
                continue
            decoded = converter.decode(aggregate[entry.name])
            if entry.policy == TensorPolicy.MEAN:
                decoded = decoded / participants
            result[entry.name] = decoded.to(dtype=local[entry.name].dtype, device=local[entry.name].device)
        return result

    def _complete(self, deadline: float) -> None:
        self.transport.request(
            "POST",
            f"/v1/rounds/{self.round_id}/complete",
            content=b"{}",
            headers={"Content-Type": "application/json"},
            deadline=deadline,
            cancellation=self.cancellation,
            idempotency_key=stable_idempotency_key("complete", self.round_id, b"{}"),
        )

    def _emit(self, event: str, **fields: object) -> None:
        if self.progress:
            self.progress({"event": event, "round_id": self.round_id, **fields})


def load_identity_key(encoded: str) -> Ed25519PrivateKey:
    try:
        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(encoded, validate=True))
    except Exception as exc:
        raise ConfigurationError("invalid Ed25519 identity private key") from exc


def load_trusted_keys(encoded: Mapping[str, str]) -> dict[str, Ed25519PublicKey]:
    try:
        return {
            name: Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))
            for name, value in encoded.items()
        }
    except Exception as exc:
        raise ConfigurationError("invalid trusted Ed25519 public key") from exc
