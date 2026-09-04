"""One bounded secure-aggregation round for protocol 1.0 or 1.1."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .protocol.errors import ArtifactValidationError, ConfigurationError, RoundExpiredError, RoundFailedError
from .protocol.integrity import FIELD_PRIME, decode_tag, digest_tensors, encode_tag, verify_tag
from .protocol.key_setup import RoundKeyMaterial
from .protocol.models import (
    LEGACY_PROTOCOL_VERSION,
    MINIMUM_PARTICIPANTS,
    SUPPORTED_PROTOCOL_VERSIONS,
    RoundContext,
    RoundState,
)
from .protocol.schema import ModelSchema, TensorPolicy, canonical_json
from .protocol.serialization import deserialize_tensors, serialize_tensors
from .protocol.tree import (
    decrypt_final_key,
    decrypt_task_artifact,
    derive_tree_plan,
    encrypt_final_key,
    encrypt_task_artifact,
    seed_tensors,
    signed_ephemeral_manifest,
    tree_plan_hash,
)
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
class PreparedRoundState:
    context: RoundContext
    schema: ModelSchema
    participants: int
    keys: RoundKeyMaterial
    deadline: float
    plan: dict[str, Any] | None = None
    plan_hash: str | None = None
    aggregate_key: dict[str, torch.Tensor] | None = None
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.keys.consume()
        if self.aggregate_key is not None:
            for tensor in self.aggregate_key.values():
                tensor.zero_()
            self.aggregate_key.clear()
            self.aggregate_key = None


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
        prepared = self.prepare()
        try:
            return self.run_prepared(prepared)
        except Exception:
            prepared.close()
            raise

    def prepare(self) -> PreparedRoundState:
        deadline = time.monotonic() + self.timeout_seconds
        keys: RoundKeyMaterial | None = None
        try:
            local = {name: value.detach() for name, value in self.model.state_dict().items()}
            status = self._status(deadline)
            schema = ModelSchema.from_state_dict(
                local, precision_bits=int(str(status["precision_bits"])), policies=self.tensor_policies
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
            participants = self._validate_manifest(context, status)
            if context.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
                raise ConfigurationError(f"unsupported protocol version {context.protocol_version!r}")
            keys = RoundKeyMaterial(context)
            self._join(keys, deadline)
            records = self._key_setup(keys, deadline)
            prepared = PreparedRoundState(context, schema, participants, keys, deadline)
            if context.protocol_version != LEGACY_PROTOCOL_VERSION:
                self._tree_key_setup(prepared, records)
            self._emit("prepared", protocol_version=context.protocol_version)
            return prepared
        except Exception:
            if keys is not None:
                keys.consume()
            raise

    def run_prepared(self, prepared: PreparedRoundState) -> dict[str, torch.Tensor]:
        if prepared.closed:
            raise ArtifactValidationError("prepared round is no longer usable")
        local = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        current_schema = ModelSchema.from_state_dict(
            local, precision_bits=prepared.schema.precision_bits, policies=self.tensor_policies
        )
        if current_schema != prepared.schema:
            raise ArtifactValidationError("prepared round schema or tensor policies do not match the model")
        encoded = self._encode(local, prepared.schema, prepared.participants)
        keys = prepared.keys
        if keys.integrity_seed is None or keys.aggregate_q is None:
            raise ArtifactValidationError("key setup did not produce verification material")
        client_tag = (
            digest_tensors(encoded, prepared.schema, keys.integrity_seed, prepared.context) + keys.q_share
        ) % FIELD_PRIME
        if prepared.context.protocol_version == LEGACY_PROTOCOL_VERSION:
            mask = keys.mask(prepared.schema)
        else:
            mask = seed_tensors(
                keys.model_key_seed, prepared.schema, prepared.context, f"private-model-key:{prepared.plan_hash}"
            )
        protected = {name: tensor + mask[name] for name, tensor in encoded.items()}
        for tensor in mask.values():
            tensor.zero_()
        try:
            self._upload_update(protected, client_tag, prepared.schema, prepared.deadline)
        finally:
            for tensor in protected.values():
                tensor.zero_()
        result_data, result_tag = self._await_result(prepared.deadline)
        aggregate = deserialize_tensors(result_data, prepared.schema)
        if prepared.context.protocol_version != LEGACY_PROTOCOL_VERSION:
            if prepared.aggregate_key is None:
                raise ArtifactValidationError("protocol 1.1 did not distribute an aggregate key")
            for name in aggregate:
                aggregate[name].sub_(prepared.aggregate_key[name])
        expected = digest_tensors(aggregate, prepared.schema, keys.integrity_seed, prepared.context)
        actual = (decode_tag(result_tag) - keys.aggregate_q) % FIELD_PRIME
        verify_tag(actual, expected)
        result = self._decode(aggregate, local, prepared.schema, prepared.participants)
        self._complete(prepared.deadline)
        prepared.close()
        self._emit("completed", state=RoundState.COMPLETED.value)
        return result

    def _validate_manifest(self, context: RoundContext, status: Mapping[str, object]) -> int:
        pinned = []
        for client_id in sorted(self.trusted_signing_keys):
            raw = self.trusted_signing_keys[client_id].public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
            pinned.append({"client_id": client_id, "signing_public_key": base64.b64encode(raw).decode()})
        if hashlib.sha256(canonical_json(pinned)).hexdigest() != context.participant_manifest_hash:
            raise ArtifactValidationError("server participant manifest differs from pinned identities")
        participants = len(pinned)
        if participants < MINIMUM_PARTICIPANTS:
            raise ConfigurationError(f"a round needs at least {MINIMUM_PARTICIPANTS} pinned participants")
        if int(str(status["expected_participant_count"])) != participants:
            raise ArtifactValidationError("round participant count differs from pinned identities")
        return participants

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

    def _key_setup(self, keys: RoundKeyMaterial, deadline: float) -> dict[str, Mapping[str, str]]:
        self._wait_for(
            {RoundState.KEY_SETUP.value, RoundState.KEY_AGGREGATION.value, RoundState.UPDATE_COLLECTION.value}, deadline
        )
        response = self.transport.request(
            "GET",
            f"/v1/rounds/{self.round_id}/artifacts/public_keys",
            deadline=deadline,
            cancellation=self.cancellation,
        )
        records = cast(dict[str, Mapping[str, str]], response.json()["public_keys"])
        if set(records) != set(self.trusted_signing_keys):
            raise ArtifactValidationError("participant identity manifest is not pinned locally")
        encoded_keys = {}
        for client_id, record in records.items():
            public_key_value = {"public_key": record["public_key"]}
            _verify(
                self.trusted_signing_keys[client_id],
                keys.context,
                "ephemeral-public-key",
                public_key_value,
                record["signature"],
            )
            encoded_keys[client_id] = record["public_key"]
        keys.set_peer_keys(encoded_keys)
        bundle_value = {"messages": keys.encrypted_shares()}
        body = {
            **bundle_value,
            "signature": _sign(self.identity_key, keys.context, "key-bundle", bundle_value),
        }
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
                incoming_records = response.json()["bundles"]
                break
            self.transport._wait(delay, deadline, self.cancellation)
            delay = min(delay * 1.6, 1.0)
        incoming = {}
        for sender, record in incoming_records.items():
            value = {"messages": record["messages"]}
            _verify(self.trusted_signing_keys[sender], keys.context, "key-bundle", value, record["signature"])
            incoming[sender] = record["messages"][self.client_id]
        keys.finalize(incoming)
        if keys.integrity_seed is None:
            raise ArtifactValidationError("key setup did not produce verification material")
        commitment = hashlib.sha256(keys.integrity_seed + keys.context.domain("context-commitment")).hexdigest()
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
        target = (
            {RoundState.UPDATE_COLLECTION.value}
            if keys.context.protocol_version == LEGACY_PROTOCOL_VERSION
            else {RoundState.KEY_AGGREGATION.value}
        )
        self._wait_for(target, deadline)
        self._emit("key_setup_complete")
        return records

    def _tree_key_setup(self, prepared: PreparedRoundState, records: Mapping[str, Mapping[str, str]]) -> None:
        response = self.transport.request(
            "GET", f"/v1/rounds/{self.round_id}/tree-plan", deadline=prepared.deadline, cancellation=self.cancellation
        )
        server = cast(dict[str, Any], response.json())
        manifest = signed_ephemeral_manifest(records)
        plan = derive_tree_plan(prepared.context, manifest)
        plan_hash = tree_plan_hash(plan)
        if (
            server.get("ephemeral_manifest") != manifest
            or server.get("plan") != plan
            or not hmac.compare_digest(str(server.get("plan_hash")), plan_hash)
        ):
            raise ArtifactValidationError("server tree plan does not match the independently derived topology")
        prepared.plan, prepared.plan_hash = plan, plan_hash
        tasks = {task["task_id"]: task for task in plan["tasks"]}
        delay = 0.05
        while True:
            response = self.transport.request(
                "GET",
                f"/v1/rounds/{self.round_id}/key-actions/next",
                deadline=prepared.deadline,
                cancellation=self.cancellation,
            )
            if response.status_code == 204:
                self.transport._wait(delay, prepared.deadline, self.cancellation)
                delay = min(delay * 1.6, 1.0)
                continue
            delay = 0.05
            action = cast(dict[str, Any], response.json())
            if action["action"] == "KEY_AGGREGATION_COMPLETE":
                if prepared.aggregate_key is None:
                    raise ArtifactValidationError("final aggregate key was not received")
                return
            if action["action"] == "PUBLISH_FINAL":
                aggregate_key = self._download_task(prepared, tasks[plan["root_task_id"]])
                try:
                    if prepared.keys.integrity_seed is None:
                        raise ArtifactValidationError("shared integrity seed is unavailable")
                    ciphertext, value, signature = encrypt_final_key(
                        aggregate_key,
                        prepared.keys.integrity_seed,
                        prepared.context,
                        plan_hash,
                        str(plan["root_client_id"]),
                        self.identity_key,
                    )
                    self._upload_encrypted("final-key", "/final-key", ciphertext, value, signature, prepared.deadline)
                finally:
                    for tensor in aggregate_key.values():
                        tensor.zero_()
                continue
            if action["action"] == "ACK_FINAL":
                prepared.aggregate_key, final_digest = self._download_final_key(prepared)
                body = canonical_json({"artifact_digest": final_digest})
                self.transport.request(
                    "POST",
                    f"/v1/rounds/{self.round_id}/final-key/ack",
                    content=body,
                    headers={"Content-Type": "application/json"},
                    deadline=prepared.deadline,
                    cancellation=self.cancellation,
                    idempotency_key=stable_idempotency_key("final_key_ack", self.round_id, body),
                )
                continue
            task_id = str(action["task_id"])
            task = tasks.get(task_id)
            if task is None or any(action.get(field) != task[field] for field in task):
                raise ArtifactValidationError("server returned non-authoritative tree task metadata")
            output = self._perform_task(prepared, task, tasks)
            try:
                ciphertext, value, signature = encrypt_task_artifact(
                    output,
                    task,
                    plan_hash,
                    prepared.schema,
                    prepared.context,
                    prepared.keys.private_key,
                    prepared.keys.peer_keys[str(task["receiver"])],
                    self.identity_key,
                )
                self._upload_encrypted(
                    f"tree_task:{task_id}",
                    f"/key-tasks/{task_id}/artifact",
                    ciphertext,
                    value,
                    signature,
                    prepared.deadline,
                )
            finally:
                for tensor in output.values():
                    tensor.zero_()
            body = canonical_json({"artifact_digest": value["ciphertext_digest"]})
            self.transport.request(
                "POST",
                f"/v1/rounds/{self.round_id}/key-tasks/{task_id}/ack",
                content=body,
                headers={"Content-Type": "application/json"},
                deadline=prepared.deadline,
                cancellation=self.cancellation,
                idempotency_key=stable_idempotency_key(f"tree_task_ack:{task_id}", self.round_id, body),
            )

    def _perform_task(
        self, prepared: PreparedRoundState, task: Mapping[str, Any], tasks: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, torch.Tensor]:
        purpose = f"private-model-key:{prepared.plan_hash}"
        if task["action"] == "GROUP_START":
            output = seed_tensors(prepared.keys.model_key_seed, prepared.schema, prepared.context, purpose)
            group_id = str(task["task_id"]).split("-")[0]
            blind = seed_tensors(
                prepared.keys.group_blind_seed,
                prepared.schema,
                prepared.context,
                f"group-blind:{prepared.plan_hash}:{group_id}",
            )
            for name in output:
                output[name].add_(blind[name])
                blind[name].zero_()
            return output
        dependencies = [tasks[dependency] for dependency in task["dependencies"]]
        output = self._download_task(prepared, dependencies[0])
        if task["action"] == "GROUP_ADD":
            private_key = seed_tensors(prepared.keys.model_key_seed, prepared.schema, prepared.context, purpose)
            for name in output:
                output[name].add_(private_key[name])
                private_key[name].zero_()
        elif task["action"] == "GROUP_UNBLIND":
            group_id = str(task["task_id"]).split("-")[0]
            blind = seed_tensors(
                prepared.keys.group_blind_seed,
                prepared.schema,
                prepared.context,
                f"group-blind:{prepared.plan_hash}:{group_id}",
            )
            for name in output:
                output[name].sub_(blind[name])
                blind[name].zero_()
        elif task["action"] == "TREE_COMBINE":
            second = self._download_task(prepared, dependencies[1])
            for name in output:
                output[name].add_(second[name])
                second[name].zero_()
        elif task["action"] != "TREE_CARRY":
            raise ArtifactValidationError("unknown tree task action")
        return output

    def _download_task(self, prepared: PreparedRoundState, task: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        response = self.transport.request(
            "GET",
            f"/v1/rounds/{self.round_id}/key-tasks/{task['task_id']}/artifact",
            deadline=prepared.deadline,
            cancellation=self.cancellation,
        )
        metadata, signature = self._encrypted_headers(response)
        return decrypt_task_artifact(
            response.content,
            metadata,
            signature,
            task,
            cast(str, prepared.plan_hash),
            prepared.schema,
            prepared.context,
            prepared.keys.private_key,
            prepared.keys.peer_keys[str(task["sender"])],
            self.trusted_signing_keys[str(task["sender"])],
        )

    def _download_final_key(self, prepared: PreparedRoundState) -> tuple[dict[str, torch.Tensor], str]:
        response = self.transport.request(
            "GET", f"/v1/rounds/{self.round_id}/final-key", deadline=prepared.deadline, cancellation=self.cancellation
        )
        metadata, signature = self._encrypted_headers(response)
        if prepared.keys.integrity_seed is None or prepared.plan is None or prepared.plan_hash is None:
            raise ArtifactValidationError("final key context is incomplete")
        root = str(prepared.plan["root_client_id"])
        tensors = decrypt_final_key(
            response.content,
            metadata,
            signature,
            prepared.keys.integrity_seed,
            prepared.context,
            prepared.plan_hash,
            root,
            self.trusted_signing_keys[root],
            prepared.schema,
        )
        return tensors, str(metadata["ciphertext_digest"])

    def _encrypted_headers(self, response: Any) -> tuple[dict[str, Any], str]:
        encoded = response.headers.get("X-Artifact-Metadata")
        signature = response.headers.get("X-Artifact-Signature")
        content_digest = response.headers.get("X-Content-SHA256")
        try:
            if (
                encoded is None
                or signature is None
                or content_digest is None
                or not hmac.compare_digest(hashlib.sha256(response.content).hexdigest(), content_digest)
            ):
                raise ValueError
            metadata = json.loads(base64.b64decode(encoded, validate=True))
            if not isinstance(metadata, dict):
                raise TypeError
        except Exception as exc:
            raise ArtifactValidationError("encrypted artifact response metadata is invalid") from exc
        return cast(dict[str, Any], metadata), signature

    def _upload_encrypted(
        self, operation: str, suffix: str, ciphertext: bytes, value: Mapping[str, Any], signature: str, deadline: float
    ) -> None:
        metadata = canonical_json(value)
        self.transport.request(
            "PUT",
            f"/v1/rounds/{self.round_id}{suffix}",
            content=ciphertext,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Artifact-Metadata": base64.b64encode(metadata).decode(),
                "X-Artifact-Signature": signature,
            },
            deadline=deadline,
            cancellation=self.cancellation,
            idempotency_key=stable_idempotency_key(
                operation, self.round_id, hashlib.sha256(ciphertext).digest() + metadata
            ),
        )

    @staticmethod
    def _encode(state: Mapping[str, torch.Tensor], schema: ModelSchema, participants: int) -> dict[str, torch.Tensor]:
        converter = FixedPointConverter(schema.precision_bits, max_aggregation_terms=participants)
        return {
            entry.name: converter.encode(state[entry.name], tensor_name=entry.name).cpu().contiguous()
            for entry in schema.uploaded_entries
        }

    def _upload_update(
        self, protected: Mapping[str, torch.Tensor], tag: int, schema: ModelSchema, deadline: float
    ) -> None:
        content = serialize_tensors(protected)
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
            idempotency_key=stable_idempotency_key(
                "update", self.round_id, hashlib.sha256(content).digest() + tag_text.encode()
            ),
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
