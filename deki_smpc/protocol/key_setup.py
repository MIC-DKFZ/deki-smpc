"""Authenticated pairwise key setup and per-round mask generation."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .errors import ArtifactValidationError
from .integrity import FIELD_PRIME
from .models import RoundContext
from .prf import pseudorandom_chunks
from .schema import ModelSchema


def _hkdf(shared: bytes, domain: bytes, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=domain).derive(shared)


def _xor_ring_mask(key: bytes, domain: bytes, count: int) -> torch.Tensor:
    output = torch.empty(count, dtype=torch.int64)
    offset = 0
    for chunk in pseudorandom_chunks(
        key,
        domain,
        count * 8,
        purpose=b"model-mask",
    ):
        # Explicit little-endian decoding keeps masks identical on every host.
        values = np.frombuffer(chunk, dtype="<i8").astype(np.int64, copy=True)
        count_in_chunk = int(values.size)
        output[offset : offset + count_in_chunk].copy_(torch.from_numpy(values))
        offset += count_in_chunk
    return output


@dataclass
class RoundKeyMaterial:
    context: RoundContext
    _private_key: X25519PrivateKey | None = field(default_factory=X25519PrivateKey.generate, repr=False)
    seed_share: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    q_share: int = field(default_factory=lambda: secrets.randbelow(FIELD_PRIME))
    model_key_seed: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    group_blind_seed: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    peer_keys: dict[str, X25519PublicKey] = field(default_factory=dict)
    integrity_seed: bytes | None = None
    aggregate_q: int | None = None
    consumed: bool = False

    @property
    def private_key(self) -> X25519PrivateKey:
        if self._private_key is None or self.consumed:
            raise ArtifactValidationError("round key material was already consumed")
        return self._private_key

    @property
    def public_key(self) -> str:
        raw = self.private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return base64.b64encode(raw).decode("ascii")

    def set_peer_keys(self, encoded: Mapping[str, str]) -> None:
        peers: dict[str, X25519PublicKey] = {}
        for client_id, value in encoded.items():
            try:
                raw = base64.b64decode(value, validate=True)
                peers[client_id] = X25519PublicKey.from_public_bytes(raw)
            except Exception as exc:
                raise ArtifactValidationError("invalid X25519 public key") from exc
        if self.context.client_id not in peers:
            raise ArtifactValidationError("key manifest omits this client")
        self.peer_keys = peers

    def encrypted_shares(self) -> dict[str, dict[str, str]]:
        payload = json.dumps(
            {"q": str(self.q_share), "seed": self.seed_share.hex()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        result: dict[str, dict[str, str]] = {}
        for recipient, public in self.peer_keys.items():
            if recipient == self.context.client_id:
                continue
            shared = self.private_key.exchange(public)
            domain = self.context.domain(f"key-envelope:{self.context.client_id}:{recipient}")
            key = _hkdf(shared, domain)
            nonce = secrets.token_bytes(12)
            ciphertext = AESGCM(key).encrypt(nonce, payload, domain)
            result[recipient] = {
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
                "nonce": base64.b64encode(nonce).decode("ascii"),
            }
        return result

    def finalize(self, incoming: Mapping[str, Mapping[str, str]]) -> None:
        shares: dict[str, tuple[bytes, int]] = {self.context.client_id: (self.seed_share, self.q_share)}
        for sender, envelope in incoming.items():
            if sender == self.context.client_id or sender not in self.peer_keys:
                raise ArtifactValidationError("invalid key-share sender")
            domain = self.context.domain(f"key-envelope:{sender}:{self.context.client_id}")
            shared = self.private_key.exchange(self.peer_keys[sender])
            try:
                key = _hkdf(shared, domain)
                clear = AESGCM(key).decrypt(
                    base64.b64decode(envelope["nonce"], validate=True),
                    base64.b64decode(envelope["ciphertext"], validate=True),
                    domain,
                )
                payload = json.loads(clear)
                seed = bytes.fromhex(payload["seed"])
                q = int(payload["q"])
            except Exception as exc:
                raise ArtifactValidationError("key-share authentication failed") from exc
            if len(seed) != 32 or not 0 <= q < FIELD_PRIME:
                raise ArtifactValidationError("invalid key-share payload")
            shares[sender] = (seed, q)
        if set(shares) != set(self.peer_keys):
            raise ArtifactValidationError("not all key shares were delivered")
        ordered = [shares[name] for name in sorted(shares)]
        self.integrity_seed = hashlib.sha256(
            self.context.domain("integrity-seed") + b"".join(seed for seed, _ in ordered)
        ).digest()
        self.aggregate_q = sum(q for _, q in ordered) % FIELD_PRIME

    def mask(self, schema: ModelSchema) -> dict[str, torch.Tensor]:
        if self.consumed:
            raise ArtifactValidationError("round key material was already consumed")
        masks: dict[str, torch.Tensor] = {}
        for entry in schema.uploaded_entries:
            combined = torch.zeros(entry.shape, dtype=torch.int64)
            for peer_id, peer_key in self.peer_keys.items():
                if peer_id == self.context.client_id:
                    continue
                shared = self.private_key.exchange(peer_key)
                pair = ":".join(sorted((self.context.client_id, peer_id)))
                domain = self.context.domain(f"model-mask:{pair}:{entry.name}")
                key = _hkdf(shared, domain)
                values = _xor_ring_mask(key, domain, combined.numel()).view(entry.shape)
                combined = combined + values if self.context.client_id < peer_id else combined - values
            masks[entry.name] = combined
        return masks

    def consume(self) -> None:
        self.consumed = True
        self.seed_share = b""
        self.q_share = 0
        self.model_key_seed = b""
        self.group_blind_seed = b""
        self.integrity_seed = None
        self.aggregate_q = None
        self.peer_keys.clear()
        self._private_key = None
