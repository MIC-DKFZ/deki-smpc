"""Domain-separated, bounded-memory pseudorandom byte streams."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

STREAM_CHUNK_BYTES = 1024 * 1024


def pseudorandom_chunks(
    key: bytes,
    domain: bytes,
    length: int,
    *,
    purpose: bytes,
    chunk_bytes: int = STREAM_CHUNK_BYTES,
) -> Iterator[bytes]:
    """Generate a deterministic AES-256-CTR stream without a full-size buffer."""
    if not key or length < 0 or chunk_bytes <= 0:
        raise ValueError("invalid pseudorandom stream parameters")
    stream_key = hmac.new(key, b"deki-prf-key-v1\x00" + purpose + b"\x00" + domain, hashlib.sha256).digest()
    counter = hmac.new(
        key,
        b"deki-prf-counter-v1\x00" + purpose + b"\x00" + domain,
        hashlib.sha256,
    ).digest()[:16]
    encryptor = Cipher(algorithms.AES(stream_key), modes.CTR(counter)).encryptor()
    remaining = length
    while remaining:
        size = min(remaining, chunk_bytes)
        yield encryptor.update(bytes(size))
        remaining -= size
    tail = encryptor.finalize()
    if tail:
        raise RuntimeError("AES-CTR unexpectedly buffered stream data")
