"""Authenticated, bounded HTTP transport for protocol v1."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from .protocol.errors import ERROR_TYPES, RoundTimeoutError, TransportError


class V1Transport:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        allow_insecure_http: bool = False,
        ca_bundle: str | bool | None = None,
        client_certificate: tuple[str, str] | None = None,
        max_attempts: int = 5,
        client: httpx.Client | None = None,
    ) -> None:
        if base_url.startswith("http://") and not allow_insecure_http:
            raise ValueError("plain HTTP requires allow_insecure_http=True")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self._owns_client = client is None
        verify: str | bool = True if ca_bundle is None else ca_bundle
        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {token}"},
            verify=verify,
            cert=client_certificate,
            timeout=httpx.Timeout(connect=5, read=30, write=30, pool=5),
        )
        self.client.headers["Authorization"] = f"Bearer {token}"

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        deadline: float,
        cancellation: Callable[[], bool] | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            if cancellation and cancellation():
                raise RoundTimeoutError("operation cancelled")
            if time.monotonic() >= deadline:
                raise RoundTimeoutError("operation deadline exceeded")
            try:
                response = self.client.request(method, path, headers=headers, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last = exc
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in {429, 502, 503, 504}:
                    self._raise_response(response)
                last = TransportError(f"server returned retryable HTTP {response.status_code}")
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        self._wait(min(float(retry_after), 5), deadline, cancellation)
                        continue
                    except ValueError:
                        pass
            if attempt + 1 < self.max_attempts:
                self._wait(min(0.1 * 2**attempt + random.uniform(0, 0.05), 2), deadline, cancellation)
        raise TransportError("transport retry budget exhausted") from last

    @staticmethod
    def _wait(delay: float, deadline: float, cancellation: Callable[[], bool] | None) -> None:
        end = time.monotonic() + min(delay, max(0.0, deadline - time.monotonic()))
        while time.monotonic() < end:
            if cancellation and cancellation():
                raise RoundTimeoutError("operation cancelled")
            time.sleep(min(0.05, max(0, end - time.monotonic())))

    @staticmethod
    def _raise_response(response: httpx.Response) -> None:
        try:
            detail = response.json().get("detail", {})
            if isinstance(detail, dict):
                code = str(detail.get("code", "TRANSPORT_ERROR"))
                message = str(detail.get("message", "request failed"))
                round_id = detail.get("round_id")
            else:
                code, message, round_id = "TRANSPORT_ERROR", str(detail), None
        except (ValueError, json.JSONDecodeError):
            code, message, round_id = "TRANSPORT_ERROR", "request failed", None
        raise ERROR_TYPES.get(code, TransportError)(message, round_id=round_id)


def stable_idempotency_key(operation: str, round_id: str, content: bytes) -> str:
    return hashlib.sha256(
        b"deki-idempotency-v1\0" + operation.encode() + b"\0" + round_id.encode() + b"\0" + content
    ).hexdigest()
