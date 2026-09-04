"""High-level reusable v1 secure-aggregation client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from threading import Lock
from typing import Any, Self

import httpx
import torch

from .protocol.errors import ArtifactValidationError, ConfigurationError
from .protocol.schema import TensorPolicy
from .round_session import AggregationRoundSession, PreparedRoundState, load_identity_key, load_trusted_keys
from .transport import V1Transport

TensorStateDict = dict[str, torch.Tensor]


class PreparedRound:
    """Opaque, synchronous, one-shot preparation handle."""

    __slots__ = ("_client_token", "_consumed", "_lock", "_round_id", "_session", "_state")

    def __init__(self, client_token: object, session: AggregationRoundSession, state: PreparedRoundState) -> None:
        self._client_token = client_token
        self._round_id = session.round_id
        self._session = session
        self._state = state
        self._consumed = False
        self._lock = Lock()

    def _take(self, client_token: object, round_id: str) -> tuple[AggregationRoundSession, PreparedRoundState]:
        with self._lock:
            if self._consumed:
                raise ArtifactValidationError("prepared round handle was already consumed")
            self._consumed = True
        if client_token is not self._client_token:
            self.close()
            raise ArtifactValidationError("prepared round belongs to another client")
        if round_id != self._round_id:
            self.close()
            raise ArtifactValidationError("prepared round belongs to another round")
        return self._session, self._state

    def close(self) -> None:
        with self._lock:
            self._consumed = True
        self._state.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> Any:
        raise TypeError("PreparedRound handles cannot be serialized")


class FedAvgClient:
    """Reusable connection factory; all cryptographic state is round-local."""

    def __init__(
        self,
        *,
        base_url: str,
        federation_id: str,
        client_id: str,
        auth_token: str,
        identity_private_key: str,
        trusted_signing_keys: Mapping[str, str],
        allow_insecure_http: bool = False,
        ca_bundle: str | bool | None = None,
        client_certificate: tuple[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not federation_id or not client_id or not auth_token:
            raise ConfigurationError("federation_id, client_id and auth_token are required")
        self.federation_id = federation_id
        self.client_id = client_id
        self.identity_key = load_identity_key(identity_private_key)
        self.trusted_signing_keys = load_trusted_keys(trusted_signing_keys)
        if client_id not in self.trusted_signing_keys:
            raise ConfigurationError("trusted_signing_keys must include this client")
        self.transport = V1Transport(
            base_url,
            auth_token,
            allow_insecure_http=allow_insecure_http,
            ca_bundle=ca_bundle,
            client_certificate=client_certificate,
            client=http_client,
        )
        self._prepared_round_owner = object()

    def prepare_round(
        self,
        *,
        model: torch.nn.Module,
        round_id: str,
        timeout: timedelta = timedelta(minutes=30),
        tensor_policies: Mapping[str, TensorPolicy | str] | None = None,
        cancellation_token: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> PreparedRound:
        """Finish round-local key setup so applications can overlap it with training."""
        if timeout.total_seconds() <= 0:
            raise ConfigurationError("timeout must be positive")
        session = AggregationRoundSession(
            self.transport,
            self.federation_id,
            self.client_id,
            round_id,
            model,
            self.identity_key,
            self.trusted_signing_keys,
            timeout.total_seconds(),
            tensor_policies,
            cancellation_token,
            progress_callback,
        )
        return PreparedRound(self._prepared_round_owner, session, session.prepare())

    def aggregate(
        self,
        *,
        model: torch.nn.Module,
        round_id: str,
        timeout: timedelta = timedelta(minutes=30),
        tensor_policies: Mapping[str, TensorPolicy | str] | None = None,
        cancellation_token: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        prepared_round: PreparedRound | None = None,
    ) -> TensorStateDict:
        if timeout.total_seconds() <= 0:
            raise ConfigurationError("timeout must be positive")
        if prepared_round is not None:
            session, state = prepared_round._take(self._prepared_round_owner, round_id)
            session.model = model
            if tensor_policies is not None:
                session.tensor_policies = tensor_policies
            if cancellation_token is not None:
                session.cancellation = cancellation_token
            if progress_callback is not None:
                session.progress = progress_callback
            try:
                return session.run_prepared(state)
            except Exception:
                state.close()
                raise
        return AggregationRoundSession(
            self.transport,
            self.federation_id,
            self.client_id,
            round_id,
            model,
            self.identity_key,
            self.trusted_signing_keys,
            timeout.total_seconds(),
            tensor_policies,
            cancellation_token,
            progress_callback,
        ).run()

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
