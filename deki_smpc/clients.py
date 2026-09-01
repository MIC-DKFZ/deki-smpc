"""High-level reusable v1 secure-aggregation client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Self

import httpx
import torch

from .protocol.errors import ConfigurationError
from .protocol.schema import TensorPolicy
from .round_session import AggregationRoundSession, load_identity_key, load_trusted_keys
from .transport import V1Transport

TensorStateDict = dict[str, torch.Tensor]


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

    def aggregate(
        self,
        *,
        model: torch.nn.Module,
        round_id: str,
        timeout: timedelta = timedelta(minutes=30),
        tensor_policies: Mapping[str, TensorPolicy | str] | None = None,
        cancellation_token: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> TensorStateDict:
        if timeout.total_seconds() <= 0:
            raise ConfigurationError("timeout must be positive")
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
