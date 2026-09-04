# Client API

The public client interface consists of `FedAvgClient` and the exception types
exported by `deki_smpc`.

## FedAvgClient

```python
FedAvgClient(
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
)
```

| Parameter | Description |
| --- | --- |
| `base_url` | Absolute URL of the aggregation API |
| `federation_id` | Stable identifier of the enrolled federation |
| `client_id` | Stable identifier of the local participant |
| `auth_token` | Participant bearer token |
| `identity_private_key` | Padded Base64 encoding of a raw 32-byte Ed25519 private key |
| `trusted_signing_keys` | Complete participant map of IDs to padded Base64 raw Ed25519 public keys |
| `allow_insecure_http` | Enables plain HTTP for isolated integration environments |
| `ca_bundle` | CA bundle path or an HTTPX verification value |
| `client_certificate` | Certificate and key paths for mutual TLS |
| `http_client` | Application-managed `httpx.Client` for transport integration |

The local `client_id` appears in `trusted_signing_keys`. The manifest hash of
this complete map must equal the manifest committed by the operator when the
round is created.

`FedAvgClient` supports the context-manager protocol. `close()` releases an
internally created HTTP client. An injected HTTP client remains under
application ownership.

## aggregate

```python
client.aggregate(
    *,
    model: torch.nn.Module,
    round_id: str,
    timeout: timedelta = timedelta(minutes=30),
    tensor_policies: Mapping[str, TensorPolicy | str] | None = None,
    cancellation_token: Callable[[], bool] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    prepared_round: PreparedRound | None = None,
) -> dict[str, torch.Tensor]
```

The method reads a detached copy of `model.state_dict()`, executes one complete
v1 round, verifies the result, and returns a new state dictionary. Returned
tensors use the dtype and device of the corresponding local tensor.

`timeout` is one deadline for the complete operation, including retries and
phase waits. A cancellation callback returns `True` to stop the operation.

The progress callback receives dictionaries with `event` and `round_id`. v1
emits these events in protocol order:

- `joined`
- `key_setup_complete`
- `update_uploaded`, including the encoded byte count
- `completed`

Progress events describe completed local actions. Round status remains
authoritative on the server.

## prepare_round

`prepare_round(...) -> PreparedRound` accepts the same model, round, timeout,
policy, cancellation, and progress arguments as `aggregate`. It synchronously
finishes key setup and, for protocol `1.1`, the group/tree protocol and final-key
receipt. The returned opaque handle is bound to that client, round, protocol,
schema, policies, and the original absolute deadline.

The handle is non-serializable, supports `with`, and is consumed as soon as an
aggregation attempt starts—even if validation then fails. Closing an unused
handle consumes its key material. `aggregate(..., prepared_round=handle)` does
not reset the preparation deadline.

Applications may submit `prepare_round` to their own executor while local
training runs, then pass the completed handle to `aggregate`. Use the same model
structure and tensor policies; model values may change during training.

## Exceptions

All public exceptions derive from `DekiSMPCError`.

| Exception | Meaning |
| --- | --- |
| `ConfigurationError` | Invalid local configuration or protocol version |
| `AuthenticationError` | Authentication or authorization failure |
| `ProtocolVersionError` | Unsupported server protocol value |
| `RoundConflictError` | Request conflicts with round state or idempotency record |
| `RoundExpiredError` | Round deadline elapsed |
| `RoundFailedError` | Server recorded a terminal round failure or abort |
| `ArtifactValidationError` | Schema, signature, serialization, or content validation failure |
| `AggregateIntegrityError` | Random linear aggregate check failed |
| `TransportError` | HTTP or retry-budget failure |
| `RoundTimeoutError` | Caller deadline or cancellation ended the operation |

Round-scoped server errors populate the exception's `round_id` attribute.

## Operational guidance

Create one client object per participant identity and reuse it across sequential
rounds. Run one active aggregation per client object. Store identity keys and
bearer tokens in the site's secret-management system. Pin the federation CA or
the expected public CA chain through `ca_bundle`.
