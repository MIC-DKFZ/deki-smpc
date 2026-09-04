# deki-smpc v1 wire format

This document specifies canonical encodings and HTTP resources for protocol v1.
This document specifies legacy wire value `1.0`, and all protocol resources use the `/v1`
prefix.

Package 1.0.1 also supports additive [wire value 1.1](wire-format-v1.1.md).

## HTTP transport

Production deployments use HTTPS. Participant requests carry
`Authorization: Bearer <participant-token>`. Operator requests use the same
header form with the independent operator token.

Every mutation also carries an `Idempotency-Key` with 1 to 128 characters.
Clients generate a separate key for each logical operation and retain it across
transport retries.

## JSON encoding

Small request and response bodies use UTF-8 JSON. Strict request models reject
unknown fields. Canonical JSON has:

- object keys in lexical order;
- compact separators `,` and `:`;
- UTF-8 output;
- Unicode characters represented directly.

Hashes are lowercase, 64-character SHA-256 hexadecimal strings. Round IDs use
canonical UUID text. X25519 and Ed25519 raw 32-byte public keys use padded
standard Base64. AES-GCM nonce and ciphertext values use the same Base64
encoding.

## Tensor artifacts

Update and result bodies use safetensors with media type
`application/vnd.safetensors`. Every included tensor is CPU-contiguous signed
int64. Tensor names and shapes match the committed model schema exactly.

The update request headers are:

| Header | Encoding |
| --- | --- |
| `Content-Type` | `application/vnd.safetensors` |
| `X-Model-Schema-Hash` | SHA-256 hexadecimal |
| `X-Integrity-Tag` | 16-byte big-endian field element as 32 lowercase hexadecimal characters |
| `Idempotency-Key` | Operation identifier, at most 128 characters |

The result response headers are:

| Header | Encoding |
| --- | --- |
| `Content-Type` | `application/vnd.safetensors` |
| `X-Integrity-Tag` | Aggregated prime-field tag |
| `X-Content-SHA256` | SHA-256 of the response body |
| `Cache-Control` | `no-store` |

Arithmetic follows the sorted tensor order from the committed schema.
Safetensors stores numeric tensor data in little-endian representation.

## Resources

| Method | Resource | Actor | Body |
| --- | --- | --- | --- |
| `POST` | `/v1/federations/{federation_id}/rounds` | Operator | Round manifest and model schema |
| `GET` | `/v1/rounds/{round_id}` | Participant | Empty |
| `GET` | `/v1/rounds/{round_id}/participants/me` | Participant | Empty |
| `PUT` | `/v1/rounds/{round_id}/artifacts/public_key` | Participant | Signed ephemeral key JSON |
| `GET` | `/v1/rounds/{round_id}/artifacts/public_keys` | Participant | Empty |
| `PUT` | `/v1/rounds/{round_id}/artifacts/key_bundle` | Participant | Signed encrypted envelopes JSON |
| `GET` | `/v1/rounds/{round_id}/artifacts/key_bundle` | Participant | Empty |
| `POST` | `/v1/rounds/{round_id}/key-setup/complete` | Participant | Key-context commitment JSON |
| `PUT` | `/v1/rounds/{round_id}/artifacts/update` | Participant | Safetensors artifact |
| `GET` | `/v1/rounds/{round_id}/artifacts/result` | Participant | Empty |
| `POST` | `/v1/rounds/{round_id}/complete` | Participant | Empty JSON object |
| `POST` | `/v1/rounds/{round_id}/abort` | Operator | Abort reason JSON |

The server repository contains the full reference in `docs/api-v1.md`.

## Polling responses

Pending key bundles return HTTP 204 with `Retry-After`. Round phase barriers
and pending results return a structured conflict response with a stable code.
The client combines this response with capped exponential backoff and the total
round deadline.

## Error envelope

Protocol errors use the following JSON shape:

```json
{
  "detail": {
    "code": "ARTIFACT_INVALID",
    "message": "artifact validation failed",
    "round_id": "c15ff8be-10a2-4b78-aad8-3c292ac630df"
  }
}
```

The `round_id` field is present for round-scoped errors. Error content is
limited to identifiers and sanitized diagnostic text.
