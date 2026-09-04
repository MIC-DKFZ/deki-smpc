# Wire format 1.1

Protocol `1.1` uses the existing `/v1` API prefix and all `1.0` resources. The
round status adds `KEY_AGGREGATION` between setup and update collection.

| Method and resource | Purpose |
| --- | --- |
| `GET /v1/rounds/{round_id}/tree-plan` | Canonical plan, plan hash, and signed ephemeral manifest |
| `GET /v1/rounds/{round_id}/key-actions/next` | Next ready action authorized for the caller, or `204` |
| `PUT /v1/rounds/{round_id}/key-tasks/{task_id}/artifact` | Immutable encrypted task output |
| `GET /v1/rounds/{round_id}/key-tasks/{task_id}/artifact` | Completed dependency, only for its receiver |
| `POST /v1/rounds/{round_id}/key-tasks/{task_id}/ack` | Atomically complete a task and activate dependents |
| `PUT /v1/rounds/{round_id}/final-key` | Root publication of the group-encrypted aggregate key |
| `GET /v1/rounds/{round_id}/final-key` | Retrieve the immutable final key artifact |
| `POST /v1/rounds/{round_id}/final-key/ack` | Participant receipt barrier |

Every mutation requires `Idempotency-Key`. Encrypted uploads carry
`X-Artifact-Metadata` (Base64 canonical JSON) and `X-Artifact-Signature`
(Base64 Ed25519). Downloads additionally carry `X-Content-SHA256`. Task
metadata contains only server-authoritative action, task ID, stage, level,
sender, receiver, dependencies, and plan hash; cryptographic metadata adds the
committed schema hash, nonce, ciphertext digest, and size.

Task upload, receipt, and dependent activation are one durable transaction per
operation. Wrong actors, early or changed replays, context mismatches, invalid
signatures/digests, and modified stored artifacts fail the round with sanitized
errors. The original `1.0` resource shapes and fixture are unchanged.
