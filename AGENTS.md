# Agent Guide

## Scope

This file applies to the entire `deki-smpc` repository. It is operational
guidance for contributors and coding agents; the protocol and security
documents remain the normative description of deki-smpc v1.

Keep changes focused, preserve unrelated work, and prefer the smallest change
that satisfies the task. Do not silently change public API, wire behavior,
cryptographic domains, or compatibility with `deki-smpc-server`.

## Repository Map

- `deki_smpc/clients.py`: public `FedAvgClient` API and client lifecycle.
- `deki_smpc/round_session.py`: bounded orchestration of one aggregation round.
- `deki_smpc/transport.py`: authenticated HTTP transport, retries, deadlines,
  error translation, and idempotency keys.
- `deki_smpc/protocol/`: schemas, canonical encoding, key setup, masking,
  serialization, integrity verification, models, and public protocol errors.
- `deki_smpc/utils.py`: fixed-point conversion and overflow protection.
- `tests/`: unit and protocol tests. `tests/fixtures/protocol-v1.json` is the
  canonical fixture consumed by both repositories.
- `examples/`: client, operator, tamper, and MNIST integration entry points.
- `docs/`: public API, protocol, wire format, security model, and development
  documentation.
- `.github/workflows/ci.yml` and `Dockerfile`: CI and production image.

Treat `build/`, `dist/`, `*.egg-info/`, caches, `.demo-secrets/`,
`.demo-server/`, MNIST data/checkpoints, and the legacy `example/` directory as
generated or local state. Do not edit or commit them.

## Environment and Quality Gate

Python 3.12 or newer is required. Set up an editable development environment
with:

```bash
python -m pip install -e '.[test]'
```

Run the complete local quality gate from the repository root:

```bash
pre-commit run --all-files
ruff check .
mypy
python -m pytest -q
```

Use `pre-commit run black --all-files` when formatting changes are intended.
For focused work, run the narrowest relevant test first, for example:

```bash
python -m pytest -q tests/test_protocol_v1.py
python -m pytest -q tests/test_fixed_point_converter.py
```

Build the production image after Dockerfile, packaging, Python-version, or
runtime-dependency changes:

```bash
docker build --pull --tag deki-smpc:local .
docker run --rm deki-smpc:local -c 'import deki_smpc'
```

## Architecture and Coding Conventions

- Keep the public API small and typed. Add public exports deliberately and
  document user-visible behavior in `docs/client-api.md` and `README.md`.
- Keep protocol primitives independent from HTTP orchestration. Transport code
  belongs in `transport.py`; round sequencing belongs in `round_session.py`.
- Follow strict mypy settings, Black formatting, Ruff linting, four-space
  indentation, and a 120-character line limit. Use `snake_case` for functions
  and variables,
  `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Prefer explicit typed exceptions from `deki_smpc.protocol.errors` at public
  boundaries. Preserve the server's stable error-code mapping.
- Keep canonical operations deterministic: sorted tensor names, compact UTF-8
  JSON, exact hashes, fixed-width field encodings, and domain-separated inputs.
- Preserve bounded-memory processing for model-sized data. Do not introduce
  avoidable full-artifact copies or unbounded polling/retry loops.
- Do not mutate the caller's model or state dictionary. Aggregation returns a
  new, verified state dictionary.
- Tests must be deterministic, offline, and independent of real credentials,
  external services, GPUs, and machine-specific paths unless explicitly marked
  as integration tests.

## Protocol and Security Invariants

Changes in cryptographic or protocol code require adversarial review, not only
happy-path tests. Preserve these invariants unless a versioned protocol change
explicitly replaces them:

- Protocol wire value `1.0` uses a complete, ordered set of at least three
  participants and equal-weight aggregation.
- Ed25519 identities are pinned; X25519 keys, shares, masks, and integrity
  material are fresh and round-local. Round context and purpose remain in every
  signature/KDF/PRF domain.
- Pairwise int64 masks cancel under two's-complement ring addition. Fixed-point
  encoding reserves headroom for the full committed participant count and
  rejects non-finite or unsafe values.
- The client verifies content digest, safetensors structure, tensor schema, and
  the prime-field aggregate tag before decoding or returning any result. Keep
  integrity-tag comparison constant-time.
- `KEEP_LOCAL` tensors never enter uploaded artifacts and retain each caller's
  local value. `MEAN` and `SUM` semantics must match the committed schema.
- Serialization remains safetensors-only for tensor artifacts; never add
  pickle-based loading for untrusted data.
- HTTPS is the default. Insecure HTTP remains an explicit development opt-in.
  Never log or expose bearer tokens, private keys, decrypted shares, mask keys,
  integrity seeds, or sensitive server error bodies.
- One caller-supplied deadline bounds the complete round. Retries preserve the
  logical idempotency key and remain cancellation-aware.
- Round key material is consumed after successful completion and is never
  reused across rounds.

When touching arithmetic, canonicalization, signatures, key derivation,
serialization, retries, or result validation, add tests for malformed,
replayed, mismatched, boundary, and tampered inputs as applicable.

## Cross-Repository Changes

`deki-smpc` and sibling `../deki-smpc-server` implement one wire contract. A
protocol or wire-format change is incomplete until the following stay aligned:

- client transport/models and server request/response handling;
- `docs/protocol-v1.md`, `docs/wire-format-v1.md`, and the server API/protocol
  documentation;
- `tests/fixtures/protocol-v1.json` and protocol tests in both repositories;
- stable error codes and typed client exceptions;
- `CHANGELOG.md` in each affected repository.

Do not change the meaning of wire value `1.0` incompatibly. Use an explicit new
protocol version for breaking wire changes. With both repositories installed
as siblings, validate server integration after any shared-contract change:

```bash
cd ../deki-smpc-server
python -m pip install -e ../deki-smpc
python -m pip install -e '.[test]'
python -m pytest -q
```

For high-risk protocol, container, or orchestration changes, also run the
Compose end-to-end and tamper scenarios documented in
`docs/development.md` and the server `README.md`.

## Documentation and Release Hygiene

- Update documentation when public APIs, configuration, defaults, errors,
  security assumptions, wire fields, or operator workflows change.
- Add user-visible changes under `Unreleased` in `CHANGELOG.md`.
- Keep `requires-python`, Ruff, mypy, CI, Docker, README, and development docs
  synchronized when changing Python support.
- Keep dependency ranges in `pyproject.toml`; preserve the CPU PyTorch install
  strategy in the Dockerfile unless the deployment contract changes.
- Never commit real tokens, identity keys, federation manifests, model data,
  checkpoints, logs, or generated artifacts. Example credentials must be
  obviously development-only.
