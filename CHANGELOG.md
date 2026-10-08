# Changelog

This file records released changes to `deki-smpc`.

## [1.0.3] - 2026-10-08

### Documentation

- Updated the citation to the published IEEE JBHI article
  (doi:10.1109/JBHI.2026.3740976) and added a BibTeX entry.

## [1.0.2] - 2026-10-07

### Documentation

- Replaced the README overview diagram with an animated walkthrough of the
  secure aggregation flow.
- Documented the CKKS, BFV, and BGV encryption and decryption pipelines used as
  experimental FHE baselines. They are not part of the protocol or client API.

## [1.0.1] - 2026-09-04

### Protocol and security

- Added wire value `1.1` with deterministic balanced groups, blinded ring key
  aggregation, logarithmic binary reduction, and one group-encrypted final-key
  distribution. New rounds default to `1.1`; wire value `1.0` is unchanged.
- Added X25519/HKDF/AES-256-GCM encryption and Ed25519 authentication for every
  intermediate safetensors key artifact, with authoritative task context bound
  into associated data and signatures.
- Kept model-mask seeds separate from integrity shares. Under `1.1`, the server
  aggregates updates that remain masked and clients remove the distributed
  aggregate key before the existing constant-time integrity check.

### Client

- Added `FedAvgClient.prepare_round(...) -> PreparedRound` and the optional
  `prepared_round` argument to `aggregate` for application-managed overlap with
  local training. Handles are bound, one-shot, non-serializable, deadline-aware,
  context-manageable, and clear sensitive state on close or consumption.

### Validation and documentation

- Added canonical `1.1` tree fixtures, deterministic topology and encrypted
  artifact tests, and cross-repository 3/5/7/12-participant end-to-end cases.
- Documented both protocol values, topology derivation, performance model,
  security limitations, wire resources, and rolling deployment.

## [1.0.0] - 2026-09-01

Initial deki-smpc v1 release.

### Fixed

- Derived the participant count from the locally pinned identity manifest
  instead of the server's `expected_participant_count`. The value drives the
  fixed-point aggregation bound and the `MEAN` divisor, and no signature
  covered it, so a modified count silently rescaled the released aggregate
  without failing the prime-field integrity check.
- Refused rounds with fewer than three pinned participants, where pairwise
  masking cannot hide an individual update.

### Changed

- Standardized product naming as `deki-smpc` throughout the repository.
- Reworked the README around a high-level introduction, visual overview, and streamlined getting-started flow.
- Raised the minimum supported Python version to 3.12.
- Made Black the authoritative formatter, with isort running before it through pre-commit.

### Protocol

- Added round-scoped secure aggregation for committed participant sets.
- Added authenticated ephemeral key setup with Ed25519, X25519, HKDF-SHA256,
  AES-256-GCM, and AES-256-CTR.
- Added pairwise cancelling int64 model masks.
- Added random linear aggregate verification in the prime field `2**127 - 1`.
- Added `MEAN`, `SUM`, and `KEEP_LOCAL` tensor policies.
- Added canonical model schemas and protocol wire value `1.0`.

### Client

- Added the reusable `FedAvgClient` interface.
- Added one deadline across transport retries and protocol phase waits.
- Added cancellation, progress callbacks, and typed public exceptions.
- Added safetensors artifact serialization and content verification.
- Added fixed-point range checks for the committed participant count.

### Validation

- Added protocol unit tests and a shared canonical fixture.
- Added a three-participant `PlainConvUNet` end-to-end aggregation test.
- Added repeated-round and aggregate-modification container scenarios.
