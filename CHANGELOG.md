# Changelog

This file records released changes to `deki-smpc`.

## [Unreleased]

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

## [1.0.0] - 2026-08-30

Initial deki-smpc v1 release.

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
