# Development and tests

## Environment

Create a Python 3.12 environment and install the package in editable mode:

```bash
python -m pip install -e '.[test]'
```

Run the complete local quality gate:

```bash
pre-commit run --all-files
ruff check .
mypy
python -m pytest -q
```

The GitHub Actions workflow runs the same commands for every push and pull
request.

## Test structure

| Path | Coverage |
| --- | --- |
| `tests/test_fixed_point_converter.py` | Fixed-point precision, dtype handling, and overflow limits |
| `tests/test_protocol_v1.py` | Schema commitments, key setup, masks, serialization, tags, and typed client behavior |
| `tests/fixtures/protocol-v1.json` | Shared canonical protocol fixture used by both repositories |
| `examples/` | Three-participant Compose scenarios invoked from the server repository |

The server test suite provides cross-repository end-to-end coverage, including
a three-participant `PlainConvUNet` aggregation and an aggregate-modification
scenario.

## Container validation

Place `deki-smpc` and `deki-smpc-server` next to each other. From the server
repository, run:

```bash
docker compose --profile e2e -p deki_v1_e2e up --build -d
test "$(docker wait deki_v1_e2e-verify-1)" = "0"
docker compose --profile e2e -p deki_v1_e2e logs verify
docker compose --profile e2e -p deki_v1_e2e down --volumes --remove-orphans
```

The scenario starts two API processes, one aggregation worker, three separate
client containers, and five consecutive rounds.

Aggregate-modification validation uses the server repository's tamper Compose
overlay. Its verifier succeeds when every participant reports
`AggregateIntegrityError`.

## Protocol changes

A protocol change updates all of these artifacts in the same review:

- client and server request models;
- `docs/protocol-v1.md`;
- `docs/wire-format-v1.md`;
- the shared canonical fixture;
- client and server protocol tests;
- `CHANGELOG.md` in each repository.

Wire compatibility is identified by the `protocol_version` field. Package
versions follow semantic versioning.
