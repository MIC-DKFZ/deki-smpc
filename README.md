# deki-smpc

`deki-smpc` is a lightweight Python client for secure federated averaging using SMPC-style masking.
It is designed to integrate into existing training loops (PyTorch, nnU-Net, MONAI, or custom code) with minimal workflow changes.

## What It Solves

- Keep your local training loop unchanged.
- Add secure aggregation with one client object and one aggregation call.
- Exchange keys and aggregate model weights through a dedicated server stack.

## Quick Start

### 1. Requirements

- Python `>=3.11`
- A running [deki-smpc-server](https://github.com/MIC-DKFZ/deki-smpc-server)
- Same model architecture on all participating clients
- Shared preshared secret across clients (meeting complexity requirements)

### 2. Install

```bash
git clone https://github.com/MIC-DKFZ/deki-smpc
cd deki-smpc
pip install -e .
```

### 3. Minimal Usage

```python
import torch
from deki_smpc import FedAvgClient

model = ...  # your local torch.nn.Module

client = FedAvgClient(
    aggregation_server_ip="127.0.0.1",
    aggregation_server_port=8080,
    num_clients=4,
    preshared_secret="my_secure_presHared_secret_123!",
    client_name="site_a",  # must be unique per client
    model=model,
)

# Returns aggregated model weights as a state_dict
aggregated_state_dict = client.aggregate()
model.load_state_dict(aggregated_state_dict)
```

## Integration Pattern

Use this package at aggregation points in your existing training loop:

1. Train locally for one round/epoch window.
2. Call `client.aggregate()`.
3. Load returned weights via `model.load_state_dict(...)`.
4. Continue local training.

This lets you control local optimization logic while delegating secure cross-client aggregation.

## API Reference

### `FedAvgClient(...)`

Main constructor arguments:

- `aggregation_server_ip: str`: Host/IP of the aggregation service.
- `aggregation_server_port: int`: Port of the aggregation service.
- `num_clients: int`: Number of expected participants (must be `>= 3`).
- `preshared_secret: str`: Shared secret used for registration/auth checks.
- `client_name: str`: Unique identifier of this client instance.
- `model: torch.nn.Module`: Local model whose `state_dict` will be aggregated.
- `ignore_model_keys: list[str] | None`: Optional keys to exclude from conversion/aggregation.
- `logging_level: int`: Standard Python logging level.

### Methods

- `prepare_transfer() -> None`: Runs key aggregation phases only.
- `aggregate() -> dict[str, torch.Tensor]`: Executes secure aggregation and returns averaged weights.

## Security Notes

- This client masks model parameters before transfer and coordinates multi-phase key aggregation.
- It helps protect individual client updates during aggregation, but does not replace end-to-end operational security.
- Use secure networking and secret management in production (private network, TLS/termination, secret rotation, access control).

## Troubleshooting

- `AssertionError: Number of clients must be at least 3`
  Set `num_clients` to the real participant count and keep it consistent across clients.

- Stuck waiting/polling
  Confirm all clients are online, registered with unique `client_name`, and connected to the same server instance.

- Secret validation failures
  Ensure the preshared secret matches on all clients and satisfies minimum complexity requirements.

- Model key mismatches
  Ensure identical model architecture/state keys on all participants, or explicitly use `ignore_model_keys`.

## Why deki-smpc

Most FL frameworks impose orchestration patterns.
`deki-smpc` focuses on secure aggregation while letting you keep your existing training code and control flow.

## Citation

Hamm, B., Kirchhoff, Y., Rokuss, M., Schader, P., Neher, P., Parampottupadam, S., Floca, R., Maier-Hein, K. (2025). Efficient Privacy-Preserving Medical Cross-Silo Federated Learning. https://doi.org/10.36227/techrxiv.174650601.13181048/v1
