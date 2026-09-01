"""Train one MNIST shard, securely aggregate the model, and save the result."""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from mnist_common import new_model
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from deki_smpc import FedAvgClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", default=os.environ.get("DEKI_ROUND_ID"))
    parser.add_argument("--site-index", type=int, required=True, help="zero-based shard index")
    parser.add_argument("--site-count", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist-data"))
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, help="optional per-site limit for a quick smoke test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--timeout-minutes", type=float, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--allow-insecure-http", action="store_true")
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def train(model: torch.nn.Module, loader: DataLoader[Any], device: torch.device, epochs: int, lr: float) -> None:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
        print(f"local epoch {epoch + 1}/{epochs}: loss={total_loss / len(loader.dataset):.4f}")


@torch.inference_mode()
def accuracy(model: torch.nn.Module, loader: DataLoader[Any], device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        labels = labels.to(device)
        predictions = model(images.to(device)).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    if not args.round_id:
        raise SystemExit("provide --round-id or DEKI_ROUND_ID")
    if args.site_count < 3 or not 0 <= args.site_index < args.site_count:
        raise SystemExit("site-count must be at least 3 and site-index must be in range")
    if (
        args.local_epochs <= 0
        or args.batch_size <= 0
        or args.timeout_minutes <= 0
        or (args.max_train_samples is not None and args.max_train_samples <= 0)
    ):
        raise SystemExit("epochs, batch size, and timeout must be positive")

    client_id = required_env("DEKI_CLIENT_ID")
    device = torch.device(args.device)
    model = new_model()
    if args.input_checkpoint:
        model.load_state_dict(torch.load(args.input_checkpoint, map_location="cpu", weights_only=True))
    model.to(device)

    transform = transforms.ToTensor()
    training_data = datasets.MNIST(args.data_dir, train=True, download=args.download, transform=transform)
    shard_indices = list(range(args.site_index, len(training_data), args.site_count))
    if args.max_train_samples is not None:
        shard_indices = shard_indices[: args.max_train_samples]
    shard = Subset(training_data, shard_indices)
    generator = torch.Generator().manual_seed(10_000 + args.site_index)
    training_loader = DataLoader(shard, batch_size=args.batch_size, shuffle=True, generator=generator)
    train(model, training_loader, device, args.local_epochs, args.learning_rate)

    trusted_keys = json.loads(required_env("DEKI_TRUSTED_SIGNING_KEYS"))
    if not isinstance(trusted_keys, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in trusted_keys.items()
    ):
        raise SystemExit("DEKI_TRUSTED_SIGNING_KEYS must be a JSON string-to-string object")

    def progress(event: dict[str, object]) -> None:
        print(f"SMPC: {event['event']}", flush=True)

    with FedAvgClient(
        base_url=required_env("DEKI_BASE_URL"),
        federation_id=required_env("DEKI_FEDERATION_ID"),
        client_id=client_id,
        auth_token=required_env("DEKI_AUTH_TOKEN"),
        identity_private_key=required_env("DEKI_IDENTITY_PRIVATE_KEY"),
        trusted_signing_keys=trusted_keys,
        allow_insecure_http=args.allow_insecure_http,
        ca_bundle=os.environ.get("DEKI_CA_BUNDLE"),
    ) as client:
        aggregate = client.aggregate(
            model=model,
            round_id=args.round_id,
            timeout=timedelta(minutes=args.timeout_minutes),
            progress_callback=progress,
        )
    model.load_state_dict(aggregate)

    if not args.skip_evaluation:
        test_data = datasets.MNIST(args.data_dir, train=False, download=args.download, transform=transform)
        test_loader = DataLoader(test_data, batch_size=512)
        print(f"aggregated test accuracy: {accuracy(model, test_loader, device):.2%}")

    output = args.output_checkpoint or Path("mnist-checkpoints") / f"{client_id}-{args.round_id}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, output)
    print(f"saved verified aggregate to {output}")


if __name__ == "__main__":
    main()
