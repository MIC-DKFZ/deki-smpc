"""Model definition shared by the MNIST participant and round operator."""

from __future__ import annotations

import torch


class MNISTClassifier(torch.nn.Module):
    """A deliberately small network for the getting-started example."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def new_model(seed: int = 2026) -> MNISTClassifier:
    """Build the same initial model at every participant without changing global RNG state."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return MNISTClassifier()
