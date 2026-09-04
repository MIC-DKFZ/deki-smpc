"""Reproducible non-CI protocol 1.0/1.1 preparation-work benchmark."""

from __future__ import annotations

import argparse
import platform
import statistics
import time

import torch

from deki_smpc.protocol.models import RoundContext
from deki_smpc.protocol.schema import ModelSchema
from deki_smpc.protocol.tree import derive_tree_plan, seed_tensors


def measure_streams(streams: int, elements: int, repetitions: int, purpose: str) -> float:
    schema = ModelSchema.from_state_dict({"weight": torch.zeros(elements)})
    context = RoundContext("benchmark", "round", "client", "1.1", schema.hash, "0" * 64, "EQUAL_WEIGHTED", 24)

    def run(repetition: int) -> float:
        started = time.perf_counter()
        for index in range(streams):
            tensors = seed_tensors(bytes([index % 251 + 1]) * 32, schema, context, f"{purpose}:{repetition}:{index}")
            tensors["weight"].zero_()
        return time.perf_counter() - started

    run(-1)
    samples = [run(repetition) for repetition in range(repetitions)]
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, nargs="+", default=[3, 5, 7, 12, 20, 64])
    parser.add_argument("--elements", type=int, default=100_000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    context = RoundContext("benchmark", "round", "client", "1.1", "1" * 64, "2" * 64, "EQUAL_WEIGHTED", 24)
    print(f"python={platform.python_version()},torch={torch.__version__},machine={platform.machine()}")
    print("participants,protocol,preparation_seconds,mask_streams,model_sized_bytes,critical_path_hops,tree_levels")
    for count in args.participants:
        manifest = [
            {"client_id": f"client-{index}", "public_key": f"key-{index}", "signature": f"signature-{index}"}
            for index in range(count)
        ]
        plan = derive_tree_plan(context, manifest)
        levels = int(plan["tree_levels"])
        model_bytes = args.elements * 8
        legacy_time = measure_streams(count - 1, args.elements, args.repetitions, "legacy-pairwise")
        tree_time = measure_streams(2, args.elements, args.repetitions, "tree-coordinator-worst-case")
        print(f"{count},1.0,{legacy_time:.6f},{count - 1},0,2,0")
        # The plan includes explicit carry artifacts for odd-width binary
        # levels, so count the canonical tasks instead of assuming a full tree.
        key_artifacts = len(plan["tasks"]) + 1  # one final distribution artifact
        critical_hops = max(len(group["members"]) for group in plan["groups"]) + levels + 2
        print(f"{count},1.1,{tree_time:.6f},2,{key_artifacts * model_bytes},{critical_hops},{levels}")


if __name__ == "__main__":
    main()
