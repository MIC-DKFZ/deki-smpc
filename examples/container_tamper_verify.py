"""Verify that every client rejected the tampered aggregate."""

import time
from pathlib import Path

deadline = time.monotonic() + 120
expected = [Path(f"/coord/client-{index}.tamper-ok") for index in range(1, 4)]
while not all(path.exists() for path in expected):
    if time.monotonic() >= deadline:
        raise RuntimeError("not all clients detected aggregate tampering")
    time.sleep(0.2)
print("all three clients rejected the valid-safetensors aggregate modification")
