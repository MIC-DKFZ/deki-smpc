"""Verify that an encrypted-tree modification prevented every client result."""

import time
from pathlib import Path

deadline = time.monotonic() + 60
expected = [Path(f"/coord/client-{index}.tree-tamper-ok") for index in range(1, 4)]
while not all(path.exists() for path in expected):
    if time.monotonic() >= deadline:
        raise RuntimeError("not all clients rejected or timed out after tree-artifact tampering")
    time.sleep(0.2)
print("encrypted tree-artifact modification prevented every client result")
