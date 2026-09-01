"""Wait for all independent clients and report smoke-test success."""

import time
from pathlib import Path

deadline = time.monotonic() + 120
expected = [Path(f"/coord/client-{index}.ok") for index in range(1, 4)]
while not all(path.exists() for path in expected):
    if time.monotonic() >= deadline:
        raise RuntimeError("not all clients completed with verified results")
    time.sleep(0.2)
print("three clients completed five consecutive integrity-verified container rounds")
