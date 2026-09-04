# Protocol preparation benchmark for 1.0.1

This non-CI benchmark compares deterministic local model-mask expansion and the
topology communication model. It intentionally has no pass/fail wall-clock
threshold. It does not measure network, serialization, server scheduling, or
end-to-end training time, so the results support no general end-to-end speedup
claim.

Command:

```bash
python -m benchmarks.protocol_versions --elements 100000 --repetitions 11
```

Observed on Python 3.12.2, PyTorch 2.13.0+cpu, x86-64:

| Sites | 1.0 local prep | 1.1 worst coordinator prep | 1.0 streams | 1.1 streams | 1.1 model-sized bytes | Tree levels | 1.1 critical hops |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.965 ms | 1.896 ms | 2 | 2 | 4,000,000 | 0 | 5 |
| 5 | 2.205 ms | 0.667 ms | 4 | 2 | 5,600,000 | 0 | 7 |
| 7 | 1.851 ms | 1.058 ms | 6 | 2 | 8,800,000 | 1 | 7 |
| 12 | 3.625 ms | 0.968 ms | 11 | 2 | 15,200,000 | 2 | 8 |
| 20 | 6.295 ms | 1.189 ms | 19 | 2 | 25,600,000 | 3 | 9 |
| 64 | 21.734 ms | 0.981 ms | 63 | 2 | 76,800,000 | 4 | 10 |

Each synthetic model has 100,000 int64 values (800,000 bytes). Protocol `1.0`
expands one pairwise stream per peer and sends no model-sized key artifacts.
The `1.1` worst-case coordinator expands its private key plus one blind, while a
non-coordinator expands only its private key for a group task. Its byte column
is the federation total for every canonical group/tree task plus the final-key
artifact. Explicit carry tasks at odd-width levels are included. The measured
local work scaled with peer count for `1.0` and stayed at two streams for `1.1`;
`1.1` trades that reduction for model-sized encrypted key traffic and
additional critical-path hops.
