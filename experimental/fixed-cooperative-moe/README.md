# Fixed-shape cooperative EXL3 MoE — opt-in review checkpoint

This directory checkpoints the tested performance work for an upstream PR. It
does **not** alter the recipe's default Docker build, overlay, or startup script.
Do not automatically enable it for other models/shapes or repin binary hashes.
The currently deployed instance uses a separate, explicitly selected overlay.

## Scope and provenance

The device kernel derives from Turboderp's ExLlamaV3 two-stage cooperative MoE
implementation, commit `58d4d7322a1b3bd70aae8412487b21cc5e205cf4`, as present in
pinned source `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`. See
[upstream change](https://github.com/turboderp-org/exllamav3/commit/58d4d7322a1b3bd70aae8412487b21cc5e205cf4)
and `native/LICENSE.exllamav3` for attribution/license.

Changes relative to that kernel are a separate namespace plus a helper fixing
the known DS4.1 dimensions/invariants at three entry points. Arithmetic in the
device body is otherwise unchanged. The parameter header only guards unused
ATen declarations so the standalone CUDA C ABI can build without Torch headers.
The original recipe's installed extension is not replaced.

- SM121a, H=5120, local I=1152, top-k=6, TP=2, K2/K3 mul1 expert projections.
- Wide/wide tiles, 1–8 physical rows, shared preallocated scratch.
- Requires `DSV41_EXL3_SERIAL_STREAMS=1` and
  `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`.
- K4 MTP, larger prefill, mixed/unsupported metadata and shapes stay stock.
- No stock retry after a partially launched native CUDA error.
- Stock baseline: recipe `979e68a62c90b24d928f5638596e0ceed90e9f34`;
  installed ExLlamaV3 1.4.5 commit `e648f1a131365aae15920073e761a3fa5a527654`.

## Measured results

Both profiles used the same checkpoint/quantization/image, 600K maximum context,
two-request capacity, fixed DSpark k=3, native FP4 KV, 3072 batched-token budget,
2816 long-prefill threshold and an eight-row fused threshold. No power, clock,
network, memory-limit or quantization changes were made for this comparison.

| Workload | Stock | Candidate | Relative change |
|---|---:|---:|---:|
| Poetry, median of 3 seeded samples | 23.62 tok/s | 29.26 tok/s | +23.9% |
| Coding, median of 3 seeded samples | 38.76 tok/s | 42.96 tok/s | +10.8% |
| Incident reasoning, median of 3 seeded samples | 31.92 tok/s | 41.26 tok/s | +29.3% |
| Reference C1, one paired pass | 31.45 tok/s | 40.23 tok/s | +27.9% |
| Reference C2 combined, one paired pass | 45.87 tok/s | 61.06 tok/s | +33.1% |
| Fully uncached 32K prefill, one paired pass | 1137.76 tok/s | 1135.35 tok/s | -0.2% |
| Two ~32K coding requests, slower completion | 74.59 s | 69.18 s | -7.3% time |

Sampled workloads used seeds 11/23/47, 512 output tokens, temperature 1 and top_p
.95. Thinking was on only for incident reasoning. Reference decode used 400
outputs, temperature 0/top_p 1; long concurrent coding used 512 outputs/request.
Poetry acceptance was 23.26–24.41% for the candidate: **50 tok/s at below 25%
acceptance was NOT achieved**. C2 is aggregate throughput, not per-stream speed.
All nine paired full response hashes differed, so this is not identical-output
timing or proof of unchanged answer quality. See `benchmark-summary.json`.

Two extra candidate reference repetitions gave C1 39.22–39.81 tok/s, C2 combined
58.92–60.00 tok/s, and uncached prefill 1138.90–1142.77 tok/s. No observed
regression in these bounded workloads is not a universal no-regression claim.
The matched stock reference had one repetition, not three. Near-600K prefill and
long production burn-in were not repeated. The pre-existing nonfatal p-only
sampler-cache coverage warning remains. Both long concurrent streams completed;
no stream errors/preemptions/OOM kills occurred in the checks.

## Numerical and safety evidence

Fixed specialization was bit-exact with upstream wide/wide on 96 synthetic,
24 actual-input and six dense fixtures, including graph repeats and 48 edge/
recovery checks. It is **not bit-exact with installed stock**: the actual-input
screen retained 35 strict rtol=atol=1e-3 failing elements out of 645,120, max
error/stock peak 0.1123%. Six independent dense checks passed a 0.3%-of-peak
screen (max fixed peak error 0.0917%, normalized RMS 0.0794%).

Four bounded sanitizer passes covered K2/three-row and K3/eight-row memcheck and
racecheck, the first 18 matching launches per pass: zero reported errors/hazards.
This does not mean every later mutation was instrumented. Actual captured rows
were subset/tiled to test other sizes, not new live C2 captures.

113 CPU stub checks and 54 actual Torch/CUDA integration cases passed. The latter
used unnormalized synthetic weight scales and retained accumulated strict failed
element counts of 6,124,455 raw / 3,912,190 post-BF16 across repeated comparisons;
they passed only the documented peak-normalized screen, not strict parity. Full
serving startup, three sampled workload suites, concurrency/prefill checks and a
short correct API answer also passed. Broad task-quality evaluation remains open.

## Build and deliberate activation

`build.sh` takes an existing ExLlamaV3 checkout containing the pinned commit and
an **empty** output directory. It archives the pinned source rather than copying
uncommitted upstream changes. It never downloads or touches runtime state:

```sh
bash experimental/fixed-cooperative-moe/build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY
```

Use the tested CUDA 13 ARM64 recipe image, preferably in a CPU-only container
limited to two CPUs/2 GiB, with the output directory mounted at `/work` to retain
the tested compiler source paths. The image ID in the original test was
`sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb`.
The build uses the original SM121a/O3/fast-math/lineinfo flags. No GPU build/test
is being requested on an occupied production server by these instructions.

The adapter deliberately pins the validated binary SHA256
`a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78` and source
SHA256 `b8688b9b9c88fbca58315e9bb632f6a241b937904534a03cbff1360fbf6980fe`.
Different paths/toolchains may produce a different binary hash. Do not bypass
the check: review and repeat the native/integration gates before repinning.
No binaries, model weights, private `.env` files or host addresses are committed.

`prepare_profile.py` verifies the original overlay, adapter and binary hashes,
then exclusively creates an opt-in copy of the stock overlay. It never installs
the result, updates defaults or restarts services. Stage the exact binary and
adapter on **both** nodes in a cache subdirectory mounted into the containers,
and use its container path for `--runtime-directory`:

```sh
python3 experimental/fixed-cooperative-moe/prepare_profile.py \
  --stock overlay/exl3.py --artifacts VERIFIED_ARTIFACT_DIRECTORY \
  --runtime-directory /root/.cache/vllm/VERIFIED_SUBDIRECTORY \
  --output NEW_OPT_IN_OVERLAY.py
```

Only after an approved maintenance window and verification, select that output
using `EXL3_OVERLAY_HOST` for the normal launcher. Keep a copy of the old `.env`.
Remove the override and restore previous settings to return to stock at the next
approved restart. The launcher must keep the serialized-stream prerequisites.
Never select an unvalidated rebuild for an active user workload.

## Review/testing status

```sh
python3 experimental/fixed-cooperative-moe/test_goal50_coop_dispatch_cpu.py --fixed
```

The CUDA gate requires `GOAL50_MAINTENANCE_TEST=1`, the selected overlay, actual
Torch/vLLM and the recipe's `test_exl3_overlay.py` importable at `/opt/dsv41`.
Run it only with sufficient free GPU memory in an approved maintenance window.
The packaging does not rerun or claim to replace the original GPU evidence.

Before an upstream merge, review numerical tolerances/quality coverage, expand
hardware/checkpoint coverage, decide how to integrate build/artifact distribution,
and rerun from the final package in a clean maintenance environment. This branch
is a source/evidence checkpoint for that review, not an automatic default rollout.
