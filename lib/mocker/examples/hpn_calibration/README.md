<!--
SPDX-License-Identifier: Apache-2.0
-->

# Calibrated hpn_replay

This directory runs the `hpn_replay` example with timing from the AISimulate
op-level performance model. The model is Qwen3-32B on 4× A100 40GB (TP=4) with
vLLM, which approximates the ASPIRE 2A node used by the HPN project.
`template.json` holds the calibrated values.

Timing comes from `ForwardPassPerfModel::best_available` with `op_level` and
`deny`, called once per role at startup. Building that model needs an embedded
Python interpreter, so this path uses the `ais-live` cargo feature and a Python
environment built from AISimulate's own `uv.lock`.

## Prerequisites

- `rustup`. The dynamo checkout pins Rust 1.96.1. Install it with
  `rustup toolchain install 1.96.1`.
- [`uv`](https://docs.astral.sh/uv/). It downloads its own Python 3.13, so no
  system Python is needed.
- `protoc` (`brew install protobuf` or `apt install protobuf-compiler`).
- Sibling checkouts on matching branches:

  ```text
  <root>/dynamo       HPN_Final (or a branch off it)
  <root>/aisimulate   HPN_Final
  ```

## Run

```bash
lib/mocker/examples/hpn_calibration/run_hpn.sh lib/mocker/examples/hpn_calibration/template.json
```

The first run builds `<root>/.hpn-venv` with `uv sync --frozen` against
`aisimulate/python/aisimulate/uv.lock`. On an M-series Mac this took about 5
minutes, including the Rust build. Later runs reuse the venv. `run_hpn.sh` records the
aisimulate commit in `<venv>.source` and rebuilds when it changes, or on every
run while the aisimulate checkout has uncommitted changes, because uv cannot
detect Rust-only edits to the editable package. Set `HPN_VENV` to put the venv
somewhere else. The script refuses to sync into an existing directory it did
not create.

The template replays one 16K-token request. The run writes `run/summary.json`,
`run/requests.jsonl` and `run/timing.json`, which records each role's resolved
perf-model config and provenance. Expect a prefill of 1578.25 ms
(`source_held_ms - prefill_admit_ms`) and a 171.80 ms handoff. To run your own
workload, copy the template and change `trace`, `out_dir` and the worker counts.

## Check the environment

Run this to confirm the venv's perf model matches the handoff numbers:

```bash
<root>/.hpn-venv/bin/python -c '
from aisimulate_core.sdk import ForwardPassPerfModelConfig as C, RustForwardPassPerfModel as M
m = M.best_available(C(model="Qwen/Qwen3-32B", system="a100_sxm", backend="vllm",
    backend_version="current", worker_type="prefill", tp=4,
    estimation_mode="op_level", fallback_policy="deny"))
print(m.estimate_forward_pass_time_ms({"scheduled_requests": {
    "num_prefill_requests": 1, "sum_prefill_tokens": 32768, "sum_prefill_kv_tokens": 0}}))'
```

It should print about `3949.74`.

## Calibrated values

| Field | Prefill | Decode | Source |
|---|---|---|---|
| `num_gpu_blocks` (512 tokens) | 439 | 534 | `estimate_kv_cache`: 40 GiB, 90% of total, TP=4, the role's batch limits |
| `max_num_batched_tokens` | 40960 | 2048 | Prefill: one-shot prompts up to the context limit. Decode: vLLM 0.31.0 API-server default for A100 |
| `max_num_seqs` | 256 | 256 | vLLM 0.31.0 API-server default for A100 |
| `kv_transfer_bytes_per_token` | 262144 | 262144 | 65,536 per TP shard × 4 |
| `kv_transfer_bandwidth` | 25 GB/s | 25 GB/s | 2× 100G Slingshot treated as one link |

Prefill holds fewer blocks than decode because its 40960-token batch needs a
larger activation reserve.

## Known distortions

- `a100_sxm.yaml` describes the 80 GB, 8-GPU part. Capacity is overridden to
  40 GiB, but timing uses the 80 GB part's memory bandwidth.
- The timing tables come from vLLM 0.14.0, the only version available for
  `a100_sxm`.
- The mock scheduler gives the timing model the batch's mean new tokens, cached
  tokens and context length, not each request's own values.
- When a decode worker preempts a request and recomputes its prompt, the mock
  charges no time for the recompute. `summary.json` reports preemptions as
  `readmissions`.
- Only macOS arm64 has been tested. Linux should work because uv's managed
  Python ships a shared libpython, but it is untested.
