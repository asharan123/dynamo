# Network-aware decode routing: local simulation runs

Date: 2026-11-05. This is a **simulation result**, not a measurement on ASPIRE 2A or a calibrated production prediction.

## Revisions and method

- AISimulate replay: `33bc0318c8c5614dd466180197ba7af672b0eac9` (`HPN_Final`).
- Dynamo replay/router: `42cdf3bb9aa22530b58945ecd2d3d4fef5986f7c` (`HPN_Final`).
- FPM query environment: an AISimulate checkout at `4d792a753f9c4bd5d3f497c4051de515bd081153`. Qwen3-32B, `a100_sxm`, vLLM `current` (0.14.0), op-level, TP=4. The queried tables are saved here. AISimulate is Apache-2.0 licensed; these tables are generated numerical outputs of that revision.
- Built `hpn_replay` in release mode and added an optional experiment-only timing-table callback to its example. The callback interpolates prefill timing over batch, new tokens, and cached tokens; decode timing over batch and context. It uses the existing `PerfModel::Ais` callback seam and never calls FPM during replay.
- For the main table, 40 prefill holdout queries within the 16K workload region had median absolute interpolation error 0.73%, p95 1.96%, maximum 2.43%; 40 decode holdouts had median 0.04%, p95 0.10%, maximum 0.13%. The larger table used for sensitivity and real traces was not separately holdout-tested.
- Simulated 8 prefill and 4 decode workers, 519 KV blocks of 512 tokens each, 262,144 total transfer bytes per token, 25 GB/s NIC bandwidth, `max_num_seqs=32`, and `max_num_batched_tokens=131072`. The last two are experiment choices, not validated target-machine settings.
- The saturation sweep used 500 requests for each of 18 workload/mode combinations at seven offered rates. Each main-grid load was 30, 50, 70, or 90% of the highest observed completed rate for that combination. These are finite-trace empirical caps, not SLO thresholds or steady-state capacity proofs.
- The main grid had 432 configurations × five paired arrival seeds = 2,160 runs. Each had 300 open-loop requests; all 648,000 requests completed. Policies were no contention, contention with `w=0`, and contention with `w=0.25`, `1`, `4`, or `1000`. Outcomes were compared to `w=0` within the same trace, transfer mode, and seed.
- The 9 `hpn_network_routing` integration tests and 2 timing-table adapter tests passed. The main runs took 238 seconds of process wall time in aggregate (median 0.087 s/run, p99 0.28 s/run, maximum 0.485 s/run). This workload size does not clear the previously identified asymptotic NIC-model cost risk.

## Main observations

A positive TTFT gain below means `(w=0 p99 - policy p99) / w=0 p99`, paired by seed. Results include cold starts.

| Transfer mode | Shared prefix | Output | Policy | Mean paired p99 TTFT gain across 20 seed/load pairs | Positive / zero / negative pairs |
|---|---:|---|---|---:|---:|
| `destination_missing` | 0% | 64/2000 mix | `w=1` | 5.12% | 20 / 0 / 0 |
| `destination_missing` | 90% | 64/2000 mix | `w=1` | -0.04% | 1 / 14 / 5 |
| `full_prompt` | 90% | 64/2000 mix | `w=1` | 8.98% | 10 / 10 / 0 |

For `full_prompt`, 90% shared prefix, mixed outputs, the mean paired p99 TTFT gain from `w=1` was 7.96%, 14.11%, 11.27%, and 2.60% at the four load levels. Five-seed ranges were broad: at 50% load, 0% to 37.71%. `w=4` often gave a similar result; `w=1000` was generally identical to `w=4` in these scenarios. The corresponding mean ITL p99 change for `w=1` over all 20 pairs was near zero; no consistent high-weight ITL penalty was observed.

The decision logs support the mechanism but are not paired causal decisions once policies diverge. At `full_prompt`, 90% shared prefix, mixed outputs, 50% load, `w=0` selected a busy receive NIC 50 times across 1,500 decisions; `w=1` did so 7 times. At 70% load the counts were 72 versus 4. The selected worker was no longer always the minimum-decode-block worker (`w=1` chose a nonminimum 10 and 15 times at those loads). The reduced contention therefore came with occasional memory-load tradeoffs.

The `destination_missing` high-hit case moves far fewer bytes after the cache warms and showed little benefit in this grid. The 0%-hit mixed-output case still benefited because transient NIC congestion occurred despite low average offered load. These observations are specific to the synthetic hash and arrival construction.

## Sensitivity and recipe traces

Around the `full_prompt`, 90%-prefix, mixed-output, 50%-load region, five seeds were run for each variant at a fixed 1.2 requests/s. Mean paired p99 TTFT gains for `w=1` were 12.41% at the sensitivity baseline, 5.10% at 100 Gbps, 3.13% at 400 Gbps, 1.80% at 6P6D, 14.70% at 4P8D, 7.21% with a 4K/32K prompt mix, and 0% with batches of ten arrivals. Several individual seeds had zero gain. Sensitivity used the larger timing table, so its baseline should be compared only with its own variants.

The recipe traces were filtered to `input_length + output_length <= 40,960`, then given Poisson open-loop arrivals. This retained 1,360 of 1,805 chat rows and 1,522 of 3,541 agent rows. At 1.0 chat requests/s and 0.4 agent requests/s, `w=1` changed p99 TTFT by +0.19% (chat, `destination_missing`), -0.14% (agent, `destination_missing`), +0.22% (chat, `full_prompt`), and +0.97% (agent, `full_prompt`). These are **single-seed realism checks**, not estimates of general trace performance.

## Limits and remaining work

- No target-machine measurement was available. The FPM source describes an A100 80GB configuration while capacity was overridden to approximate 40GB; timing and networking on the hypothetical ASPIRE 2A scenario remain unvalidated. The FPM's vLLM 0.14.0 data also differs from Dynamo's pinned serving version.
- The two-input lookup addresses the one-dimensional NPZ prefill defect, but its table comes from a different AISimulate checkout. The larger table's interpolation error and coverage should be validated separately before publishing a quantitative result.
- The experiment uses oracle NIC and zero-lag KV-index state. It excludes protocol overhead, fabric contention, and live router telemetry delay.
- The saturation sweep's finite-trace peak throughput is a practical load scale, not an SLO. The maximum request rate meeting an SLO cannot be computed until TTFT/ITL thresholds are chosen.
- The built-in NIC scheduler's repeated global rate recalculation and vector plan lookup remain source-level performance risks; the 300-request runs do not establish large-resident-state cost.
- The scripts default to `/private/tmp/hpn_experiments_20261105` and accept environment overrides. Full raw outputs remain there and are not pushed. This tracked bundle contains summary data, timing tables, run scripts, and sample decision logs.

## Reproduce from this bundle

The following commands require the matched AISimulate checkout at the relative path configured in Dynamo's `Cargo.toml`, Rust and `protoc`, and the two recipe traces materialized with Git LFS. Set `HPN_EXP_ROOT` to a writable directory outside the tracked evidence folder.

```sh
cd dynamo
cargo build --release -p dynamo-mocker --features replay-bench --example hpn_replay
export HPN_EXP_ROOT=/private/tmp/hpn_experiments_replay
mkdir -p "$HPN_EXP_ROOT"
cp benchmarks/evidence/network-aware-routing/2026-11-05/two_input_timing*.json "$HPN_EXP_ROOT/"
python3 benchmarks/evidence/network-aware-routing/2026-11-05/sweep_saturation.py
python3 benchmarks/evidence/network-aware-routing/2026-11-05/main_grid.py
python3 benchmarks/evidence/network-aware-routing/2026-11-05/sensitivity.py
python3 benchmarks/evidence/network-aware-routing/2026-11-05/real_traces.py
```

`generate_timing_tables.py` regenerates the committed timing values through the FPM API, using the source revision above. The copied tables were reproduced byte for byte from that environment. The scripts use `HPN_REPLAY_BIN` and `DYNAMO_ROOT` overrides if the binary or repository lives elsewhere.
