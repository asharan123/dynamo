# A100 40GB nominal hardware rerun (6 November 2026)

**Historical run:** These results use AISimulate `33bc0318`. The decode-preemption timing fix in `edcbc713` changes some high-cache results. Use the [updated rerun](edcbc713-rerun/REPORT.md) for current conclusions; this report and its original result files remain as before-and-after evidence.

## Scope and provenance

This is a **simulation sensitivity study**, not a measurement on NSCC ASPIRE 2A. It reruns the network-aware decode routing follow-up experiments with the handoff's target layout: 8 prefill nodes and 4 decode nodes, one TP=4 worker per four-GPU node, Qwen3-32B, 16,384-token prompts, 262,144 KV bytes per token, and 439 prefill / 534 decode GPU blocks. The model has 256 maximum sequences per worker and maximum batched tokens of 40,960 (prefill) / 2,048 (decode). A run has 1,000 Poisson-arrival requests. Each policy comparison uses the same trace seed.

AISimulate source: `33bc0318c8c5614dd466180197ba7af672b0eac9`. Dynamo source: `9defc4e36925a1f10749e9ca5139c6f6a998ce82`. All 378 retained runs completed with 1,000 requests: 249 nominal 25 GB/s cases, 45 cases at 16.7 GB/s, 45 cases at 12.5 GB/s, and 39 decode timing sensitivity cases. The four `results_*.jsonl` files contain the per-run summary and source revisions. Full traces, configurations, decision logs and replay logs are in the matching `/private/tmp/hpn_nominal_40*20261106` directories on the experiment machine.

The isolated `a100_sxm` systems overlay used in all runs has nominal 40 GiB GPU capacity, 1,555 GB/s HBM bandwidth, and four GPUs per node; its three value changes relative to the checked-in `a100_sxm.yaml` are captured in `a100_sxm_nominal_40gb.yaml`. The 1,555 GB/s number is [NVIDIA's published peak for A100 40GB SXM](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/), while the checked-in 80GB SXM configuration uses [NVIDIA's 2,039 GB/s peak](https://developer.nvidia.com/blog/supercharging-worlds-fastest-ai-supercomputing-platform-on-hgx-a100-80gb-gpus/). Neither number is a measured application bandwidth. The overlay was selected using `systems_paths`, so the checked-in AISimulate profile was not changed.

**Crucial timing caveat:** the overlay still points to the checked-in `data/a100_sxm` op profiles, which were obtained for A100 80GB. The replay uses `estimation_mode=op_level`; changing nominal `mem_bw` does not replace those operation timings with A100 40GB measurements. The NIC model is one fused full-duplex logical link, not two independently scheduled Slingshot NICs or a dragonfly fabric. Thus these runs are closer in capacity and nominal link assumptions to the target, but they do **not** establish actual ASPIRE 2A latency or throughput.

## Experiment design

- Policies: baseline `transfer_load_weight=0`, then weights 1, 4, and, for the full 25 GB/s high-load sweep, 1,000. Positive paired TTFT p99 gain means `(baseline p99 - policy p99) / baseline p99 × 100`.
- Mixed output: half the requests generate 64 tokens, half 2,000. Short output: all generate 64. `h0` has no shared prefix blocks; `h90` shares 29 of 32 hash IDs within each of 20 prompt families.
- The 25 GB/s study repeats the earlier high-load, short-output, cache-hit, transfer-mode, and load-knee experiments using the nominal 40GB overlay. The 16.7 and 12.5 GB/s sweeps use five paired seeds at the high-load mixed settings. These are effective-bandwidth sensitivity points for uneven TP-rank-to-NIC placement, not measurements of the Slingshot network.
- A separate representative sweep sets `decode_speedup_ratio=1555/2039≈0.763` at 25 GB/s. This is a heuristic timing perturbation to expose sensitivity to slower 40GB memory behavior, **not** a calibrated A100 40GB profile. It is inappropriate to present the resulting times as predictions for the machine.

## Main results

At 25 GB/s, the following are paired-seed results under `destination_missing` transfer timing. Transfer p99 values are averages of each run's transfer p99, in milliseconds. TTFT gains are percentage points relative to weight 0.

| Workload | Seeds | Weight | TTFT p99 gain mean / median | Gain range | Transfer p99 baseline → policy | Decode readmissions baseline → policy |
|---|---:|---:|---:|---:|---:|---:|
| Mixed, h0, 1.62 req/s | 10 | 1 | +3.78 / +3.15% | −9.57 to +30.59% | 331.6 → 254.4 | 0 → 0 |
| Mixed, h0, 1.62 req/s | 10 | 4 | −2.20 / +3.46% | −46.45 to +7.27% | 331.6 → 195.5 | 0 → 0 |
| Mixed, h90, 1.98 req/s | 10 | 1 | +14.40 / −0.04% | −0.56 to +62.40% | 237.8 → 198.6 | 43 → 18 |
| Mixed, h90, 1.98 req/s | 10 | 4 | +14.48 / +0.05% | −0.38 to +62.50% | 237.8 → 198.4 | 43 → 10 |
| Short, h0, 1.5 req/s | 3 | 1 | +5.30 / +5.10% | +4.14 to +6.65% | 279.1 → 172.6 | 0 → 0 |
| Short, h90, 10 req/s | 3 | 4 | +2.78 / +2.68% | +1.29 to +4.37% | 628.2 → 313.3 | 0 → 0 |

The h90 mixed-workload means are driven by large queueing gains in a minority of seeds; their medians are near zero. Increasing the routing weight is not consistently better. The full-prompt transfer mode is particularly fragile near saturation: at 1.98 req/s, five seeds yielded mean TTFT p99 gains of −75.46% (weight 1) and −12.97% (weight 4), while medians were −0.03% and +0.24%, respectively. The worst weight-1 paired gain was −377.78%. The minimum and maximum across seeds should accompany any quoted mean.

## NIC and decode timing sensitivity

The NIC table uses the **same five seeds 101–105** for each bandwidth, h0 mixed output, 1.62 req/s, and `destination_missing` mode. All values are simulation outputs. The apparent increase in benefit as effective bandwidth falls is conditional on this load and policy; individual seeds still regress.

| Effective logical NIC bandwidth | Weight | Paired TTFT p99 gain median (range) | Transfer p99 baseline → policy |
|---:|---:|---:|---:|
| 25 GB/s | 1 | +3.57% (−8.94 to +5.61%) | 330.9 → 250.8 ms |
| 25 GB/s | 4 | +6.33% (−5.33 to +7.00%) | 330.9 → 195.4 ms |
| 16.7 GB/s | 1 | +4.26% (−4.18 to +5.85%) | 514.1 → 411.0 ms |
| 16.7 GB/s | 4 | +10.45% (−18.54 to +12.00%) | 514.1 → 281.8 ms |
| 12.5 GB/s | 1 | +4.80% (−9.79 to +9.17%) | 740.3 → 580.8 ms |
| 12.5 GB/s | 4 | +14.89% (−9.84 to +18.14%) | 740.3 → 394.1 ms |

For h90 mixed output at 1.98 req/s, TTFT p99 median gains stayed small at all three effective bandwidths (between −0.18% and +0.21% for weights 1 and 4), despite lower transfer p99. A full-prompt, 12.5 GB/s case had one severe weight-1 queueing regression (−483.55% paired TTFT p99 gain), so a smaller transfer time does not imply a safer overall tail.

The decode-speed perturbation changes the conclusion at high load. At 25 GB/s, h0 mixed output and 1.62 req/s, five paired seeds gave weight-1 median TTFT p99 gain **−0.08%** and weight-4 **−6.90%**. In h90 mixed output at 1.98 req/s, medians were +43.79% and +18.42%, respectively, but the ranges were −15.80 to +61.31% and −62.22 to +56.65%. Decode readmissions in the five h90 baseline runs rose to 350. In the short h0 1.5 req/s case, three paired seeds still gave a +4.36% median gain for weight 1. This perturbation demonstrates sensitivity; it is not evidence that one weight will perform best on A100 40GB hardware.

## Interpretation and next measurement

The rerun supports a narrow conclusion: under the nominal 40GB capacity and logical-NIC assumptions, transfer-aware routing often reduces transfer p99, while end-to-end TTFT p99 gains vary by cache reuse, output length, offered load, transfer accounting mode, and seed. The current evidence cannot rank the policies for the real ASPIRE 2A cluster. To make a hardware-matched claim, collect or import A100 40GB TP=4 operation/serving timings and Slingshot transfer behavior (including rank-to-NIC placement), then rerun identical paired traces. The simulated 12-node 8P4D slice also does not capture all 64 nodes or network-wide contention.

## Reproduction notes

`followups_nominal_40gb.py` is a snapshot of the run driver; its absolute paths identify the machine and the isolated systems overlay used here. Each retained result carries its source revisions. The summary calculation is `python3 ../2026-11-06/summarize.py results_25gbps.jsonl` (or another `results_*.jsonl` file). The driver sets `HPN_NIC_GBPS` to 25 by default; the NIC sweeps set it to 16.7 or 12.5. The decode timing sweep sets `HPN_DECODE_RATIO=0.7626287395782246`. The full per-case `experiment.json`, `trace.jsonl`, `run/summary.json`, `run/timing.json`, `decisions.jsonl`, and `runner.log` are retained in `/private/tmp/hpn_nominal_40*20261106` and can be copied into an archive if long-term raw replay evidence is needed.
