# Nominal A100 40GB replay after decode recomputation fix

## Why this rerun was necessary

AISimulate `edcbc713ed65bf1c66a2fb3969523ee3cf77fc1a` changes vLLM scheduler accounting: when a preempted decode request is readmitted and must recompute prompt tokens, that work now receives prefill timing. The earlier experiment used `33bc0318c8c5614dd466180197ba7af672b0eac9`, which charged zero prefill time on decode workers. The old results remain in the parent directory for comparison.

All **378** cases were rerun against the new AISimulate revision and Dynamo `011665977c21f5b91eb5b0d5d748364fa1155deb`. The same nominal 40GB systems overlay, 8P4D topology, timing provider, four bandwidth/timing sweeps, workload parameters, trace seeds and routing weights were used. Every trace was verified byte-identical to its earlier counterpart; each run completed 1,000 requests. The four `results_*.jsonl` files contain all new per-case summaries and source revisions. The corresponding full configurations, traces, timing records and decision logs are under `/private/tmp/hpn_edcbc713_20261106/` on the experiment machine.

The original system limitation still applies: operation profiles are from A100 80GB; the 40GB system overlay is nominal, the two Slingshot NICs are represented by one logical link, and these runs are **not validated predictions for ASPIRE 2A**. The decode-speed ratio sweep is a heuristic sensitivity test, not a calibrated 40GB timing profile.

## Direct comparison with the old revision

Of 378 per-run summaries, **74 changed**. Every changed case had decode readmissions in the earlier run. No no-cache (`h0`) summary changed. Two runs with readmissions had identical summaries. Thus the code change's observed effect is confined to runs that exercise the corrected preemption path, but those runs can have substantially different queueing and TTFT.

The following TTFT p99 gains compare each routing weight to weight 0 on the same trace seed. Positive means routing reduced TTFT p99. Values are percentage-point gains, summarized across seeds; these are **paired** policy comparisons within each source revision.

| Workload and setting | Weight | Old mean / median | New mean / median | New range |
|---|---:|---:|---:|---:|
| Mixed h90, 1.98 req/s, 25 GB/s, destination-missing (10 seeds) | 1 | +14.40 / −0.04% | +13.88 / −0.04% | −0.56 to +75.59% |
| Same | 4 | +14.48 / +0.05% | +13.78 / +0.05% | −0.38 to +75.66% |
| Mixed h90, 1.98 req/s, 16.7 GB/s, full-prompt (5 seeds) | 1 | +15.59 / +0.04% | +0.57 / +0.04% | −0.01 to +2.64% |
| Same | 4 | +15.47 / +0.04% | −0.11 / +0.04% | −0.89 to +0.24% |
| Mixed h90, 1.98 req/s, 12.5 GB/s, full-prompt (5 seeds) | 1 | −96.62 / +0.05% | −114.25 / +0.05% | −571.70 to +0.31% |
| Mixed h90, 1.98 req/s, decode-speed ratio 0.763 (5 seeds) | 1 | +31.62 / +43.79% | +33.39 / +30.17% | +19.37 to +49.15% |
| Same | 4 | +13.11 / +18.42% | +44.95 / +37.61% | +32.13 to +64.37% |

The decode-speed h90 case is especially sensitive: baseline decode readmissions increased from **350 to 472** across five seeds after charging recomputation, while weight 1 had 235 → 408 and weight 4 had 196 → 317. Its policy ranking by median TTFT p99 gain changed: weight 4 now exceeds weight 1 in this particular heuristic sweep. The absolute baseline TTFT p99 median also moved from 29.3 s to 33.7 s. These large latencies indicate a congested simulated regime, not a realistic measured service-level result.

By contrast, no-cache results—including the 25 GB/s, 1.62 req/s mixed run with weight-1 median TTFT p99 gain of **+3.15%** across ten seeds—were unchanged. The 25 GB/s high-cache run still has median TTFT p99 gain near zero despite lower transfer p99. The original qualitative conclusion therefore remains: transfer-aware routing can save transfer time, but it does not reliably improve TTFT across workloads and seeds. The recomputation fix makes the old high-cache quantitative values obsolete.

## Reproduction and checks

`rerun_parallel.py` lists the 378-case matrix. `run_prebuilt.sh` records the exact binary/source environment used to execute cases after `run_hpn.sh` rebuilt AISimulate and Dynamo; these scripts contain local absolute paths and are execution snapshots, not portable setup tools. The parent directory's `followups_nominal_40gb.py` contains trace generation and per-case configuration. To summarize one new sweep, run `python3 ../../2026-11-06/summarize.py results_25gbps.jsonl` from this directory. The rerun completed with 249 nominal 25 GB/s cases, 45 cases each at 16.7 and 12.5 GB/s, and 39 decode-speed-ratio cases. The changed-summary counts were 42, 9, 8 and 15, respectively. All 378 new results record AISimulate `edcbc713` and Dynamo `011665977c`.
