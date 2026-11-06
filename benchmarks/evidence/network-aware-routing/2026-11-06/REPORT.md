# Follow-up live-AIS replay experiments (2026-11-06)

## Setup and scope

- Dynamo `1a8336401ae531e897fcc8930c91ab568db0be0b` and AISimulate `33bc0318c8c5614dd466180197ba7af672b0eac9`, recorded in every `timing.json`.
- Ran the merged `hpn_calibration/run_hpn.sh` with live AIS timing and its template: 8 prefill, 4 decode workers; 439/534 role-specific blocks; 256 sequences; 40,960/2,048 batched-token limits; 262,144 transfer bytes/token; 25 GB/s NIC; 512-token blocks.
- Each run had 1,000 Poisson arrivals, 16K input tokens, 20 randomly selected prompt families, and either 0 or 29 shared 512-token blocks. Mixed output was a shuffled 50/50 split of 64 and 2,000 tokens. Traces were seeded and regenerated identically for each policy and transfer mode. The independent generator is [followups.py](followups.py); the questions it tests are the handoff's stated next steps.
- The team's `results/phase2` scripts were not present in this checkout, so this generator implements the handoff's stated design independently. Results are not exact repeats of the team's preliminary runs.
- All 251 local replay outputs contained 1,000 completed requests; 249 are planned cases and two are harmless repeated 1.0-req/s saturation cases created before the rate was added to output tags. The committed [results.jsonl](results.jsonl) contains the 249 planned summaries. All timing provenance records match the revisions above.
- Positive TTFT gain below means `(w=0 p99 - w>0 p99) / w=0 p99`, paired by arrival seed. Figures are descriptive; no confidence interval is claimed.

## 1. More seeds at 90% mixed-output load

Ten new paired seeds per hit condition, with `destination_missing`. Rates use the team's preliminary saturation values: 1.62 req/s for zero shared blocks and 1.98 req/s for 29 shared blocks.

| Shared blocks | Weight | Mean paired TTFT p99 gain | Median | Range | Positive / negative seeds | Decode readmissions, w=0 / policy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | -1.03% | +3.23% | -26.61% to +7.59% | 7 / 3 | 0 / 0 |
| 0 | 4 | -1.27% | +6.79% | -40.82% to +8.46% | 7 / 3 | 0 / 0 |
| 29 | 1 | -5.18% | -0.03% | -90.56% to +39.45% | 3 / 6 (one tie) | 21 / 13 |
| 29 | 4 | +4.02% | +0.08% | -0.25% to +39.45% | 6 / 3 (one tie) | 21 / 18 |

At zero hit, mean transfer p99 fell from 337 ms at w=0 to 241 ms (w=1) and 195 ms (w=4), but mean reserve-wait p99 rose from 1,422 ms to 1,650/1,624 ms. At high hit, mean transfer p99 fell from 228 ms to about 198 ms. Some high-hit seeds had readmissions and several-second TTFT tails. The policy changes transfer time more consistently than TTFT p99.

## 2. Output length 64 only

The single-seed w=0 rate sweep spanned 1–8 req/s for zero hit and 1–25 req/s for high hit. Under the handoff's first-quarter versus last-quarter median TTFT rule, zero hit remained stable through 5 req/s and became unstable at 6 req/s. The high-hit sweep did not yield a clean saturation bracket: the early quarter already had large queueing at some rates, so no saturation percentage is assigned to its policy comparisons.

| Shared blocks | Rate (req/s) | Seeds | Weight | Mean paired TTFT p99 gain | Range | Mean transfer p99, w=0 → policy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.5 | 3 | 1 | +5.66% | +5.08% to +6.73% | 281 → 173 ms |
| 0 | 2.0 | 3 | 1 | +4.97% | 0% to +7.68% | 301 → 173 ms |
| 0 | 4.0 | 3 | 1 | +0.58% | -0.75% to +2.80% | 347 → 324 ms |
| 29 | 5.0 | 3 | 1 | +0.25% | +0.07% to +0.44% | 298 → 185 ms |
| 29 | 10.0 | 3 | 1 | +2.26% | +0.80% to +3.55% | 540 → 355 ms |
| 29 | 10.0 | 3 | 4 | +2.93% | +1.53% to +4.99% | 540 → 253 ms |

At 1–2.5 req/s in the high-hit short-output cases, TTFT p99 was effectively unchanged even when transfer p99 declined. All short-output policy comparisons had zero decode readmissions.

## 3. Matched `destination_missing` versus `full_prompt`

At zero hit, the two modes produced identical outputs across three seeds at 1.26 req/s, as expected because no destination had a shared prefix. Both modes had mean paired TTFT p99 gains of +6.07% for w=1 and +6.47% for w=4.

At 29 shared blocks and 1.54 req/s, all three seeds had no readmissions. Under w=0, mean transfer p99 was 201 ms with `destination_missing` and 334 ms with `full_prompt`, while TTFT p99 was about 1,784–1,786 ms. Mean paired TTFT p99 gain with w=1 was +0.11% and +0.06%, respectively. The extra transfer cost did not translate into a meaningful TTFT p99 gain at this rate.

At 29 shared blocks and 1.98 req/s, five new seeds gave these paired gains:

| Mode | Weight | Mean paired TTFT p99 gain | Median | Range | Positive / negative seeds | Mean transfer p99, w=0 → policy | Decode readmissions, w=0 / policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `destination_missing` | 1 | +9.33% | +0.18% | -4.52% to +50.65% | 3 / 2 | 231 → 200 ms | 12 / 7 |
| `destination_missing` | 4 | +14.17% | +0.26% | +0.18% to +69.79% | 5 / 0 | 231 → 199 ms | 12 / 10 |
| `full_prompt` | 1 | +2.71% | +0.15% | -0.22% to +13.07% | 4 / 1 | 384 → 306 ms | 10 / 12 |
| `full_prompt` | 4 | +12.39% | +0.06% | -0.21% to +61.53% | 4 / 1 | 384 → 221 ms | 10 / 26 |

The large means arise mainly from one seed with seconds of queueing. Full-prompt w=4 also increased readmissions substantially (10 to 26 across five seeds). These results do not establish a reliable full-prompt TTFT advantage.

## Limits and next use

- Runs are simulations, not target-machine measurements. AIS timing uses the 80GB A100 profile while capacity approximates the 40GB part.
- The trace generator was recreated from the handoff description; use the team's original `results/phase2/common.py` if an exact comparison with its published preliminary numbers is needed.
- Initial cold requests are included in p99. The high-hit 64-token saturation estimate needs a more robust warmup or steady-state protocol.
- Queueing and readmissions dominate some p99 outcomes. Compare transfer and reserve-wait distributions, and report seed ranges whenever interpreting an apparent gain.
- Full run configurations, traces, logs, request records, decision logs, and timing provenance remain in `/private/tmp/hpn_live_followups_20261106` on the experiment machine. The committed result summaries carry source revisions but omit those large per-request artifacts.

## Reproduction

From a checkout with sibling `dynamo` and `aisimulate` directories, the merged live-AIS runner's prerequisites, and a writable output directory:

```sh
cd dynamo
for phase in highload short_saturation short_saturation_extended short short_nearknee modes modes_highload; do
  python3 benchmarks/evidence/network-aware-routing/2026-11-06/followups.py "$phase" --out /private/tmp/hpn_live_followups_replay
done
```

The earlier 2026-11-05 table-based 2,160-run grid is retained in branch history at commit `06e14bb75c` as preliminary evidence. Its harness changes and evidence files are removed from this PR's final diff because they use the pre-merge calibration configuration.
