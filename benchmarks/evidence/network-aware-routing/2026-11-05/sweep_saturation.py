import json
import os
import random
import subprocess
import time
from pathlib import Path

root = Path(os.environ.get("HPN_EXP_ROOT", "/private/tmp/hpn_experiments_20261105"))
binary = Path(
    os.environ.get(
        "HPN_REPLAY_BIN",
        str(Path(__file__).resolve().parents[4] / "target/release/examples/hpn_replay"),
    )
)
results = []
for mode in ("destination_missing", "full_prompt"):
    for hit, shared in ((0, 0), (50, 16), (90, 29)):
        for output in ("short", "medium", "mixed"):
            for rate in (1, 2, 4, 8, 16, 32, 64):
                tag = f"sweep_{mode}_{hit}_{output}_{rate}"
                rng = random.Random(101)
                t = 0.0
                lengths = (
                    [64] * 500
                    if output == "short"
                    else [400] * 500
                    if output == "medium"
                    else [64] * 250 + [2000] * 250
                )
                if output == "mixed":
                    rng.shuffle(lengths)
                with (root / f"{tag}.jsonl").open("w") as out:
                    for i, nout in enumerate(lengths):
                        t += rng.expovariate(rate / 1000)
                        ids = list(range(shared)) + list(
                            range(
                                1000 + i * (32 - shared), 1000 + (i + 1) * (32 - shared)
                            )
                        )
                        out.write(
                            json.dumps(
                                {
                                    "timestamp": t,
                                    "input_length": 16384,
                                    "output_length": nout,
                                    "hash_ids": ids,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                engine = {
                    "backend": "vllm",
                    "num_gpu_blocks": 519,
                    "max_num_batched_tokens": 131072,
                    "max_num_seqs": 32,
                    "kv_transfer_bandwidth": 25.0,
                    "kv_transfer_bytes_per_token": 262144,
                    "kv_transfer_timing_mode": mode,
                    "kv_transfer_workers_per_nic": 1,
                }
                cfg = {
                    "trace": f"{tag}.jsonl",
                    "out_dir": tag,
                    "timing_table": "two_input_timing.json",
                    "block_size": 512,
                    "num_prefill_workers": 8,
                    "num_decode_workers": 4,
                    "engine": engine,
                    "router": {"transfer_load_weight": 0},
                }
                (root / f"{tag}.json").write_text(json.dumps(cfg) + "\n")
                start = time.perf_counter()
                p = subprocess.run(
                    [str(binary), str(root / f"{tag}.json")],
                    capture_output=True,
                    text=True,
                )
                if p.returncode:
                    raise RuntimeError(f"{tag}: {p.stderr}")
                summary = json.loads((root / tag / "summary.json").read_text())
                rows = [
                    json.loads(x)
                    for x in (root / tag / "requests.jsonl").read_text().splitlines()
                ]
                assert (
                    len(rows) == 500
                    and summary["ttft_ms"]["count"] == 500
                    and summary["e2e_ms"]["count"] == 500
                )
                span = max(x["terminal_time_ms"] for x in rows) - min(
                    x["arrival_time_ms"] for x in rows
                )
                throughput = 500000 / span
                results.append(
                    {
                        "mode": mode,
                        "hit": hit,
                        "output": output,
                        "offered_rps": rate,
                        "throughput_rps": throughput,
                        "ttft_p99_ms": summary["ttft_ms"]["p99"],
                        "wall_s": time.perf_counter() - start,
                    }
                )
            print("done", mode, hit, output, flush=True)
(root / "saturation_sweep.json").write_text(json.dumps(results, indent=2) + "\n")
for mode in ("destination_missing", "full_prompt"):
    for hit in (0, 50, 90):
        for output in ("short", "medium", "mixed"):
            x = [
                z
                for z in results
                if (z["mode"], z["hit"], z["output"]) == (mode, hit, output)
            ]
            print(
                mode,
                hit,
                output,
                [
                    (
                        z["offered_rps"],
                        round(z["throughput_rps"], 1),
                        round(z["ttft_p99_ms"]),
                    )
                    for z in x
                ],
            )
