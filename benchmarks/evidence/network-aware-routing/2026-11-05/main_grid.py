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
sweep = json.loads((root / "saturation_sweep.json").read_text())
cap = {
    (mode, hit, output): max(
        z["throughput_rps"]
        for z in sweep
        if (z["mode"], z["hit"], z["output"]) == (mode, hit, output)
    )
    for mode in ("destination_missing", "full_prompt")
    for hit in (0, 50, 90)
    for output in ("short", "medium", "mixed")
}
policies = [
    ("independent", 0, None),
    ("w0", 0, 1),
    ("w025", 0.25, 1),
    ("w1", 1, 1),
    ("w4", 4, 1),
    ("w1000", 1000, 1),
]
results_path = root / "main_grid_results.jsonl"
seen = set()
if results_path.exists():
    for line in results_path.read_text().splitlines():
        seen.add(json.loads(line)["tag"])
with results_path.open("a") as result_file:
    for mode in ("destination_missing", "full_prompt"):
        for hit, shared in ((0, 0), (50, 16), (90, 29)):
            for output in ("short", "medium", "mixed"):
                for load in (30, 50, 70, 90):
                    rate = cap[(mode, hit, output)] * load / 100
                    for seed in (11, 22, 33, 44, 55):
                        base = f"main_{mode}_h{hit}_{output}_l{load}_s{seed}"
                        trace = root / f"{base}.jsonl"
                        if not trace.exists():
                            rng = random.Random(seed)
                            t = 0.0
                            lengths = (
                                [64] * 300
                                if output == "short"
                                else [400] * 300
                                if output == "medium"
                                else [64] * 150 + [2000] * 150
                            )
                            if output == "mixed":
                                rng.shuffle(lengths)
                            with trace.open("w") as out:
                                for i, nout in enumerate(lengths):
                                    t += rng.expovariate(rate / 1000)
                                    ids = list(range(shared)) + list(
                                        range(
                                            1000 + i * (32 - shared),
                                            1000 + (i + 1) * (32 - shared),
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
                        for policy, weight, nics in policies:
                            tag = f"{base}_{policy}"
                            if tag in seen:
                                continue
                            engine = {
                                "backend": "vllm",
                                "num_gpu_blocks": 519,
                                "max_num_batched_tokens": 131072,
                                "max_num_seqs": 32,
                                "kv_transfer_bandwidth": 25.0,
                                "kv_transfer_bytes_per_token": 262144,
                                "kv_transfer_timing_mode": mode,
                            }
                            if nics is not None:
                                engine["kv_transfer_workers_per_nic"] = nics
                            cfg = {
                                "trace": trace.name,
                                "out_dir": f"main_outputs/{tag}",
                                "timing_table": "two_input_timing.json",
                                "block_size": 512,
                                "num_prefill_workers": 8,
                                "num_decode_workers": 4,
                                "engine": engine,
                                "router": {
                                    "transfer_load_weight": weight,
                                    "transfer_decision_log": str(
                                        root / "main_decisions" / f"{tag}.jsonl"
                                    ),
                                },
                            }
                            (root / "main_decisions").mkdir(exist_ok=True)
                            cfg_path = root / f"{tag}.json"
                            cfg_path.write_text(
                                json.dumps(cfg, separators=(",", ":")) + "\n"
                            )
                            started = time.perf_counter()
                            proc = subprocess.run(
                                [str(binary), str(cfg_path)],
                                capture_output=True,
                                text=True,
                            )
                            if proc.returncode:
                                raise RuntimeError(f"{tag}: {proc.stderr}")
                            outdir = root / cfg["out_dir"]
                            summary = json.loads((outdir / "summary.json").read_text())
                            rows = [
                                json.loads(line)
                                for line in (outdir / "requests.jsonl")
                                .read_text()
                                .splitlines()
                            ]
                            if (
                                len(rows) != 300
                                or any(
                                    summary[k]["count"] != 300
                                    for k in ("ttft_ms", "e2e_ms", "handoff_ms")
                                )
                                or any(
                                    x.get("terminal_status") != "completed"
                                    for x in rows
                                )
                            ):
                                raise RuntimeError(
                                    f"{tag}: incomplete replay or missing timings"
                                )
                            result = {
                                "tag": tag,
                                "mode": mode,
                                "hit": hit,
                                "output": output,
                                "load": load,
                                "seed": seed,
                                "policy": policy,
                                "weight": weight,
                                "offered_rps": rate,
                                "saturation_cap_rps": cap[(mode, hit, output)],
                                "wall_s": time.perf_counter() - started,
                                "requests": 300,
                                "realized_prefill_hit": sum(
                                    x.get("reused_input_tokens") or 0 for x in rows
                                )
                                / (300 * 16384),
                                "mean_missing_tokens": sum(
                                    x.get("destination_missing_tokens") or 0
                                    for x in rows
                                )
                                / 300,
                                "ttft_p50": summary["ttft_ms"]["p50"],
                                "ttft_p99": summary["ttft_ms"]["p99"],
                                "handoff_p99": summary["handoff_ms"]["p99"],
                                "itl_p99": summary["itl_ms"]["p99"],
                                "e2e_p99": summary["e2e_ms"]["p99"],
                            }
                            result_file.write(
                                json.dumps(result, separators=(",", ":")) + "\n"
                            )
                            result_file.flush()
                            seen.add(tag)
                        print("completed", base, flush=True)
print("total", len(seen))
