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
variants = {
    "baseline": (25.0, 8, 4, False, False),
    "bandwidth_100g": (12.5, 8, 4, False, False),
    "bandwidth_400g": (50.0, 8, 4, False, False),
    "ratio_6p6d": (25.0, 6, 6, False, False),
    "ratio_4p8d": (25.0, 4, 8, False, False),
    "prompt_mix": (25.0, 8, 4, True, False),
    "bursty": (25.0, 8, 4, False, True),
}
results = []
for variant, (bandwidth, np, nd, prompt_mix, bursty) in variants.items():
    for seed in (11, 22, 33, 44, 55):
        tag = f"sensitivity_{variant}_s{seed}"
        rng = random.Random(seed)
        outputs = [64] * 150 + [2000] * 150
        rng.shuffle(outputs)
        prompts = [4096] * 150 + [32768] * 150 if prompt_mix else [16384] * 300
        if prompt_mix:
            rng.shuffle(prompts)
        t = 0.0
        with (root / f"{tag}.jsonl").open("w") as out:
            for i, (nin, nout) in enumerate(zip(prompts, outputs)):
                if bursty:
                    t = (i // 10) * 10 * 1000 / 1.2 + (i % 10) * 2 + rng.uniform(0, 0.5)
                else:
                    t += rng.expovariate(1.2 / 1000)
                if prompt_mix:
                    shared = 7 if nin == 4096 else 58
                    domain = 100000 if nin == 4096 else 200000
                else:
                    shared = 29
                    domain = 0
                blocks = nin // 512
                ids = list(range(domain, domain + shared)) + list(
                    range(
                        1000000 + i * (blocks - shared),
                        1000000 + (i + 1) * (blocks - shared),
                    )
                )
                out.write(
                    json.dumps(
                        {
                            "timestamp": t,
                            "input_length": nin,
                            "output_length": nout,
                            "hash_ids": ids,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        for policy, weight in (("w0", 0), ("w1", 1), ("w4", 4)):
            run = f"{tag}_{policy}"
            engine = {
                "backend": "vllm",
                "num_gpu_blocks": 519,
                "max_num_batched_tokens": 131072,
                "max_num_seqs": 32,
                "kv_transfer_bandwidth": bandwidth,
                "kv_transfer_bytes_per_token": 262144,
                "kv_transfer_timing_mode": "full_prompt",
                "kv_transfer_workers_per_nic": 1,
            }
            cfg = {
                "trace": f"{tag}.jsonl",
                "out_dir": f"sensitivity_outputs/{run}",
                "timing_table": "two_input_timing_extended.json",
                "block_size": 512,
                "num_prefill_workers": np,
                "num_decode_workers": nd,
                "engine": engine,
                "router": {"transfer_load_weight": weight},
            }
            path = root / f"{run}.json"
            path.write_text(json.dumps(cfg, separators=(",", ":")) + "\n")
            started = time.perf_counter()
            p = subprocess.run([str(binary), str(path)], capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError(f"{run}: {p.stderr}")
            s = json.loads((root / cfg["out_dir"] / "summary.json").read_text())
            rows = [
                json.loads(x)
                for x in (root / cfg["out_dir"] / "requests.jsonl")
                .read_text()
                .splitlines()
            ]
            assert (
                len(rows) == 300
                and s["ttft_ms"]["count"] == 300
                and all(x["terminal_status"] == "completed" for x in rows)
            )
            results.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "policy": policy,
                    "ttft_p99": s["ttft_ms"]["p99"],
                    "handoff_p99": s["handoff_ms"]["p99"],
                    "itl_p99": s["itl_ms"]["p99"],
                    "realized_hit": sum(x.get("reused_input_tokens") or 0 for x in rows)
                    / sum(x["input_length"] for x in rows),
                    "wall_s": time.perf_counter() - started,
                }
            )
        print("completed", tag, flush=True)
(root / "sensitivity_results.json").write_text(json.dumps(results, indent=2) + "\n")
print("runs", len(results))
