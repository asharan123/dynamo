import json
import os
import random
import subprocess
import time
from pathlib import Path

root = Path(os.environ.get("HPN_EXP_ROOT", "/private/tmp/hpn_experiments_20261105"))
repo = Path(os.environ.get("DYNAMO_ROOT", str(Path(__file__).resolve().parents[4])))
binary = repo / "target/release/examples/hpn_replay"
inputs = {
    "chat": repo
    / "recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl",
    "agent": repo
    / "recipes/deepseek-v4/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl",
}
rates = {"chat": 1.0, "agent": 0.4}
results = []
for name, path in inputs.items():
    source = [json.loads(line) for line in path.read_text().splitlines()]
    kept = [
        x
        for x in source
        if x["input_length"] + x["output_length"] <= 40960
        and x["input_length"] > 0
        and x["output_length"] > 0
    ]
    rng = random.Random(2026)
    t = 0.0
    trace = root / f"real_{name}.jsonl"
    with trace.open("w") as out:
        for x in kept:
            t += rng.expovariate(rates[name] / 1000)
            out.write(
                json.dumps(
                    {
                        "timestamp": t,
                        "input_length": x["input_length"],
                        "output_length": x["output_length"],
                        "hash_ids": x["hash_ids"],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    print(name, "source", len(source), "kept", len(kept), flush=True)
    for mode in ("destination_missing", "full_prompt"):
        for policy, weight in (("w0", 0), ("w1", 1), ("w4", 4)):
            tag = f"real_{name}_{mode}_{policy}"
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
                "trace": trace.name,
                "out_dir": f"real_outputs/{tag}",
                "timing_table": "two_input_timing_extended.json",
                "block_size": 512,
                "num_prefill_workers": 8,
                "num_decode_workers": 4,
                "engine": engine,
                "router": {
                    "transfer_load_weight": weight,
                    "transfer_decision_log": str(
                        root / "real_decisions" / f"{tag}.jsonl"
                    ),
                },
            }
            (root / "real_decisions").mkdir(exist_ok=True)
            c = root / f"{tag}.json"
            c.write_text(json.dumps(cfg, separators=(",", ":")) + "\n")
            start = time.perf_counter()
            p = subprocess.run([str(binary), str(c)], capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError(f"{tag}: {p.stderr}")
            s = json.loads((root / cfg["out_dir"] / "summary.json").read_text())
            rows = [
                json.loads(line)
                for line in (root / cfg["out_dir"] / "requests.jsonl")
                .read_text()
                .splitlines()
            ]
            assert (
                len(rows) == len(kept)
                and s["ttft_ms"]["count"] == len(kept)
                and all(x["terminal_status"] == "completed" for x in rows)
            )
            results.append(
                {
                    "trace": name,
                    "source_rows": len(source),
                    "kept_rows": len(kept),
                    "mode": mode,
                    "policy": policy,
                    "rate": rates[name],
                    "ttft_p99": s["ttft_ms"]["p99"],
                    "handoff_p99": s["handoff_ms"]["p99"],
                    "itl_p99": s["itl_ms"]["p99"],
                    "realized_hit": sum(x.get("reused_input_tokens") or 0 for x in rows)
                    / sum(x["input_length"] for x in rows),
                    "wall_s": time.perf_counter() - start,
                }
            )
            print("completed", tag, flush=True)
(root / "real_trace_results.json").write_text(json.dumps(results, indent=2) + "\n")
