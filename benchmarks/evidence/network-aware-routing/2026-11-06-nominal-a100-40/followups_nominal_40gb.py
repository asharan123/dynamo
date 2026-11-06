# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run independent follow-up workloads through the merged live AIS replay harness."""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

DYNAMO_ROOT = Path('/Users/ambikasharan/cs509/aisimulate-1/dynamo')
TEMPLATE = DYNAMO_ROOT / "lib/mocker/examples/hpn_calibration/template.json"
RUNNER = DYNAMO_ROOT / "lib/mocker/examples/hpn_calibration/run_hpn.sh"


def trace(
    path: Path, seed: int, hit: int, output: str, rate: float, n: int = 1000
) -> None:
    if path.exists():
        return
    rng = random.Random(seed)
    lengths = (
        [64] * n
        if output == "short"
        else [64] * (n // 2) + [2000] * (n - n // 2)
    )
    rng.shuffle(lengths)
    shared = 29 if hit == 90 else 0
    t = 0.0
    with path.open("w") as file:
        for i, length in enumerate(lengths):
            t += rng.expovariate(rate / 1000)
            family = rng.randrange(20)
            ids = [family * 100 + j for j in range(shared)]
            ids += [100000 + i * 32 + j for j in range(32 - shared)]
            row = {
                "timestamp": t,
                "input_length": 16384,
                "output_length": length,
                "hash_ids": ids,
            }
            file.write(json.dumps(row, separators=(",", ":")) + "\n")


def run(
    out: Path,
    case: str,
    seed: int,
    hit: int,
    output: str,
    rate: float,
    mode: str,
    weight: float,
) -> None:
    tag = f"{case}_s{seed}_h{hit}_{output}_r{rate}_{mode}_w{weight}_c1"
    folder = out / tag
    folder.mkdir(exist_ok=True)
    trace(folder / "trace.jsonl", seed, hit, output, rate)
    config = json.loads(TEMPLATE.read_text())
    config["engine"]["timing_model"]["config"]["systems_paths"] = [
        "/private/tmp/hpn_a100_40_systems"
    ]
    config["trace"] = "trace.jsonl"
    config["out_dir"] = "run"
    config["engine"]["kv_transfer_timing_mode"] = mode
    config["engine"]["kv_transfer_bandwidth"] = float(os.environ.get("HPN_NIC_GBPS", "25.0"))
    config["decode_engine"]["decode_speedup_ratio"] = float(os.environ.get("HPN_DECODE_RATIO", "1.0"))
    config["router"]["transfer_load_weight"] = weight
    config["router"]["transfer_decision_log"] = str(folder / "decisions.jsonl")
    (folder / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    result_file = folder / "result.json"
    if result_file.exists():
        if json.loads(result_file.read_text())["tag"] != tag:
            raise ValueError(f"stale result: {result_file}")
        return
    start = time.perf_counter()
    with (folder / "runner.log").open("w") as log:
        proc = subprocess.run(
            [str(RUNNER), str(folder / "experiment.json")],
            cwd=DYNAMO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode:
        raise RuntimeError(f'{tag}: runner failed; see {folder / "runner.log"}')
    summary = json.loads((folder / "run/summary.json").read_text())
    if summary.get("requests") != 1000:
        raise RuntimeError(f'{tag}: only {summary.get("requests")} requests')
    timing = json.loads((folder / "run/timing.json").read_text())
    row = {
        "tag": tag,
        "case": case,
        "seed": seed,
        "hit": hit,
        "output": output,
        "rate": rate,
        "mode": mode,
        "weight": weight,
        "contention": True,
        "wall_s": time.perf_counter() - start,
        "summary": summary,
        "sources": timing["sources"],
    }
    result_file.write_text(json.dumps(row, indent=2) + "\n")
    print(
        tag,
        "ttft_p99=",
        round(summary["ttft_ms"]["p99"], 2),
        "readmissions=",
        summary["readmissions"]["decode"],
        flush=True,
    )


def main(phase: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if phase == "highload":
        for hit, rate in ((0, 1.62), (90, 1.98)):
            for seed in range(101, 111):
                for weight in (0, 1, 4, 1000):
                    run(
                        out, phase, seed, hit, "mixed", rate,
                        "destination_missing", weight,
                    )
    elif phase == "short_saturation":
        for hit in (0, 90):
            for rate in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
                run(out, phase, 201, hit, "short", rate, "destination_missing", 0)
    elif phase == "short_saturation_extended":
        for hit, rates in ((0, (4, 5, 6, 7, 8)), (90, (5, 10, 15, 20, 25))):
            for rate in rates:
                run(out, phase, 201, hit, "short", rate, "destination_missing", 0)
    elif phase == "short":
        for hit, rates in ((0, (1.0, 1.5, 2.0)), (90, (1.5, 2.0, 2.5))):
            for rate in rates:
                for seed in (202, 203, 204):
                    for weight in (0, 1, 4):
                        run(
                            out, f"{phase}_r{rate}", seed, hit, "short", rate,
                            "destination_missing", weight,
                        )
    elif phase == "short_nearknee":
        for hit, rate in ((0, 4), (90, 5), (90, 10)):
            for seed in (205, 206, 207):
                for weight in (0, 1, 4):
                    run(
                        out, phase, seed, hit, "short", rate,
                        "destination_missing", weight,
                    )
    elif phase == "modes":
        for hit, rate in ((0, 1.26), (90, 1.54)):
            for seed in (301, 302, 303):
                for mode in ("destination_missing", "full_prompt"):
                    for weight in (0, 1, 4):
                        run(out, phase, seed, hit, "mixed", rate, mode, weight)
    elif phase == "modes_highload":
        for seed in (304, 305, 306, 307, 308):
            for mode in ("destination_missing", "full_prompt"):
                for weight in (0, 1, 4):
                    run(out, phase, seed, 90, "mixed", 1.98, mode, weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "highload", "short_saturation", "short_saturation_extended", "short",
            "short_nearknee", "modes", "modes_highload",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.phase, args.out)
