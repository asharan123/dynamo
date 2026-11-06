# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Summarize paired policy results from followups.py's committed JSONL output."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path: Path) -> None:
    groups = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["case"], row["hit"], row["output"], row["rate"], row["mode"])
        run_key = (row["seed"], row["weight"])
        if run_key in groups[key]:
            raise ValueError(f"duplicate run: {key} {run_key}")
        groups[key][run_key] = row

    for key, runs in sorted(groups.items()):
        seeds = sorted({seed for seed, weight in runs if weight == 0})
        for weight in (1, 4, 1000):
            if not seeds or any((seed, weight) not in runs for seed in seeds):
                continue
            gains = []
            for seed in seeds:
                baseline = runs[seed, 0]["summary"]["ttft_ms"]["p99"]
                policy = runs[seed, weight]["summary"]["ttft_ms"]["p99"]
                gains.append(100 * (baseline - policy) / baseline)
            transfers = [
                statistics.mean(
                    runs[seed, candidate]["summary"]["transfer_ms"]["p99"]
                    for seed in seeds
                )
                for candidate in (0, weight)
            ]
            readmissions = [
                sum(
                    runs[seed, candidate]["summary"]["readmissions"]["decode"]
                    for seed in seeds
                )
                for candidate in (0, weight)
            ]
            print(
                f"{key} n={len(seeds)} w={weight} "
                f"gain_mean={statistics.mean(gains):+.2f}% "
                f"gain_median={statistics.median(gains):+.2f}% "
                f"range=[{min(gains):+.2f}%,{max(gains):+.2f}%] "
                f"positive={sum(gain > 0 for gain in gains)} "
                f"negative={sum(gain < 0 for gain in gains)} "
                f"transfer_p99={transfers[0]:.1f}/{transfers[1]:.1f}ms "
                f"decode_readmissions={readmissions[0]}/{readmissions[1]}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("results.jsonl"),
    )
    summarize(parser.parse_args().results)
