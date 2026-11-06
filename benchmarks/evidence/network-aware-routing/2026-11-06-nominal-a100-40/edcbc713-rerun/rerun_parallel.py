import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "nominal_followups",
    "/Users/ambikasharan/cs509/aisimulate-1/dynamo/benchmarks/evidence/network-aware-routing/2026-11-06-nominal-a100-40/followups_nominal_40gb.py",
)
followups = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followups)
followups.RUNNER = Path("/private/tmp/hpn_direct_edcbc713.sh")
root = Path("/private/tmp/hpn_edcbc713_20261106")


def execute(tasks, nic, ratio):
    os.environ["HPN_NIC_GBPS"] = str(nic)
    os.environ["HPN_DECODE_RATIO"] = str(ratio)
    tasks[0][0].mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(lambda task: followups.run(*task), tasks):
            pass


out = root / "nominal_25"
tasks = []
for hit, rate in ((0, 1.62), (90, 1.98)):
    for seed in range(101, 111):
        for weight in (0, 1, 4, 1000):
            tasks.append((out, "highload", seed, hit, "mixed", rate, "destination_missing", weight))
for hit in (0, 90):
    for rate in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        tasks.append((out, "short_saturation", 201, hit, "short", rate, "destination_missing", 0))
for hit, rates in ((0, (4, 5, 6, 7, 8)), (90, (5, 10, 15, 20, 25))):
    for rate in rates:
        tasks.append((out, "short_saturation_extended", 201, hit, "short", rate, "destination_missing", 0))
for hit, rates in ((0, (1.0, 1.5, 2.0)), (90, (1.5, 2.0, 2.5))):
    for rate in rates:
        for seed in (202, 203, 204):
            for weight in (0, 1, 4):
                tasks.append((out, f"short_r{rate}", seed, hit, "short", rate, "destination_missing", weight))
for hit, rate in ((0, 4), (90, 5), (90, 10)):
    for seed in (205, 206, 207):
        for weight in (0, 1, 4):
            tasks.append((out, "short_nearknee", seed, hit, "short", rate, "destination_missing", weight))
for hit, rate in ((0, 1.26), (90, 1.54)):
    for seed in (301, 302, 303):
        for mode in ("destination_missing", "full_prompt"):
            for weight in (0, 1, 4):
                tasks.append((out, "modes", seed, hit, "mixed", rate, mode, weight))
for seed in (304, 305, 306, 307, 308):
    for mode in ("destination_missing", "full_prompt"):
        for weight in (0, 1, 4):
            tasks.append((out, "modes_highload", seed, 90, "mixed", 1.98, mode, weight))
assert len(tasks) == 249
execute(tasks, 25, 1)

for bw, folder in ((16.7, "nic_16p7"), (12.5, "nic_12p5")):
    out = root / folder
    tasks = []
    for hit, rate in ((0, 1.62), (90, 1.98)):
        for seed in range(101, 106):
            for weight in (0, 1, 4):
                tasks.append((out, "nic_sensitivity", seed, hit, "mixed", rate, "destination_missing", weight))
    for seed in range(101, 106):
        for weight in (0, 1, 4):
            tasks.append((out, "nic_sensitivity", seed, 90, "mixed", 1.98, "full_prompt", weight))
    assert len(tasks) == 45
    execute(tasks, bw, 1)

out = root / "decode_ratio_0p763"
tasks = []
for hit, rate in ((0, 1.62), (90, 1.98)):
    for seed in range(101, 106):
        for weight in (0, 1, 4):
            tasks.append((out, "decode_sensitivity", seed, hit, "mixed", rate, "destination_missing", weight))
for seed in (202, 203, 204):
    for weight in (0, 1, 4):
        tasks.append((out, "decode_short", seed, 0, "short", 1.5, "destination_missing", weight))
assert len(tasks) == 39
execute(tasks, 25, 1555 / 2039)
