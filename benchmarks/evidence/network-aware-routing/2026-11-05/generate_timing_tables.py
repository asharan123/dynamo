"""Generate the experiment timing tables through AISimulate's canonical FPM API.

Run with an AISimulate environment pinned to the revision in REPORT.md.
"""

import json
import os
from pathlib import Path

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

ROOT = Path(os.environ.get("HPN_EXP_ROOT", "/private/tmp/hpn_experiments_20261105"))
ROOT.mkdir(parents=True, exist_ok=True)
BATCHES = [1, 2, 4, 8, 16, 32]


def model(role):
    return RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="Qwen/Qwen3-32B",
            system="a100_sxm",
            backend="vllm",
            backend_version="current",
            worker_type=role,
            tp=4,
            estimation_mode="op_level",
            fallback_policy="deny",
        )
    )


def generate(name, new_axis, cached_axis, context_axis, prefill, decode):
    prefill_ms = [
        [
            [
                prefill.estimate_forward_pass_time_ms(
                    {
                        "scheduled_requests": {
                            "num_prefill_requests": batch,
                            "sum_prefill_tokens": batch * new,
                            "sum_prefill_kv_tokens": batch * cached,
                        }
                    }
                )
                for cached in cached_axis
            ]
            for new in new_axis
        ]
        for batch in BATCHES
    ]
    decode_ms = [
        [
            decode.estimate_forward_pass_time_ms(
                {
                    "scheduled_requests": {
                        "num_decode_requests": batch,
                        "sum_decode_kv_tokens": batch * context,
                    }
                }
            )
            for context in context_axis
        ]
        for batch in BATCHES
    ]
    table = {
        "model": "Qwen/Qwen3-32B",
        "system": "a100_sxm",
        "backend": "vllm",
        "backend_version": "current",
        "tp": 4,
        "prefill_batch": BATCHES,
        "prefill_new": new_axis,
        "prefill_cached": cached_axis,
        "prefill_ms": prefill_ms,
        "decode_batch": BATCHES,
        "decode_context": context_axis,
        "decode_ms": decode_ms,
    }
    (ROOT / name).write_text(json.dumps(table, separators=(",", ":")) + "\n")


def main():
    prefill, decode = model("prefill"), model("decode")
    generate(
        "two_input_timing.json",
        [1, 512, 1024, 2048, 4096, 8192, 16384],
        [0, 512, 1024, 2048, 4096, 8192, 12288, 16384],
        [1, 4096, 8192, 16384, 32768, 40960],
        prefill,
        decode,
    )
    generate(
        "two_input_timing_extended.json",
        [1, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 40960],
        [0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 40960],
        [1, 4096, 8192, 16384, 32768, 40960, 49152, 65536],
        prefill,
        decode,
    )


if __name__ == "__main__":
    main()
