#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs hpn_replay with live AISimulate timing (feature ais-live).
#
#   run_hpn.sh <experiment.json>
#
# Expects sibling checkouts: <root>/dynamo and <root>/aisimulate. The Python
# environment is built from aisimulate's own uv.lock into <root>/venv (override
# with HPN_VENV) and is rebuilt whenever aisimulate's commit changes.
set -euo pipefail

readonly RUST_TOOLCHAIN=1.96.1
readonly PYTHON_VERSION=3.13

die() { echo "run_hpn.sh: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: run_hpn.sh <experiment.json>"
[[ -f $1 ]] || die "experiment file not found: $1"
experiment="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dynamo_root="$(cd "$script_dir/../../../.." && pwd)"
root="$(dirname "$dynamo_root")"
ais_root="$root/aisimulate"
ais_project="$ais_root/python/aisimulate"
venv="${HPN_VENV:-$root/venv}"
stamp="$venv/.aisimulate-commit"

for tool in uv cargo rustup protoc git; do
    command -v "$tool" >/dev/null || die "$tool not found; see README.md prerequisites"
done
[[ -f $ais_project/uv.lock ]] ||
    die "aisimulate checkout not found at $ais_root (clone it next to dynamo)"
rustup toolchain list | grep -q "^$RUST_TOOLCHAIN" ||
    die "Rust $RUST_TOOLCHAIN missing; run: rustup toolchain install $RUST_TOOLCHAIN"

# uv does not notice Rust-only changes to the editable aisimulate package, so
# rebuild it whenever the checkout's commit differs from the venv's, or the
# checkout has uncommitted changes.
commit="$(git -C "$ais_root" rev-parse HEAD)"
dirty="$(git -C "$ais_root" status --porcelain -- crates python)"
if [[ ! -x $venv/bin/python || "$(cat "$stamp" 2>/dev/null)" != "$commit" || -n $dirty ]]; then
    [[ -n $dirty ]] && echo "run_hpn.sh: aisimulate has uncommitted changes; rebuilding" >&2
    echo "run_hpn.sh: syncing $venv for aisimulate $commit" >&2
    UV_PROJECT_ENVIRONMENT="$venv" RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN" \
        uv sync --frozen --python "$PYTHON_VERSION" --managed-python \
        --project "$ais_project" --reinstall-package aisimulate
    echo "$commit" >"$stamp"
fi

python="$venv/bin/python"
# The embedded interpreter does not activate the venv or read its .pth files,
# so hand it the venv's resolved module path.
pythonpath="$("$python" -c 'import sys; print(":".join(p for p in sys.path[1:] if p))')"

cd "$dynamo_root"
PYO3_PYTHON="$python" PYTHONPATH="$pythonpath" \
    cargo run --release -p dynamo-mocker --features ais-live \
    --example hpn_replay -- "$experiment"
