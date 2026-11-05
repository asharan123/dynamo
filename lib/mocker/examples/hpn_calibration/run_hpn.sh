#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Runs hpn_replay with live AISimulate timing (feature ais-live).
#
#   run_hpn.sh <experiment.json>
#
# Expects sibling checkouts: <root>/dynamo and <root>/aisimulate. The Python
# environment is built from aisimulate's own uv.lock into <root>/.hpn-venv
# (override with HPN_VENV) and is rebuilt whenever the aisimulate source changes.
set -euo pipefail

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
venv="${HPN_VENV:-$root/.hpn-venv}"
stamp="$venv.source"

for tool in uv cargo rustup protoc git; do
    command -v "$tool" >/dev/null || die "$tool not found; see README.md prerequisites"
done
[[ -f $ais_project/uv.lock ]] ||
    die "aisimulate checkout not found at $ais_root (clone it next to dynamo)"

# Build the editable aisimulate package with dynamo's pinned Rust toolchain.
toolchain="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$dynamo_root/rust-toolchain.toml")"
[[ -n $toolchain ]] || die "no channel in $dynamo_root/rust-toolchain.toml"
rustup run "$toolchain" rustc -V >/dev/null 2>&1 ||
    die "Rust $toolchain missing; run: rustup toolchain install $toolchain"

# uv does not notice Rust-only changes to the editable aisimulate package, so
# <venv>.source records the source the venv was built from. A dirty tree never matches,
# so it rebuilds every run and again once the edits are reverted.
commit="$(git -C "$ais_root" rev-parse HEAD 2>/dev/null)" ||
    die "aisimulate at $ais_root is not a git checkout"
source_id="$commit"
[[ -z "$(git -C "$ais_root" status --porcelain -- crates python)" ]] || source_id="$commit-dirty"

if [[ "$(cat "$stamp" 2>/dev/null)" != "$commit" || $source_id != "$commit" ]]; then
    if [[ -e $venv && ! -f $stamp ]]; then
        die "$venv exists but was not created by run_hpn.sh; remove it or set HPN_VENV"
    fi
    echo "syncing" >"$stamp"
    echo "run_hpn.sh: syncing $venv for aisimulate $source_id" >&2
    UV_PROJECT_ENVIRONMENT="$venv" RUSTUP_TOOLCHAIN="$toolchain" \
        uv sync --frozen --python "$PYTHON_VERSION" --managed-python \
        --project "$ais_project" --reinstall-package aisimulate
    echo "$source_id" >"$stamp"
fi

python="$venv/bin/python"
"$python" -c 'import aisimulate_core.sdk' ||
    die "the venv at $venv cannot import aisimulate_core; delete it and rerun"
# The embedded interpreter does not activate the venv or read its .pth files,
# so hand it the venv's resolved module path (this replaces any PYTHONPATH).
pythonpath="$("$python" -c 'import sys; print(":".join(p for p in sys.path[1:] if p))')"

cd "$dynamo_root"
PYO3_PYTHON="$python" PYTHONPATH="$pythonpath" \
    cargo run --release -p dynamo-mocker --features ais-live \
    --example hpn_replay -- "$experiment"
