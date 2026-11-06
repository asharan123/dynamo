#!/bin/bash
set -euo pipefail
root=/Users/ambikasharan/cs509/aisimulate-1
venv="$root/.hpn-venv"
export PYTHONPATH
PYTHONPATH="$("$venv/bin/python" -E -c 'import sys; print(":".join(p for p in sys.path[1:] if p))')"
export HPN_AISIMULATE_SOURCE=edcbc713ed65bf1c66a2fb3969523ee3cf77fc1a
export HPN_DYNAMO_SOURCE=011665977c21f5b91eb5b0d5d748364fa1155deb
exec "$root/dynamo/target/release/examples/hpn_replay" "$1"
