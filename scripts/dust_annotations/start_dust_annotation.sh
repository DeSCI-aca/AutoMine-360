#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x /home/ivan/anaconda3/bin/python ]]; then
    exec /home/ivan/anaconda3/bin/python "$script_dir/annotate_dust.py" "$@"
else
    exec python3 "$script_dir/annotate_dust.py" "$@"
fi
