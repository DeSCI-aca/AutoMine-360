#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SAM_ENV_PYTHON="${SAM_ENV_PYTHON:-/home/ivan/anaconda3/envs/multicam_sam/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
"$SAM_ENV_PYTHON" - <<'PY'
import torch,os
if not torch.cuda.is_available() and os.environ.get('SAM_ALLOW_CPU')!='1':
    raise SystemExit('GPU不可用。请在本机终端启动，并确认nvidia-smi正常；如需CPU，设置SAM_ALLOW_CPU=1。')
print('SAM device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','| torch:',torch.__version__,'| CUDA runtime:',torch.version.cuda,flush=True)
PY
exec "$SAM_ENV_PYTHON" server.py --scene "${1:-/home/ivan/project/IJRR/scene_0}" "${@:2}"
