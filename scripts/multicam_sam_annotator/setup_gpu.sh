#!/usr/bin/env bash
set -euo pipefail
source /home/ivan/anaconda3/etc/profile.d/conda.sh
conda create -n multicam_sam python=3.11 pip -y
conda activate multicam_sam
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 pillow pyyaml opencv-python-headless==4.11.0.86
python -m pip install git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf
