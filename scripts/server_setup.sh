#!/usr/bin/env bash
# One-time GPU server setup. Run from the repository root after git clone/pull:
#   bash scripts/server_setup.sh
# Creates .venv, installs CUDA torch + embedkd, runs the test suite, then
# downloads and validates CUB-200-2011 into ./data.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi || { echo "nvidia-smi failed: is this the right machine?"; exit 1; }

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip -q

echo "== Installing torch (CUDA) and embedkd =="
# torch and torchvision from the SAME index (timm depends on torchvision;
# mixing indexes causes "operator torchvision::nms does not exist").
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install -e ".[dev,onnx,plots]" -q

echo "== Environment sanity: test suite (CPU, ~10s) =="
pytest -q

echo "== GPU smoke: quickstart with AMP =="
embedkd fit --config configs/quickstart_cpu.yaml \
    --set run.tag=gpu_smoke --set train.amp=true

echo "== Dataset: CUB-200-2011 (~1.1 GB) =="
embedkd datasets download cub200 --root data
embedkd datasets validate --config configs/d1_cub200_teacher.yaml

echo "== Setup complete =="
echo "Next: bash scripts/run_d1_cub200.sh   (inside tmux; ~1-2 GPU days total)"
