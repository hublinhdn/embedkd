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
# mixing indexes causes "operator torchvision::nms does not exist"). The index
# and the versions are those of the reference machine, recorded in
# docs/reference-environment.md; the CUDA build tag is not served by PyPI.
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130 -q
pip install -e ".[dev,onnx,plots]" -q

# Reproducing published numbers rather than developing? Pin everything else to
# the reference set as well:
#   pip install -r requirements.lock

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
