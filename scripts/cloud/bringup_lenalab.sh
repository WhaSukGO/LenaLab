#!/usr/bin/env bash
# Bring LenaLab up on a fresh GPU VM (bare VM with Docker + NVIDIA Container Toolkit, e.g. Lambda
# Cloud / GCP / AWS g5 / RunPod "VM"). Idempotent-ish: safe to re-run. ~10-15 min first time.
#
# PREREQS (the VM image must already have these — Lambda/RunPod GPU templates do):
#   - NVIDIA driver + `nvidia-smi` works
#   - Docker + NVIDIA Container Toolkit (`docker run --gpus all ... nvidia-smi` works)
#   - git, curl, python3
#
# YOU MUST SET THESE FIRST (export before running):
#   export VER2_REPO=<git url of your Touchstone/blueberry_ver2 spine>     # or rsync it up yourself
#   export ANTHROPIC_API_KEY=<your company key>                            # for live (billed) runs
# Optional: export LENALAB_REPO=<git url of LenaLab>   (defaults to the public repo)
set -euo pipefail

LENALAB_REPO="${LENALAB_REPO:-https://github.com/WhaSukGO/LenaLab.git}"
ROOT="${ROOT:-$HOME/devel/whasuk}"
mkdir -p "$ROOT"; cd "$ROOT"

echo "== 0. sanity: GPU + docker =="
nvidia-smi -L
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi -L

echo "== 1. clone LenaLab + the Touchstone spine (siblings) =="
[ -d LenaLab ] || git clone "$LENALAB_REPO" LenaLab
if [ ! -d blueberry_ver2 ]; then
  : "${VER2_REPO:?set VER2_REPO to your Touchstone/blueberry_ver2 git url, or rsync it to $ROOT/blueberry_ver2}"
  git clone "$VER2_REPO" blueberry_ver2
fi
cd LenaLab

echo "== 2. build the GPU sandbox image (torch+cuda+cv2) and the prep image (nuscenes-devkit) =="
docker build -f docker/Dockerfile.gpu-torch -t vo-gpu-torch:1 .
docker build -f docker/Dockerfile.bev       -t vo-bev:1 .

echo "== 3. fetch nuScenes mini (public, ~4GB) + prep BEV + occupancy caches =="
NUSC="$HOME/.cache/vo_lab/nuscenes"; mkdir -p "$NUSC"
if [ ! -f "$NUSC/v1.0-mini.tgz.done" ]; then
  curl -L -o "$NUSC/v1.0-mini.tgz" \
    "https://www.nuscenes.org/data/v1.0-mini.tgz"        # public, no auth
  ( cd "$NUSC" && tar xzf v1.0-mini.tgz && touch v1.0-mini.tgz.done )
fi
docker run --rm -v "$NUSC:/data" -v "$HOME/.cache/vo_lab/bev:/out" \
  -v "$(pwd)/scripts/prep_nuscenes_bev.py:/p.py" vo-bev:1 python /p.py /data /out
docker run --rm -v "$NUSC:/data" -v "$HOME/.cache/vo_lab/occ:/out" \
  -v "$(pwd)/scripts/prep_nuscenes_occ.py:/p.py" vo-bev:1 python /p.py /data /out

echo "== 4. .env (API key for live runs) =="
[ -n "${ANTHROPIC_API_KEY:-}" ] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > .env || \
  echo "  (no ANTHROPIC_API_KEY set — calibration works offline; live runs need it in .env)"

echo "== 5. smoke test (non-billed): occupancy calibration gate =="
export PYTHONPATH=".:../blueberry_ver2"
python3 -m vo_lab.run_occ_scaffold_calibration || echo "  (calibration returned non-zero — check the log)"

echo "== DONE. Live run examples =="
echo "  PYTHONPATH=.:../blueberry_ver2 python3 -m vo_lab.run_occ_scaffold_implement 0.051"
echo "  PYTHONPATH=.:../blueberry_ver2 python3 -m vo_lab.run_bev_scaffold_implement 0.082"
echo "  parallel fan-out: scripts/cloud/fanout.sh occ 3 0.051"
