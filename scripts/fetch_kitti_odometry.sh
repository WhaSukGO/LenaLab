#!/usr/bin/env bash
# Fetch KITTI odometry (grayscale stereo) once, extract only the sequences we use (00/05/07).
# Monolithic ~21.6 GB gray zip (no per-sequence download); calib + poses are tiny.
set -euo pipefail
B=https://s3.eu-central-1.amazonaws.com/avg-kitti
CACHE="${LAB_KITTI_CACHE:-$HOME/.cache/vo_lab/kitti}"
SEQS="${KITTI_SEQS:-00 05 07}"
mkdir -p "$CACHE"; cd "$CACHE"

for z in data_odometry_calib.zip data_odometry_poses.zip data_odometry_gray.zip; do
  if [ ! -f "$z.done" ]; then
    echo "[kitti] downloading $z ..."
    curl -fL -C - -o "$z" "$B/$z"
    touch "$z.done"
  else
    echo "[kitti] $z already downloaded"
  fi
done

echo "[kitti] extracting calib + poses (small) ..."
unzip -oq data_odometry_calib.zip
unzip -oq data_odometry_poses.zip

echo "[kitti] extracting gray sequences: $SEQS ..."
for s in $SEQS; do
  unzip -oq data_odometry_gray.zip "dataset/sequences/$s/*" || true
done

echo "[kitti] done. layout:"
ls -d "$CACHE"/dataset/sequences/*/ 2>/dev/null | head
echo "[kitti] poses:"; ls "$CACHE"/dataset/poses/ 2>/dev/null | head
echo "[kitti] DONE"
