"""Classical monocular Visual Odometry — reference solver (Track A, MVP).

Runs as a harness JOB (not in an agent turn): reads an image sequence from $LAB_DATA,
writes the estimated camera trajectory (one `tx ty tz` per frame) to
$LAB_ARTIFACTS/traj.txt, plus a generator-reported metric stub to metrics.json.

Pipeline: ORB features -> BFMatcher(Hamming, crossCheck) between consecutive frames ->
findEssentialMat(RANSAC) -> recoverPose -> accumulate global pose with unit step scale.
Monocular scale is unobservable, so the global scale is left to the evaluator's Sim(3)
alignment (eval.py). intrinsics are read from $LAB_DATA/intrinsics.txt.

VO_DEGENERATE=1 makes this emit a static (origin) trajectory — the deliberately-bad
NEGATIVE control for the calibration gate (must be REJECTED)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_intrinsics(data_dir: Path) -> np.ndarray:
    fx, fy, cx, cy = np.loadtxt(data_dir / "intrinsics.txt")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _frames(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("frame_*.png"))


def main() -> int:
    data_dir = Path(os.environ["LAB_DATA"])
    artifacts = Path(os.environ["LAB_ARTIFACTS"])
    artifacts.mkdir(parents=True, exist_ok=True)
    degenerate = os.environ.get("VO_DEGENERATE") == "1"

    frames = _frames(data_dir)
    n = len(frames)
    if n == 0:
        print("ERROR: no frames in LAB_DATA", file=sys.stderr)
        return 2

    if degenerate:
        # Negative control: pretend nothing moved. A trajectory of all-zeros has no shape,
        # so the evaluator's Sim(3) alignment can't fit it to the moving ground truth.
        traj = np.zeros((n, 3), dtype=np.float64)
        np.savetxt(artifacts / "traj.txt", traj, fmt="%.6f")
        json.dump({"reported_ate_rmse": 0.0, "mode": "degenerate"},
                  open(artifacts / "metrics.json", "w"))
        print(f"degenerate trajectory written ({n} frames)")
        return 0

    K = _load_intrinsics(data_dir)
    # Tunable knobs (the committee's menu sets these via env; defaults are a solid baseline).
    nfeatures = int(os.environ.get("LAB_NFEATURES", "1500"))
    ransac_thresh = float(os.environ.get("LAB_RANSAC_THRESH", "1.0"))
    ratio = float(os.environ.get("LAB_RATIO", "0"))  # >0 -> Lowe ratio test instead of crossCheck
    orb = cv2.ORB_create(nfeatures=nfeatures)
    use_ratio = ratio > 0.0
    bf = (cv2.BFMatcher(cv2.NORM_HAMMING) if use_ratio
          else cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True))

    def _match(des_a, des_b):
        if use_ratio:
            knn = bf.knnMatch(des_a, des_b, k=2)
            return [m for pair in knn if len(pair) == 2
                    for m, n in [pair] if m.distance < ratio * n.distance]
        return bf.match(des_a, des_b)

    # Global camera pose (cam-to-world): world point = R_f @ x_cam + t_f. Start at origin.
    R_f = np.eye(3)
    t_f = np.zeros((3, 1))
    traj = [t_f.flatten().copy()]

    prev = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    kp_prev, des_prev = orb.detectAndCompute(prev, None)

    for i in range(1, n):
        cur = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        kp_cur, des_cur = orb.detectAndCompute(cur, None)
        updated = False
        if des_prev is not None and des_cur is not None and len(kp_prev) >= 8 and len(kp_cur) >= 8:
            matches = _match(des_prev, des_cur)
            if len(matches) >= 8:
                matches = sorted(matches, key=lambda m: m.distance)
                pts1 = np.float64([kp_prev[m.queryIdx].pt for m in matches])
                pts2 = np.float64([kp_cur[m.trainIdx].pt for m in matches])
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                               prob=0.999, threshold=ransac_thresh)
                if E is not None and E.shape == (3, 3):
                    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
                    # standard monocular VO accumulation with unit step scale
                    t_f = t_f + R_f @ t
                    R_f = R @ R_f
                    updated = True
        if not updated:
            print(f"WARN: frame {i} pose not updated (insufficient matches)")
        traj.append(t_f.flatten().copy())
        kp_prev, des_prev = kp_cur, des_cur

    traj = np.array(traj)
    np.savetxt(artifacts / "traj.txt", traj, fmt="%.6f")
    # reported metric is a self-report only; the evaluator re-measures on held-out GT.
    json.dump({"reported_ate_rmse": None, "frames": n, "mode": "orb-mono"},
              open(artifacts / "metrics.json", "w"))
    print(f"trajectory written: {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
