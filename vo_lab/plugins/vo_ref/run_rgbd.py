"""Classical RGB-D Visual Odometry — reference solver for the improvement experiment.

Uses the DEPTH channel, so absolute scale is OBSERVABLE (no monocular ambiguity): features
in frame i are back-projected to metric 3-D points via depth, then matched to frame i+1 and
solved with PnP (3D-2D), giving a metric relative pose. Poses are accumulated into a
camera-trajectory.

Reads from $LAB_DATA: frame_%04d.png, depth_%04d.png (16-bit), intrinsics.txt
(fx fy cx cy depth_scale). Writes $LAB_ARTIFACTS/traj.txt (one `tx ty tz` per frame).

VO_DEGENERATE=1 emits a static trajectory (negative control)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    data = Path(os.environ["LAB_DATA"])
    art = Path(os.environ["LAB_ARTIFACTS"]); art.mkdir(parents=True, exist_ok=True)
    frames = sorted(data.glob("frame_*.png"))
    n = len(frames)
    if n == 0:
        print("ERROR: no frames in LAB_DATA", file=sys.stderr); return 2

    if os.environ.get("VO_DEGENERATE") == "1":
        np.savetxt(art / "traj.txt", np.zeros((n, 3)), fmt="%.6f")
        print(f"degenerate trajectory ({n} frames)"); return 0

    fx, fy, cx, cy, dscale = np.loadtxt(data / "intrinsics.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def load(i):
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(str(data / f"depth_{i:04d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / dscale
        return g, d

    def backproject(kps, depth):
        pts3d, idx = [], []
        for j, kp in enumerate(kps):
            u, v = kp.pt
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
                z = depth[vi, ui]
                if 0.1 < z < 8.0:  # valid TUM depth range
                    pts3d.append([(u - cx) * z / fx, (v - cy) * z / fy, z]); idx.append(j)
        return np.array(pts3d, np.float64), idx

    Twc = np.eye(4)                       # world-from-camera
    traj = [Twc[:3, 3].copy()]
    g0, d0 = load(0)
    kp0, des0 = orb.detectAndCompute(g0, None)

    for i in range(1, n):
        g1, _ = load(i)
        kp1, des1 = orb.detectAndCompute(g1, None)
        ok = False
        if des0 is not None and des1 is not None and len(kp0) >= 6 and len(kp1) >= 6:
            matches = bf.match(des0, des1)
            if len(matches) >= 6:
                p3d_all, idx = backproject(kp0, d0)
                keep = {kp0_i: row for row, kp0_i in enumerate(idx)}
                obj, img = [], []
                for m in matches:
                    if m.queryIdx in keep:
                        obj.append(p3d_all[keep[m.queryIdx]]); img.append(kp1[m.trainIdx].pt)
                if len(obj) >= 6:
                    obj = np.array(obj, np.float64); img = np.array(img, np.float64)
                    okp, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj, img, K, None, reprojectionError=3.0, iterationsCount=100)
                    if okp and inl is not None and len(inl) >= 6:
                        R, _ = cv2.Rodrigues(rvec)
                        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = tvec.ravel()  # cam_i -> cam_{i+1}
                        Twc = Twc @ np.linalg.inv(M)
                        ok = True
        if not ok:
            print(f"WARN frame {i}: pose held (insufficient PnP)")
        traj.append(Twc[:3, 3].copy())
        g0, d0, kp0, des0 = g1, load(i)[1], kp1, des1

    np.savetxt(art / "traj.txt", np.array(traj), fmt="%.6f")
    print(f"RGB-D trajectory written: {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
