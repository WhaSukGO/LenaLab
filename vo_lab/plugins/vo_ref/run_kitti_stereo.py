"""Classical STEREO Visual Odometry — reference solver for the KITTI generalization test.

Stereo makes absolute scale OBSERVABLE (like RGB-D depth, but recovered from the left/right
baseline): a disparity map gives metric depth (Z = fx * baseline / disparity); features in
left_i are back-projected to metric 3-D, matched to left_{i+1}, and solved with PnP (3D-2D)
for a metric relative pose. Poses accumulate into a camera trajectory.

Reads from $LAB_DATA: left_%06d.png, right_%06d.png, intrinsics.txt (fx fy cx cy baseline_m).
Writes $LAB_ARTIFACTS/traj.txt (one `tx ty tz` camera centre per frame).

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
    left = sorted(data.glob("left_*.png"))
    n = len(left)
    if n == 0:
        print("ERROR: no left_*.png frames in LAB_DATA", file=sys.stderr); return 2

    if os.environ.get("VO_DEGENERATE") == "1":
        np.savetxt(art / "traj.txt", np.zeros((n, 3)), fmt="%.6f")
        print(f"degenerate trajectory ({n} frames)"); return 0

    fx, fy, cx, cy, baseline = np.loadtxt(data / "intrinsics.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                                 P1=8 * 7 * 7, P2=32 * 7 * 7, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2)

    def depth_of(i):
        l = cv2.imread(str(data / f"left_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(str(data / f"right_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        disp = sgbm.compute(l, r).astype(np.float64) / 16.0    # SGBM returns fixed-point *16
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(disp > 0.5, fx * baseline / disp, 0.0)
        return l, z

    def backproject(kps, depth):
        pts3d, idx = [], []
        for j, kp in enumerate(kps):
            u, v = kp.pt
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
                z = depth[vi, ui]
                if 1.0 < z < 60.0:                              # valid KITTI stereo range (m)
                    pts3d.append([(u - cx) * z / fx, (v - cy) * z / fy, z]); idx.append(j)
        return np.array(pts3d, np.float64), idx

    Twc = np.eye(4)                                            # world-from-camera
    traj = [Twc[:3, 3].copy()]
    poses = [Twc[:3, :4].reshape(-1).copy()]                   # full 3x4 (KITTI format)
    g0, z0 = depth_of(0)
    kp0, des0 = orb.detectAndCompute(g0, None)

    for i in range(1, n):
        g1, z1 = depth_of(i)
        kp1, des1 = orb.detectAndCompute(g1, None)
        ok = False
        if des0 is not None and des1 is not None and len(kp0) >= 6 and len(kp1) >= 6:
            matches = bf.match(des0, des1)
            if len(matches) >= 6:
                p3d_all, idx = backproject(kp0, z0)
                keep = {kp0_i: row for row, kp0_i in enumerate(idx)}
                obj, img = [], []
                for m in matches:
                    if m.queryIdx in keep:
                        obj.append(p3d_all[keep[m.queryIdx]]); img.append(kp1[m.trainIdx].pt)
                if len(obj) >= 6:
                    obj = np.array(obj, np.float64); img = np.array(img, np.float64)
                    okp, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj, img, K, None, reprojectionError=2.0, iterationsCount=150)
                    if okp and inl is not None and len(inl) >= 6:
                        R, _ = cv2.Rodrigues(rvec)
                        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = tvec.ravel()  # cam_i -> cam_{i+1}
                        Twc = Twc @ np.linalg.inv(M)
                        ok = True
        if not ok:
            print(f"WARN frame {i}: pose held (insufficient PnP)")
        traj.append(Twc[:3, 3].copy())
        poses.append(Twc[:3, :4].reshape(-1).copy())
        g0, z0, kp0, des0 = g1, z1, kp1, des1

    np.savetxt(art / "traj.txt", np.array(traj), fmt="%.6f")
    np.savetxt(art / "poses.txt", np.array(poses), fmt="%.8e")   # full 6-DoF for official metric
    print(f"stereo VO trajectory written: {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
