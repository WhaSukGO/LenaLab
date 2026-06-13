"""Synthetic VISUAL-INERTIAL sequence generator for M3.

Builds on synthetic_stereo (procedural stereo + exact GT) and synthetic_imu (honest IMU), and adds
VISION BLACKOUTS — short stretches of near-textureless frames where the stereo VO loses tracking.
On a blackout, vision-only odometry stalls and accrues large error; a working VIO must BRIDGE the
gap with the IMU. That is what makes the fusion necessary (not decorative) and the test meaningful.

Per sequence it writes (same held-out contract as KITTI, plus imu.txt):
  input/{left_%06d.png, right_%06d.png, intrinsics.txt, imu.txt}, gt.txt, gt_poses.txt
imu.txt: one line per frame `wx wy wz ax ay az` (body-frame gyro rad/s + accel m/s^2), interval [i,i+1].
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

from .synthetic_stereo import (make_trajectory, make_world, _render_view, FX, FY, CX, CY,
                               BASELINE, W, H)
from .synthetic_imu import generate_imu

DT = 0.1  # s per frame (~8.5 m/s at 0.85 m/frame)


def default_blackouts(n):
    """A couple of vision-blackout stretches placed in the middle of the run."""
    return [(int(n * 0.30), max(8, n // 18)), (int(n * 0.62), max(10, n // 16))]


def _is_blackout(i, blackouts):
    return any(s <= i < s + L for s, L in blackouts)


def generate_vio_sequence(input_dir, gt_dir=None, *, kind="A", n=280, seed=12345, blackouts=None):
    inp = Path(input_dir); inp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if blackouts is None:
        blackouts = default_blackouts(n)
    Twc = make_trajectory(kind, n)
    zreach = max(float(T[2, 3]) for T in Twc)
    planes = make_world(zmax=zreach)
    np.savetxt(inp / "intrinsics.txt", np.array([FX, FY, CX, CY, BASELINE]), fmt="%.6f")
    centres, poses = [], []
    for i, T in enumerate(Twc):
        if _is_blackout(i, blackouts):
            # near-textureless frame: uniform mid-grey + tiny noise -> no ORB corners, no disparity
            left = np.clip(118 + rng.normal(0, 2.0, (H, W)), 0, 255).astype(np.uint8)
            right = np.clip(118 + rng.normal(0, 2.0, (H, W)), 0, 255).astype(np.uint8)
        else:
            left = _render_view(planes, T, 0.0, rng)
            right = _render_view(planes, T, BASELINE, rng)
        cv2.imwrite(str(inp / f"left_{i:06d}.png"), left)
        cv2.imwrite(str(inp / f"right_{i:06d}.png"), right)
        centres.append(T[:3, 3].copy()); poses.append(T[:3, :4].reshape(-1).copy())
    imu = generate_imu(Twc, DT, seed=seed + 7)
    imu_rows = np.hstack([imu["gyro"], imu["accel"]])
    np.savetxt(inp / "imu.txt", imu_rows, fmt="%.8e")          # provided sensor (survives gt* staging)
    if gt_dir is not None:
        g = Path(gt_dir); g.mkdir(parents=True, exist_ok=True)
        np.savetxt(g / "gt.txt", np.array(centres), fmt="%.6f")
        np.savetxt(g / "gt_poses.txt", np.array(poses), fmt="%.8e")
    return {"n": n, "blackouts": blackouts, "dt": DT}
