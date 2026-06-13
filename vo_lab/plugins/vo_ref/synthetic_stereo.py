"""Importable procedural synthetic-stereo generator (validated: the reference VO scores ~1.7%
t_err on it — see scripts/contamination_synthetic_kitti.py). Used by the contamination probe and
by SyntheticStereoProvider for a billed agent-authoring run on data no model has ever seen.

Geometry: rectified stereo (right cam = left translated +b along camera-x; disparity = fx*b/Z).
World = textured planes (corridor); per-pixel inverse ray-cast with a deterministic 3-D value-noise
texture so ORB finds corners AND SGBM block-matches. Exact ground truth by construction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

# camera / image config (KITTI-ish, fixed so intrinsics.txt is consistent across sequences)
W, H = 1226, 370
FX = FY = 707.0
CX, CY = 612.0, 185.0
BASELINE = 0.54

_uu, _vv = np.meshgrid(np.arange(W), np.arange(H))
_RAYS_CAM = np.stack([(_uu - CX) / FX, (_vv - CY) / FY, np.ones_like(_uu, float)], -1)


def _hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647).astype(np.uint64)
    h ^= (h >> np.uint64(13)); h = (h * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
    return (h.astype(np.float64) / 4294967295.0)


def _vnoise(P, scale):
    q = P / scale
    f = np.floor(q).astype(np.int64)
    t = q - f
    t = t * t * (3 - 2 * t)
    c = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (np.where(dx, t[:, 0], 1 - t[:, 0]) *
                     np.where(dy, t[:, 1], 1 - t[:, 1]) *
                     np.where(dz, t[:, 2], 1 - t[:, 2]))
                c = c + w * _hash3(f[:, 0] + dx, f[:, 1] + dy, f[:, 2] + dz)
    return c


def texture_at(P):
    v = (0.62 * _vnoise(P, 0.45) + 0.38 * _vnoise(P, 1.8))
    return np.clip(v, 0, 1)


def make_world(zmax):
    planes = [
        {"n": np.array([0.0, 1.0, 0.0]), "D": 5.0},     # ground  (y is DOWN)
        {"n": np.array([0.0, -1.0, 0.0]), "D": 8.0},    # ceiling
        {"n": np.array([1.0, 0.0, 0.0]), "D": -30.0},   # left wall
        {"n": np.array([-1.0, 0.0, 0.0]), "D": -30.0},  # right wall
    ]
    planes.append({"n": np.array([0.0, 0.0, -1.0]), "D": -(zmax + 70.0)})  # far back wall
    return planes


def make_trajectory(kind, n):
    """4x4 cam->world poses. Forward = +z; gentle yaw. `kind` varies the turn style."""
    Twc = []
    pos = np.zeros(3); yaw = 0.0; speed = 0.85
    for i in range(n):
        if kind == "A":
            dyaw = 0.0030 * np.sin(i * 0.020)
        elif kind == "B":
            dyaw = 0.0040 * np.sin(i * 0.012)
        else:  # "C" — slow broad drift
            dyaw = 0.0022 * np.sin(i * 0.008) + 0.0010 * np.cos(i * 0.03)
        yaw += dyaw
        fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        pos = pos + speed * fwd
        py = 0.05 * np.sin(i * 0.06)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [pos[0], py, pos[2]]
        Twc.append(T)
    return Twc


def _render_view(planes, Twc, cam_offset_x, rng):
    C = Twc[:3, 3] + Twc[:3, :3] @ np.array([cam_offset_x, 0.0, 0.0])
    dirs = _RAYS_CAM.reshape(-1, 3) @ Twc[:3, :3].T
    N = dirs.shape[0]
    best_t = np.full(N, np.inf); hit = np.zeros((N, 3))
    for pl in planes:
        n, D = pl["n"], pl["D"]
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (D - C @ n) / denom
        ok = (denom != 0) & (t > 1.0) & (t < best_t)
        if ok.any():
            best_t[ok] = t[ok]; hit[ok] = C + t[ok, None] * dirs[ok]
    valid = np.isfinite(best_t)
    tex = np.full(N, 0.06)
    if valid.any():
        tex[valid] = 0.10 + 0.85 * texture_at(hit[valid])
    img = (np.clip(tex, 0, 1) * 255).reshape(H, W).astype(np.uint8)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    return img


def generate_sequence(input_dir, gt_dir=None, *, kind="A", n=280, seed=12345):
    """Render a synthetic stereo sequence: left/right/intrinsics into input_dir. If gt_dir is given,
    also write gt.txt (camera centres) + gt_poses.txt (3x4 cam->world) into it. Returns n frames."""
    inp = Path(input_dir); inp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Twc = make_trajectory(kind, n)
    zreach = max(float(T[2, 3]) for T in Twc)
    planes = make_world(zmax=zreach)
    np.savetxt(inp / "intrinsics.txt", np.array([FX, FY, CX, CY, BASELINE]), fmt="%.6f")
    centres, poses = [], []
    for i, T in enumerate(Twc):
        left = _render_view(planes, T, 0.0, rng)
        right = _render_view(planes, T, BASELINE, rng)
        cv2.imwrite(str(inp / f"left_{i:06d}.png"), left)
        cv2.imwrite(str(inp / f"right_{i:06d}.png"), right)
        centres.append(T[:3, 3].copy()); poses.append(T[:3, :4].reshape(-1).copy())
    if gt_dir is not None:
        g = Path(gt_dir); g.mkdir(parents=True, exist_ok=True)
        np.savetxt(g / "gt.txt", np.array(centres), fmt="%.6f")
        np.savetxt(g / "gt_poses.txt", np.array(poses), fmt="%.8e")
    return n
