"""GSplatModule — novel-view rendering of REAL KITTI scenes for the fidelity-ladder 'gaussian' rung.

Honest scope: this is **stereo-depth reprojection / point-splat rendering**, NOT optimized 3D Gaussian
Splatting. Optimized 3DGS (fitting/optimising a Gaussian mixture, gsplat/nerfstudio + CUDA) is a
research-grade effort on driving scenes (low parallax, dynamics, unbounded) and needs a heavy install.
This module delivers the achievable, verifiable core of the SAME idea — render **real-appearance** novel
views with **exact GT poses** — using only OpenCV/numpy:

  1. reconstruct(): for source frames, SGBM stereo disparity -> metric depth -> back-project EVERY pixel
     to a colour-carrying 3D point in world frame (KITTI cam->world poses).
  2. render(): project the world points into a target camera, z-buffer (near wins), splat + hole-fill.

So the 'gaussian' rung trains the learned VO on render-round-tripped real pixels (real appearance, render
artifacts, exact poses) — a faithful proxy for "high-fidelity rendered training data." TODO(phase2):
swap reconstruct/render internals for optimised 3DGS (gsplat) behind this same interface; the provider,
harness and grader do not change.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_calib_P(calib_path: Path):
    """KITTI calib.txt -> (fx, fy, cx, cy, baseline_m) from P0/P1 (baseline = -P1[0,3]/fx)."""
    rows = {}
    for ln in Path(calib_path).read_text().splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            rows[k.strip()] = np.fromstring(v, sep=" ")
    P0 = rows["P0"].reshape(3, 4)
    P1 = rows["P1"].reshape(3, 4)
    fx, fy, cx, cy = P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2]
    baseline = -P1[0, 3] / fx
    return float(fx), float(fy), float(cx), float(cy), float(baseline)


def _sgbm():
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                                 P1=8 * 7 * 7, P2=32 * 7 * 7, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2)


def depth_from_stereo(left, right, fx, baseline):
    disp = _sgbm().compute(left, right).astype(np.float32) / 16.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(disp > 0.5, fx * baseline / disp, 0.0)


def cloud_from_frame(gray, depth, K, Twc, *, step=2, zmax=45.0):
    """Back-project valid pixels of one frame to world points + grayscale colours."""
    fx, fy, cx, cy = K
    h, w = depth.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    zz = depth[vs, us]
    m = (zz > 0.5) & (zz < zmax)
    u, v, d = us[m].astype(np.float64), vs[m].astype(np.float64), zz[m].astype(np.float64)
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    pc = np.stack([x, y, d], axis=1)                       # camera frame
    Pw = (Twc[:3, :3] @ pc.T + Twc[:3, 3:4]).T             # world frame
    return Pw, gray[vs, us][m]


def render_view(points, colours, K, Twc_target, H, W, *, splat=2):
    """Project world points into the target camera (cam->world Twc_target), z-buffer, splat + hole-fill."""
    fx, fy, cx, cy = K
    R = Twc_target[:3, :3].T                               # world->cam
    t = -R @ Twc_target[:3, 3]
    Pc = (R @ points.T + t[:, None]).T
    z = Pc[:, 2]
    m = z > 0.3
    Pc, c, z = Pc[m], colours[m], z[m]
    u = np.round(fx * Pc[:, 0] / z + cx).astype(int)
    v = np.round(fy * Pc[:, 1] / z + cy).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, c = u[inb], v[inb], z[inb], c[inb]
    order = np.argsort(-z)                                 # far first -> near overwrites (z-buffer)
    u, v, c = u[order], v[order], c[order]
    img = np.zeros((H, W), np.uint8)
    img[v, u] = c
    if splat:                                              # fill pinholes from sparse projection
        k = np.ones((splat, splat), np.uint8)
        filled = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
        img = np.where(img > 0, img, filled)
    return img


class GSplatModule:
    """Reconstruct a colour point cloud from REAL KITTI stereo + render novel views with exact poses."""

    def __init__(self, *, step=2, zmax=45.0, src_radius=1, splat=2):
        self.step, self.zmax, self.src_radius, self.splat = step, zmax, src_radius, splat

    def render_sequence(self, seq_dir: Path, poses: np.ndarray, frame_idx: list[int], out_dir: Path,
                        world_offset=None):
        """Render one scene: for each chosen frame i, synthesise its view from neighbours {i-r..i+r}.

        seq_dir: KITTI sequence dir (image_0/, image_1/, calib.txt). poses: full (M,3,4) cam->world.
        frame_idx: source frame indices to render (in order). Writes out_dir/left_%06d.png +
        poses.txt (3x4 rows for the rendered frames) + intrinsics.txt.

        world_offset: optional 3-vector. The point cloud is built from the REAL camera positions, but
        each target view is rendered from a camera translated by world_offset (a parallel path), and the
        WRITTEN pose carries that offset. Relative motion between consecutive frames is unchanged (pure
        translation of the whole path), so the VO supervision stays exact — this is the NOVEL-VIEWPOINT
        augmentation real data cannot provide.
        """
        seq_dir, out_dir = Path(seq_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fx, fy, cx, cy, baseline = read_calib_P(seq_dir / "calib.txt")
        K = (fx, fy, cx, cy)
        left_paths = sorted((seq_dir / "image_0").glob("*.png"))
        right_paths = sorted((seq_dir / "image_1").glob("*.png"))
        H, W = cv2.imread(str(left_paths[frame_idx[0]]), cv2.IMREAD_GRAYSCALE).shape
        np.savetxt(out_dir / "intrinsics.txt", np.array([fx, fy, cx, cy]), fmt="%.6f")

        def Twc(i):
            T = np.eye(4); T[:3, :4] = poses[i]; return T

        cache = {}                                          # memoise per-source-frame world clouds

        def cloud(i):
            if i not in cache:
                L = cv2.imread(str(left_paths[i]), cv2.IMREAD_GRAYSCALE)
                Rr = cv2.imread(str(right_paths[i]), cv2.IMREAD_GRAYSCALE)
                d = depth_from_stereo(L, Rr, fx, baseline)
                cache[i] = cloud_from_frame(L, d, K, Twc(i), step=self.step, zmax=self.zmax)
            return cache[i]

        off = np.zeros(3) if world_offset is None else np.asarray(world_offset, float)
        kept_poses = []
        for out_j, i in enumerate(frame_idx):
            srcs = [s for s in range(i - self.src_radius, i + self.src_radius + 1)
                    if 0 <= s < len(left_paths)]
            P = np.concatenate([cloud(s)[0] for s in srcs], axis=0)   # cloud from REAL camera positions
            C = np.concatenate([cloud(s)[1] for s in srcs], axis=0)
            T_tgt = Twc(i); T_tgt[:3, 3] = T_tgt[:3, 3] + off          # render from the offset viewpoint
            img = render_view(P, C, K, T_tgt, H, W, splat=self.splat)
            cv2.imwrite(str(out_dir / f"left_{out_j:06d}.png"), img)
            kept_poses.append(T_tgt[:3, :4].reshape(-1))              # written pose carries the offset
            for s in list(cache):                           # evict far sources to bound memory
                if s < i - self.src_radius:
                    cache.pop(s, None)
        np.savetxt(out_dir / "poses.txt", np.array(kept_poses), fmt="%.8e")
        return len(frame_idx)
