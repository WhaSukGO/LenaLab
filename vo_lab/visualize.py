"""Render a VO trial: run an authored VO on a frame set, Sim(3)-align the estimate to the
held-out ground truth, and produce (a) a trajectory comparison plot and (b) a demo video
(input frame on the left, trajectory-so-far on the right).

  python -m vo_lab.visualize <main.py> <frames_dir> <gt.txt> <out_dir>

Used for documentation/demos; not part of the verified loop (it reads the GT to draw it)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """s,R,t minimizing ||dst - (s R src + t)|| (Umeyama 1991, with scale)."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    var_s = (Xs ** 2).sum() / n
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    return s, R, mu_d - s * R @ mu_s


def run_vo(code: Path, frames_dir: Path) -> np.ndarray:
    out = Path(tempfile.mkdtemp())
    env = dict(os.environ, LAB_DATA=str(frames_dir), LAB_ARTIFACTS=str(out))
    subprocess.run([sys.executable, str(code)], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.loadtxt(out / "traj.txt").reshape(-1, 3)


def render(code: Path, frames_dir: Path, gt_path: Path, out_dir: Path,
           *, title: str = "Monocular VO vs ground truth", fps: int = 15) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    est = run_vo(code, frames_dir)
    gt = np.loadtxt(gt_path).reshape(-1, 3)
    m = min(len(est), len(gt))
    est, gt = est[:m], gt[:m]
    s, R, t = umeyama_sim3(est, gt)
    aligned = (s * (R @ est.T)).T + t
    ate = float(np.sqrt(((gt - aligned) ** 2).sum(1).mean()))

    # (a) static comparison plot: XZ and XY projections
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (a, b, lab) in zip(axes, [(0, 2, "X-Z (top-down)"), (0, 1, "X-Y")]):
        ax.plot(gt[:, a], gt[:, b], "k-", lw=2, label="ground truth")
        ax.plot(aligned[:, a], aligned[:, b], "r--", lw=1.5, label="estimated (sim3-aligned)")
        ax.scatter(gt[0, a], gt[0, b], c="g", s=40, zorder=5, label="start")
        ax.set_title(lab); ax.set_aspect("equal", "datalim"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(f"{title}\nATE-RMSE = {ate:.4f} m over {m} frames")
    fig.tight_layout()
    plot_path = out_dir / "trajectory.png"
    fig.savefig(plot_path, dpi=110); plt.close(fig)

    # (b) demo video: input frame (left) + trajectory-so-far (right)
    frames = sorted(frames_dir.glob("frame_*.png"))[:m]
    lo = np.minimum(gt.min(0), aligned.min(0)); hi = np.maximum(gt.max(0), aligned.max(0))
    video_path = out_dir / "vo_demo.mp4"
    with imageio.get_writer(video_path, fps=fps, macro_block_size=None) as w:
        for i, fp in enumerate(frames):
            img = imageio.imread(fp)
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            fig = plt.figure(figsize=(4.8, 4.8), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(gt[:, 0], gt[:, 2], "k-", lw=1, alpha=.3)
            ax.plot(aligned[: i + 1, 0], aligned[: i + 1, 2], "r-", lw=2)
            ax.plot(gt[: i + 1, 0], gt[: i + 1, 2], "k-", lw=2)
            ax.scatter(aligned[i, 0], aligned[i, 2], c="r", s=30)
            ax.set_xlim(lo[0] - .1, hi[0] + .1); ax.set_ylim(lo[2] - .1, hi[2] + .1)
            ax.set_title(f"trajectory (frame {i+1}/{m})"); ax.set_aspect("equal"); ax.grid(alpha=.3)
            fig.tight_layout(); fig.canvas.draw()
            traj_img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            plt.close(fig)
            import cv2
            h = img.shape[0]
            traj_rs = cv2.resize(traj_img, (int(traj_img.shape[1] * h / traj_img.shape[0]), h))
            img_rs = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
            w.append_data(np.concatenate([img_rs, traj_rs], axis=1))

    return {"ate_rmse": ate, "frames": m, "plot": str(plot_path), "video": str(video_path),
            "recovered_scale": s}


if __name__ == "__main__":
    code, frames_dir, gt_path, out_dir = sys.argv[1:5]
    info = render(Path(code), Path(frames_dir), Path(gt_path), Path(out_dir))
    print(info)
