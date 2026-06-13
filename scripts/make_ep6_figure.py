#!/usr/bin/env python3
"""Episode 6 figure: the agent's VERIFIED KITTI stereo VO on the held-out driving sequences.

Runs the archived KITTI artifact on each unseen held-out sequence (05, 07), SE(3)-aligns to
GT (the grader's policy), and renders top-down driving trajectories. The cross-domain capstone:
an agent trained on nothing but indoor TUM now tracks outdoor driving, metrically.
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "blog"
OUT.mkdir(parents=True, exist_ok=True)
HO = ROOT / "_vo_kitti_impl_run/cache/heldout/vo-kitti-heldout-05_07"
ARTIFACT = ROOT / "artifacts/agent_authored_vo_kitti_v1.py"


def run_artifact(code: Path, data_dir: Path, timeout=600) -> np.ndarray:
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, LAB_DATA=str(data_dir), LAB_ARTIFACTS=str(tmp))
    subprocess.run([sys.executable, str(code)], env=env, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.loadtxt(tmp / "traj.txt").reshape(-1, 3)


def umeyama_se3(src, dst):
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    aligned = (R @ src.T).T + (mu_d - R @ mu_s)
    ate = float(np.sqrt(((dst - aligned) ** 2).sum(1).mean()))
    return aligned, ate


def main():
    seqs = ["seq_05", "seq_07"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, sq in zip(axes, seqs):
        gt = np.loadtxt(HO / sq / "gt.txt").reshape(-1, 3)
        est = run_artifact(ARTIFACT, HO / sq / "input")
        m = min(len(gt), len(est))
        al, ate = umeyama_se3(est[:m], gt[:m])
        ax.plot(gt[:m, 0], gt[:m, 2], "k-", lw=2.4, label="ground truth")
        ax.plot(al[:, 0], al[:, 2], "-", color="#1a9850", lw=1.7,
                label=f"agent stereo VO  {ate:.2f} m")
        ax.scatter([gt[0, 0]], [gt[0, 2]], c="green", s=45, zorder=5)
        ax.set_title(f"KITTI {sq} (held-out, unseen)", fontweight="bold")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m, forward)")
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
    fig.suptitle("Episode 6 — Generalization to outdoor driving (KITTI stereo): VERIFIED\n"
                 "an agent that had only seen indoor TUM, now tracking metric driving paths",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = OUT / "ep6_kitti_verified.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
