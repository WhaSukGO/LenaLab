#!/usr/bin/env python3
"""Episode 8 figure: the agent's GPU-trained learned VO on held-out KITTI (Track B, learned).

Uses the trajectories the agent's training job already produced (no retraining), Sim(3)-aligns
to GT (monocular -> scale unobservable), and plots both held-out driving sequences. Honest:
learned VO drifts (sub-classical) but the agent's optical-flow design beat the reference.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "blog"; OUT.mkdir(parents=True, exist_ok=True)
WS = ROOT / "_vo_kitti_learned_impl_run/workspaces/vo-learned-impl-001"
HO = ROOT / "_vo_kitti_learned_impl_run/cache/heldout/vo-kitti-learn-test-05_07"


def sim3(src, dst):
    n = len(src); mu_s, mu_d = src.mean(0), dst.mean(0); Xs, Xd = src - mu_s, dst - mu_d
    vs = (Xs ** 2).sum() / n; U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[-1, -1] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) / vs
    al = (s * (R @ src.T)).T + (mu_d - s * R @ mu_s)
    return al, float(np.sqrt(((dst - al) ** 2).sum(1).mean()))


fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, s in zip(axes, ["05", "07"]):
    est = np.loadtxt(WS / "artifacts" / f"traj_{s}.txt").reshape(-1, 3)
    gt = np.loadtxt(HO / f"seq_{s}" / "gt.txt").reshape(-1, 3)
    m = min(len(est), len(gt)); al, ate = sim3(est[:m], gt[:m]); g = gt[:m]
    ax.plot(g[:, 0], g[:, 2], "k-", lw=2.4, label="ground truth")
    ax.plot(al[:, 0], al[:, 2], "-", color="#2c7fb8", lw=1.6,
            label=f"agent learned VO (Sim3)  {ate:.1f} m")
    ax.scatter([g[0, 0]], [g[0, 2]], c="green", s=45, zorder=5)
    ax.set_title(f"KITTI seq_{s} (held-out, unseen)", fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
fig.suptitle("Episode 8 — Learned VO on GPU: the agent authors ML (+ optical flow) and beats "
             "the reference\nVERIFIED 19.8 m — drifts (sub-classical), but a real learned result",
             fontsize=12.5, fontweight="bold")
fig.tight_layout()
p = OUT / "ep8_learned.png"
fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
print("wrote", p)
