"""Sim-to-real closure figure: (1) KITTI seq_07 trajectory — ground truth vs sim-trained (collapses to
a straight line) vs real-trained (recovers the shape but still drifts); (2) the three corners as bars.
Shows the honest answer: real training closes much of the appearance gap (69.6->27.2m) but learned VO
stays far from classical on real driving."""
import sys; sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = Path("/home/ws/devel/whasuk/LenaLab")


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); s, d = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((d.T @ s) / len(src)); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: U[:, -1] *= -1; Rm = U @ Vt
    var = (s ** 2).sum() / len(src); c = D.sum() / var if var > 1e-12 else 1.0
    return (c * (Rm @ src.T)).T + (mu_d - c * Rm @ mu_s)


def load(p): return np.loadtxt(p).reshape(-1, 3)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

# (1) seq_07 trajectory: GT vs sim-trained vs real-trained
gt = load(next((R / "_vo_real_learned_run/cache/heldout").rglob("seq_07/gt.txt")))
sim = load(R / "_vo_sim2real_run/workspaces/sim2real-kitti-001/artifacts/traj_07.txt")
real = load(R / "_vo_real_learned_run/workspaces/real-learned-kitti-001/artifacts/traj_07.txt")
ax = axes[0]
n = min(len(gt), len(sim), len(real)); gt, sim, real = gt[:n], sim[:n], real[:n]
ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.6, label="ground truth", zorder=3)
ax.plot(*umeyama_sim3(sim, gt)[:, [0, 2]].T, "--", color="#d62728", lw=1.8, label="SIM-trained (69.6 m — collapses)", zorder=4)
ax.plot(*umeyama_sim3(real, gt)[:, [0, 2]].T, "-", color="#1f77b4", lw=1.8, label="REAL-trained (27.2 m — recovers shape)", zorder=5)
ax.scatter([gt[0, 0]], [gt[0, 2]], c="g", s=55, zorder=6, label="start")
ax.set_title("Real KITTI seq_07: training on real data\nrecovers the path shape sim-training misses", fontsize=11)
ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5)

# (2) the three corners
labels = ["synthetic\n→ synthetic\n(in-domain)", "REAL\n→ real\n(07/09)", "synthetic\n→ real\n(07/09)"]
vals = [0.45, 27.24, 69.57]; cols = ["#2ca02c", "#1f77b4", "#d62728"]
b = axes[1].bar(labels, vals, color=cols, log=True)
axes[1].axhline(3.5, color="0.4", ls=":", lw=1.6)
axes[1].text(2.4, 3.8, "classical VO on real KITTI ≈ few m", fontsize=8.5, color="0.3", ha="right")
axes[1].set_title("Held-out Sim3 ATE (log scale)\nreal data closes ~2.5× of the gap — but not to classical", fontsize=11)
axes[1].set_ylabel("ATE (m)"); axes[1].grid(alpha=0.3, axis="y", which="both")
for bar, v in zip(b, vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.2f} m", ha="center", fontsize=10)

fig.suptitle("Closing the sim-to-real loop: the gap is REAL (real training 69.6→27.2 m, ~2.5×) — but learned VO at this scale\n"
             "stays far from classical on real driving. Both true: appearance gap is large AND learned-VO-on-real isn't competitive.",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = R / "artifacts/blog/sim2real_closure.png"
fig.savefig(out, dpi=120); print("wrote", out)
