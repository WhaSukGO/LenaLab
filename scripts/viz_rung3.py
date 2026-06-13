"""Comprehensive rung-3 figure: (1) the net genuinely learns (loss curve), (2) it generalises on
unseen synthetic (trajectory tracks), (3) it collapses on real KITTI (sim-to-real), (4) the numbers.
Tells the whole story: learned-method capability is real, but sim-trained learned VO is not deployable.
"""
import sys, re; sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R = Path("/home/ws/devel/whasuk/LenaLab")


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((d.T @ s) / len(src))
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1; Rm = U @ Vt
    var = (s ** 2).sum() / len(src)
    c = D.sum() / var if var > 1e-12 else 1.0
    return (c * (Rm @ src.T)).T + (mu_d - c * Rm @ mu_s)


def load(p):
    a = np.loadtxt(p); return a.reshape(-1, 3)


def traj_panel(ax, gt_p, pred_p, title, color_ok):
    gt, pred = load(gt_p), load(pred_p)
    n = min(len(gt), len(pred)); gt, pred = gt[:n], pred[:n]
    al = umeyama_sim3(pred, gt)
    ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.5, label="ground truth", zorder=3)
    ax.plot(al[:, 0], al[:, 2], "--", color=color_ok, lw=1.8, label="learned VO (Sim3-aligned)", zorder=4)
    ax.scatter([gt[0, 0]], [gt[0, 2]], c="g", s=55, zorder=5, label="start")
    ax.set_title(title, fontsize=10.5); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# (1) training convergence
log = next(Path("_vo_synth_learned_impl_run").rglob("sandbox.log"))
pts = [(int(m[0]), float(m[1])) for m in re.findall(r"ep\s+0*(\d+)/\d+\s+loss=([0-9.]+)", log.read_text())]
ep, ls = zip(*sorted(set(pts)))
axes[0].semilogy(ep, ls, "o-", color="#1f77b4", lw=2)
axes[0].set_title("1. The net genuinely learns\n(training loss, log scale)", fontsize=10.5)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3, which="both")
axes[0].annotate(f"{ls[0]:.2f} → {ls[-1]:.3f}", xy=(0.5, 0.85), xycoords="axes fraction", fontsize=10, ha="center")

# (2) in-domain synthetic — tracks
sa = R / "_vo_synth_learned_impl_run/workspaces/synth-learned-impl-001/artifacts"
sh = R / "_vo_synth_learned_impl_run/cache/heldout"
traj_panel(axes[1], next(sh.rglob("seq_lte2/gt.txt")), sa / "traj_lte2.txt",
           "2. UNSEEN SYNTHETIC — generalises\nATE 0.45 m (beats reference 3.26 m)", "#2ca02c")

# (3) sim-to-real KITTI — collapses
ka = R / "_vo_sim2real_run/workspaces/sim2real-kitti-001/artifacts"
kh = R / "_vo_sim2real_run/cache/heldout"
traj_panel(axes[2], next(kh.rglob("seq_07/gt.txt")), ka / "traj_07.txt",
           "3. REAL KITTI photos — collapses\nATE 72 m (~150x worse)", "#d62728")

# (4) the numbers
labels = ["reference\nlearned VO", "agent VO\n(unseen synthetic)", "agent VO\n(real KITTI)"]
vals = [3.26, 0.45, 69.57]
cols = ["#7f7f7f", "#2ca02c", "#d62728"]
b = axes[3].bar(labels, vals, color=cols, log=True)
axes[3].set_title("4. Held-out Sim3 ATE (log scale)\ncapability ✓  ·  deployability ✗", fontsize=10.5)
axes[3].set_ylabel("ATE (m)"); axes[3].grid(alpha=0.3, axis="y", which="both")
for bar, v in zip(b, vals):
    axes[3].text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.2f} m", ha="center", fontsize=9.5)

fig.suptitle("Rung 3 — the agent authored AND trained a learned VO that GENERALISES on data it can't have memorised (0.45 m on unseen synthetic),\n"
             "but the same sim-trained model COLLAPSES on real KITTI photos (~70 m): real learned-method capability, not real-world deployability",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.9])
out = R / "artifacts/blog/rung3_learned.png"
fig.savefig(out, dpi=120); print("wrote", out)
