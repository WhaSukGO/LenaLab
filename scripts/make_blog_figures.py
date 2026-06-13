#!/usr/bin/env python3
"""One-off: regenerate per-episode figures for the lab-chronicle blog.

Runs each archived agent artifact on its cached (held-out) frames, aligns the estimate to
ground truth with the SAME policy the grader uses (Sim3 for monocular, SE3 for RGB-D/SLAM),
and renders a trajectory comparison. Also draws the Episode-0 pipeline schematic.

Outputs PNGs into artifacts/blog/. Reads ground truth only to DRAW it — not part of the
verified loop. Numbers reproduce the recorded ATEs because the cached frames are the exact
strided subset the calibration/grader used.
"""
from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "blog"
OUT.mkdir(parents=True, exist_ok=True)


def run_artifact(code: Path, data_dir: Path, env_extra: dict | None = None, timeout=600) -> np.ndarray:
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, LAB_DATA=str(data_dir), LAB_ARTIFACTS=str(tmp))
    if env_extra:
        env.update(env_extra)
    subprocess.run([sys.executable, str(code)], env=env, check=True, timeout=timeout,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.loadtxt(tmp / "traj.txt").reshape(-1, 3)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    var_s = (Xs ** 2).sum() / n
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if (with_scale and var_s > 1e-12) else 1.0
    t = mu_d - s * R @ mu_s
    aligned = (s * (R @ src.T)).T + t
    ate = float(np.sqrt(((dst - aligned) ** 2).sum(1).mean()))
    return aligned, ate


def align_to_gt(est, gt, with_scale):
    m = min(len(est), len(gt))
    return (*umeyama(est[:m], gt[:m], with_scale), gt[:m])


# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline():
    """Episode 0 — the solver ⟂ verifier schematic."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")

    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="#333", lw=1.6))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=12.5,
                fontweight="bold")
        ax.text(x + w / 2, y + h - 0.78, "\n".join(lines), ha="center", va="top", fontsize=9.2,
                color="#222")

    box(0.4, 0.7, 4.3, 2.8, "SOLVER   (AI lives here)",
        ["Claude agent in a sandbox", "• writes the algorithm from scratch",
         "• no host shell · no network", "• eval.py off-limits",
         "produces → code / trajectory"], "#dbe9ff")
    box(6.3, 0.7, 4.3, 2.8, "VERIFIER   (no AI — pure Python)",
        ["• runs the code as a job", "• measures on a HELD-OUT split",
         "• closed-form geometry (ATE/RPE)", "• fixed alignment, GT isolated",
         "decides → VERIFIED / REJECTED"], "#d9f5e3")

    ax.add_patch(FancyArrowPatch((4.75, 2.1), (6.25, 2.1), arrowstyle="-|>",
                                 mutation_scale=22, lw=2.2, color="#444"))
    ax.text(5.5, 2.45, "artifact", ha="center", fontsize=9.5, style="italic")
    ax.text(5.5, 1.72, "(the only thing\nthat crosses)", ha="center", fontsize=8, color="#666")
    ax.text(5.5, 3.95, '"It ran" is never success — only a held-out number under a fixed bar earns ✔',
            ha="center", fontsize=10.5, fontweight="bold", color="#444")
    fig.tight_layout()
    p = OUT / "ep0_pipeline.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def fig_trajectory(est, gt, ate, title, fname, with_scale_label):
    """Single-trajectory top-down + side comparison (Episodes 1)."""
    aligned, ate_c, gt = align_to_gt(est, gt, with_scale=with_scale_label == "Sim(3)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (a, b, lab) in zip(axes, [(0, 2, "X–Z (top-down)"), (0, 1, "X–Y")]):
        ax.plot(gt[:, a], gt[:, b], "k-", lw=2, label="ground truth")
        ax.plot(aligned[:, a], aligned[:, b], "r-", lw=1.4, alpha=0.85, label="agent estimate")
        ax.scatter([gt[0, a]], [gt[0, b]], c="green", s=40, zorder=5, label="start")
        ax.set_title(lab); ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title}\nheld-out ATE = {ate:.3f} m  ({with_scale_label}-aligned)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = OUT / fname
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}  (recomputed ATE={ate_c:.3f})")


def fig_slam(gt, slam_est, vo_est, agent_est, fname):
    """Episode 4 — two panels: (A) loop closure works; (B) the agent diverged."""
    s_al, s_ate, g = align_to_gt(slam_est, gt, with_scale=False)
    v_al, v_ate, _ = align_to_gt(vo_est, gt, with_scale=False)
    a_al, a_ate, _ = align_to_gt(agent_est, gt, with_scale=False)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.4))

    # Panel A — GT vs reference SLAM vs VO-only (X-Z top-down)
    axA.plot(g[:, 0], g[:, 2], "k-", lw=2.4, label="ground truth")
    axA.plot(s_al[:, 0], s_al[:, 2], "-", color="#1a9850", lw=1.8,
             label=f"reference SLAM + loop closure  ({s_ate:.2f} m)")
    axA.plot(v_al[:, 0], v_al[:, 2], "-", color="#d73027", lw=1.4, alpha=0.85,
             label=f"VO-only, no loop closure  ({v_ate:.2f} m)")
    axA.scatter([g[0, 0]], [g[0, 2]], c="green", s=45, zorder=5)
    axA.set_title("A. Loop closure is necessary", fontweight="bold")
    axA.set_xlabel("X (m)"); axA.set_ylabel("Z (m)")
    axA.set_aspect("equal", "datalim"); axA.grid(alpha=0.3); axA.legend(fontsize=8.5, loc="best")

    # Panel B — the agent's diverged trajectory at its own scale
    axB.plot(a_al[:, 0], a_al[:, 2], "-", color="#762a83", lw=1.0, alpha=0.8,
             label=f"agent SLAM (diverged)  {a_ate:.0f} m")
    axB.plot(g[:, 0], g[:, 2], "k-", lw=2.4, label="ground truth (≈3 m scale)")
    axB.scatter([g[0, 0]], [g[0, 2]], c="green", s=45, zorder=5)
    axB.set_title("B. The agent's pose-graph diverged → REJECTED", fontweight="bold")
    axB.set_xlabel("X (m)"); axB.set_ylabel("Z (m)")
    axB.grid(alpha=0.3); axB.legend(fontsize=8.5, loc="best")
    axB.text(0.5, -0.16, "the verifier caught it — nothing false was accepted",
             transform=axB.transAxes, ha="center", fontsize=9.5, style="italic", color="#555")

    fig.suptitle("Episode 4 — SLAM with loop closure", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = OUT / fname
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}  (SLAM={s_ate:.2f}  VO-only={v_ate:.2f}  agent={a_ate:.1f})")


if __name__ == "__main__":
    fig_pipeline()

    # Episode 1 — monocular VO v1 (Sim3)
    tum_frames = ROOT / "_vo_tum_impl_run/cache/data/vo-tum-frames"
    tum_gt = np.loadtxt(ROOT / "_vo_tum_impl_run/cache/heldout/vo-tum-gt/gt.txt").reshape(-1, 3)
    v1 = run_artifact(ROOT / "artifacts/agent_authored_vo_tum_v1.py", tum_frames)
    fig_trajectory(v1, tum_gt, 0.124, "Episode 1 — Monocular VO v1 (PnP + landmark map)",
                   "ep1_mono_v1.png", "Sim(3)")

    # Episode 4 — SLAM on held-out fr1_room (SE3)
    sd = ROOT / "_vo_slam_run/cache/heldout/vo-rgbd-heldout-fr1_room/seq_fr1_room"
    slam_in = sd / "input"
    slam_gt = np.loadtxt(sd / "gt.txt").reshape(-1, 3)
    slam_ref = run_artifact(ROOT / "vo_lab/plugins/vo_ref/run_slam.py", slam_in)
    vo_only = run_artifact(ROOT / "vo_lab/plugins/vo_ref/run_rgbd.py", slam_in)
    agent_slam = run_artifact(ROOT / "artifacts/agent_authored_vo_slam_v1.py", slam_in)
    fig_slam(slam_gt, slam_ref, vo_only, agent_slam, "ep4_slam.png")

    print("\nAll figures in", OUT)
