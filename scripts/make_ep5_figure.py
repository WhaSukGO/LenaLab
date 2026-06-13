#!/usr/bin/env python3
"""Episode 5 figure: the agent's VERIFIED SLAM re-run (v2) vs ground truth and VO-only.

Runs the archived v2 artifact on the held-out fr1_room sequence, SE(3)-aligns to GT (the
grader's policy), and renders a top-down trajectory. Mirrors ep4 panel A but now with the
AGENT's own trajectory hugging GT — the before/after of the failure-memory loop.
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
SEQ = ROOT / "_vo_slam_impl_run2/cache/heldout/vo-rgbd-heldout-fr1_room/seq_fr1_room"


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
    gt = np.loadtxt(SEQ / "gt.txt").reshape(-1, 3)
    agent = run_artifact(ROOT / "artifacts/agent_authored_vo_slam_v2.py", SEQ / "input")
    vo = run_artifact(ROOT / "vo_lab/plugins/vo_ref/run_rgbd.py", SEQ / "input")
    m = min(len(gt), len(agent), len(vo))
    gt, agent, vo = gt[:m], agent[:m], vo[:m]
    a_al, a_ate = umeyama_se3(agent, gt)
    v_al, v_ate = umeyama_se3(vo, gt)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.4, label="ground truth")
    ax.plot(a_al[:, 0], a_al[:, 2], "-", color="#1a9850", lw=1.9,
            label=f"agent SLAM v2 (loop closure)  {a_ate:.2f} m  [VERIFIED]")
    ax.plot(v_al[:, 0], v_al[:, 2], "-", color="#d73027", lw=1.3, alpha=0.8,
            label=f"VO-only, no loop closure  {v_ate:.2f} m")
    ax.scatter([gt[0, 0]], [gt[0, 2]], c="green", s=45, zorder=5)
    ax.set_title("Episode 5 — Agent SLAM re-run after the failure-memory loop\n"
                 "the same frontier that diverged to 412 m, now VERIFIED", fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    p = OUT / "ep5_slam_verified.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}  (agent v2={a_ate:.3f} m, VO-only={v_ate:.3f} m)")


if __name__ == "__main__":
    main()
