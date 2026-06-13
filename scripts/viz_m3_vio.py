"""Blog figure for M3: per held-out VIO sequence, top-down trajectory of ground truth vs stereo-VO-
alone (fails the blackouts) vs the agent's authored VIO (bridges them with the IMU). Blackout frames
are shaded so the rescue is visible. Runs the agent main.py + the front-end on each held-out seq."""
import os, sys, subprocess, tempfile, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ws/devel/whasuk/LenaLab"); sys.path.insert(0, str(ROOT))
from vo_lab.plugins.vo_ref.synthetic_vio import default_blackouts
RUN = ROOT / "_vo_vio_run"
CODE = RUN / "workspaces/vo-kitti-impl-001/code"
REF = ROOT / "vo_lab/plugins/vo_ref/run_kitti_stereo.py"
OUT = ROOT / "artifacts/blog"; OUT.mkdir(parents=True, exist_ok=True)


EVAL = ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py"


def run(solver, seq_input, cwd=None):
    """Run a solver; return (traj centres, poses-file-path) or (None, None)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run([sys.executable, str(solver)], cwd=str(cwd) if cwd else None,
                       env=dict(os.environ, LAB_DATA=str(seq_input), LAB_ARTIFACTS=str(tmp)),
                       check=True, capture_output=True, timeout=1200)
    except Exception as e:
        print("solver failed:", str(e)[:150]); return None, None
    f = tmp / "traj.txt"
    return (np.loadtxt(f).reshape(-1, 3) if f.exists() else None), (tmp / "poses.txt")


def grade_all(ho, seqs, poses_by_seq):
    """eval_kitti over a {seq_name: poses_path} set -> per-seq t_err%."""
    art = Path(tempfile.mkdtemp())
    for sq in seqs:
        s = sq.name.replace("seq_", "")
        pf = poses_by_seq.get(sq.name)
        if pf and Path(pf).exists():
            (art / f"poses_{s}.txt").write_bytes(Path(pf).read_bytes())
            (art / f"traj_{s}.txt").write_text("0 0 0\n")  # eval prefers poses_ when present
    ev = art / "eval"
    subprocess.run([sys.executable, str(EVAL)],
                   env=dict(os.environ, LAB_DATA=str(ho), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                   check=True, capture_output=True)
    d = json.load(open(ev / "heldout.json"))
    return {k: v["t_err_pct"] for k, v in d["per_seq"].items()}, d["t_err_pct"]


def main():
    ho = next((RUN / "cache/heldout").glob("*"))
    seqs = sorted(ho.glob("seq_*"))
    vo_tr, vio_tr, vo_pp, vio_pp = {}, {}, {}, {}
    for sq in seqs:
        vo_tr[sq.name], vo_pp[sq.name] = run(REF, sq / "input")
        vio_tr[sq.name], vio_pp[sq.name] = run(CODE / "main.py", sq / "input", cwd=CODE)
    vo_per, vo_mean = grade_all(ho, seqs, vo_pp)            # VO-alone t_err on THESE held-out seqs
    vio_per, vio_mean = grade_all(ho, seqs, vio_pp)
    fig, axes = plt.subplots(1, len(seqs), figsize=(6 * len(seqs), 5.2))
    if len(seqs) == 1:
        axes = [axes]
    for ax, sq in zip(axes, seqs):
        s = sq.name.replace("seq_", "")
        gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3); n = len(gt)
        ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.5, label="ground truth", zorder=3)
        if vo_tr[sq.name] is not None:
            v = vo_tr[sq.name]; ax.plot(v[:, 0], v[:, 2], "-", color="0.6", lw=1.5, label="stereo VO alone", zorder=2)
        if vio_tr[sq.name] is not None:
            v = vio_tr[sq.name]; ax.plot(v[:, 0], v[:, 2], "r--", lw=1.8, label="agent VIO (IMU-fused)", zorder=4)
        for st, L in default_blackouts(n):
            seg = gt[st:st + L]
            ax.plot(seg[:, 0], seg[:, 2], "-", color="orange", lw=6, alpha=0.35,
                    label="vision blackout" if st == default_blackouts(n)[0][0] else None, zorder=1)
        vop = next((x for k, x in vo_per.items() if s in k), None)
        vip = next((x for k, x in vio_per.items() if s in k), None)
        ax.set_title(f"{sq.name}  ·  VO-alone {vop:.1f}%  →  VIO {vip:.2f}%", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"M3 — the agent's VIO (red) fuses the IMU through vision blackouts (orange), where "
                 f"stereo-VO-alone (grey) loses motion\nheld-out: VO-alone {vo_mean:.1f}%  →  agent VIO "
                 f"{vio_mean:.2f}%   (bar 7%; reference VIO 4.2%)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = OUT / "m3_vio.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
