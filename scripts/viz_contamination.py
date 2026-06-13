"""Blog-ready visualization for the contamination probe (synthetic billed run).

Produces one figure with:
  (1) a sample synthetic stereo LEFT frame — shows the procedural world the agent has never seen;
  (2) per held-out sequence: top-down (X-Z) trajectory, the AGENT's estimate vs ground truth vs the
      reference VO, all anchored at the origin so accumulated drift is visible; official t_err noted.

Runs the agent's authored main.py + the reference solver on each held-out seq (GT-isolated: only
input/ is read), loads gt.txt, and plots. Usage:
  python3 scripts/viz_contamination.py [run_root]
Default run_root = ./_vo_synth_impl_run
"""
import os, sys, subprocess, tempfile, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_vo_synth_impl_run"
REF = ROOT / "vo_lab/plugins/vo_ref/run_kitti_stereo.py"
EVAL = ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py"
OUT = ROOT / "artifacts/blog"; OUT.mkdir(parents=True, exist_ok=True)


def find(p, *globs):
    for g in globs:
        m = sorted(Path(p).glob(g))
        if m:
            return m[0]
    return None


def run_solver(main_py, seq_input, tag):
    tmp = Path(tempfile.mkdtemp(prefix=f"viz_{tag}_"))
    try:
        subprocess.run([sys.executable, str(main_py)],
                       env=dict(os.environ, LAB_DATA=str(seq_input), LAB_ARTIFACTS=str(tmp)),
                       check=True, capture_output=True, timeout=1200)
    except Exception as e:
        print(f"  {tag} solver failed: {str(e)[:160]}")
        return None
    f = tmp / "traj.txt"
    return np.loadtxt(f).reshape(-1, 3) if f.exists() else None


def official_terr(heldout_dir, main_py):
    """Grade the agent on the held-out via the official metric (re-run, GT-isolated)."""
    art = Path(tempfile.mkdtemp(prefix="viz_grade_"))
    for sq in sorted(Path(heldout_dir).glob("seq_*")):
        s = sq.name.replace("seq_", "")
        tmp = Path(tempfile.mkdtemp())
        try:
            subprocess.run([sys.executable, str(main_py)],
                           env=dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(tmp)),
                           check=True, capture_output=True, timeout=1200)
            for f in ("traj.txt", "poses.txt"):
                if (tmp / f).exists():
                    (art / f"{f.split('.')[0]}_{s}.txt").write_bytes((tmp / f).read_bytes())
        except Exception:
            pass
    ev = art / "eval"
    subprocess.run([sys.executable, str(EVAL)],
                   env=dict(os.environ, LAB_DATA=str(heldout_dir), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                   check=True, capture_output=True)
    return json.load(open(ev / "heldout.json"))


def main():
    code = RUN / "workspaces/vo-kitti-impl-001/code"
    main_py = code / "main.py"
    heldout = find(RUN / "cache/heldout", "*")
    if not main_py.exists() or heldout is None:
        print(f"missing agent main.py ({main_py.exists()}) or held-out cache; run not finished?")
        return 1
    seqs = sorted(heldout.glob("seq_*"))
    print(f"agent main.py: {main_py} | held-out seqs: {[s.name for s in seqs]}")

    grade = official_terr(heldout, main_py)
    mean_terr = grade["t_err_pct"]
    per = grade["per_seq"]
    print(f"agent official t_err: {mean_terr:.3f}% | per-seq:",
          {k: round(v['t_err_pct'], 2) for k, v in per.items()})

    n = len(seqs)
    fig = plt.figure(figsize=(5 * (n + 1), 4.6))
    # panel 1: a sample synthetic frame
    ax0 = fig.add_subplot(1, n + 1, 1)
    sample = find(seqs[0] / "input", "left_000100.png", "left_0000*.png", "left_*.png")
    if sample is not None:
        ax0.imshow(plt.imread(str(sample)), cmap="gray")
    ax0.set_title("synthetic stereo frame\n(never-seen procedural world)", fontsize=10)
    ax0.axis("off")
    # panels: trajectory per seq
    for i, sq in enumerate(seqs):
        s = sq.name.replace("seq_", "")
        gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3)
        est = run_solver(main_py, sq / "input", f"agent_{s}")
        ref = run_solver(REF, sq / "input", f"ref_{s}")
        ax = fig.add_subplot(1, n + 1, i + 2)
        ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2.5, label="ground truth")
        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 2], "-", color="0.6", lw=1.4, label="reference VO")
        if est is not None:
            ax.plot(est[:, 0], est[:, 2], "r--", lw=1.6, label="agent (authored)")
        t = per.get(s, {}).get("t_err_pct")
        ax.set_title(f"{sq.name}  ·  agent t_err = {t:.2f}%" if t is not None else sq.name, fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.axis("equal")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Contamination probe — agent-authored stereo VO on PROVABLY-UNSEEN synthetic data\n"
                 f"agent mean t_err {mean_terr:.2f}%   vs   reference VO 1.91%   (bar 4%)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / "contamination_synthetic.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
