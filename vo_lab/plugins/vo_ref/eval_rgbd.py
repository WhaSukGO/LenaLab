"""Independent RGB-D / generalization grader — HARNESS-OWNED (the solver never writes this).

Two integrity upgrades over the monocular grader:
  1. GENERALIZATION — runs the solver's $LAB_CODE/main.py on EACH held-out sequence
     ($LAB_DATA/seq_*/input) that the solver never authored against, and scores each. A
     result must generalize across scenes, not overfit one.
  2. METRIC + RPE — aligns with SE(3) (no scale freebie), so a solver must use depth to get
     absolute scale right; reports ATE-RMSE, RPE (local drift), and a scale diagnostic.

Aggregates mean ATE across held-out sequences into $LAB_EVAL_OUT/heldout.json (key
`ate_rmse`, the bar metric). The held-out ground truth lives at seq_*/gt.txt — OUTSIDE the
input dir handed to the solver's code, so it cannot be read."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def umeyama_se3(src, dst):
    """R,t (no scale) minimizing ||dst-(R src + t)||; returns aligned, sim3-scale diagnostic."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    aligned = (R @ src.T).T + (mu_d - R @ mu_s)
    var_s = (Xs ** 2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    return aligned, scale


def score(est, gt):
    m = min(len(est), len(gt))
    if m < 3:
        return {"ate_rmse": 1e9, "rpe_trans": 1e9, "scale": 1.0, "n": m}
    est, gt = est[:m], gt[:m]
    aligned, scale = umeyama_se3(est, gt)
    ate = float(np.sqrt(((gt - aligned) ** 2).sum(1).mean()))
    de, dg = aligned[1:] - aligned[:-1], gt[1:] - gt[:-1]      # delta=1 relative translations
    rpe = float(np.sqrt(((de - dg) ** 2).sum(1).mean()))
    return {"ate_rmse": ate, "rpe_trans": rpe, "scale": scale, "n": m}


def main() -> int:
    heldout = Path(os.environ["LAB_DATA"])        # root with seq_*/ subdirs
    code = Path(os.environ["LAB_CODE"])
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in heldout.glob("seq_*") if (p / "input").is_dir())
    if not seqs:
        json.dump({"ate_rmse": 1e9, "error": "no held-out sequences"},
                  open(eval_out / "heldout.json", "w")); return 0

    per_seq = {}
    for sq in seqs:
        run_art = Path(tempfile.mkdtemp())
        env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(run_art))
        try:                                       # run the SOLVER's code on this unseen seq
            subprocess.run([sys.executable, str(code / "main.py")], env=env,
                           timeout=600, check=True, capture_output=True)
            est = np.loadtxt(run_art / "traj.txt").reshape(-1, 3)
            gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3)
            per_seq[sq.name] = score(est, gt)
        except Exception as e:                     # failed/timed-out run -> large error
            per_seq[sq.name] = {"ate_rmse": 1e9, "rpe_trans": 1e9, "scale": 1.0,
                                "error": str(e)[:120]}

    ate = float(np.mean([s["ate_rmse"] for s in per_seq.values()]))
    rpe = float(np.mean([s["rpe_trans"] for s in per_seq.values()]))
    scale_err = float(np.mean([abs(s["scale"] - 1.0) for s in per_seq.values()]))
    out = {"ate_rmse": ate, "rpe_trans": rpe, "scale_err": scale_err,
           "n_seqs": len(seqs), "align": "se3", "per_seq": per_seq}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"mean held-out ATE={ate:.4f} m  RPE={rpe:.4f}  scale_err={scale_err:.3f}  "
          f"over {len(seqs)} seqs: { {k: round(v['ate_rmse'],3) for k,v in per_seq.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
