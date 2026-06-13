"""Independent RGB-D / generalization grader — HARNESS-OWNED (the solver never writes this).

GT-ISOLATED design: the solver's main.py is NOT run here. A separate GT-free inference job
(infer_heldout.py) already ran main.py on each held-out sequence and wrote
$LAB_ARTIFACTS/traj_<n>.txt. This grader only READS those trajectories + the held-out GT
($LAB_DATA/seq_<n>/gt.txt) and scores — so the solver's code never executes in a container
where ground truth is mounted (closes the GT-read leak).

  1. GENERALIZATION — scores each held-out sequence the solver never authored against.
  2. METRIC + RPE — SE(3) alignment (no scale freebie); reports ATE-RMSE, RPE, scale error.

Aggregates mean ATE across held-out sequences into $LAB_EVAL_OUT/heldout.json."""
from __future__ import annotations

import json
import os
import sys
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
    heldout = Path(os.environ["LAB_DATA"])        # root with seq_*/ subdirs (incl. gt.txt)
    art = Path(os.environ["LAB_ARTIFACTS"])       # holds traj_<n>.txt from the GT-free infer job
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in heldout.glob("seq_*") if (p / "gt.txt").exists())
    if not seqs:
        json.dump({"ate_rmse": 1e9, "error": "no held-out sequences"},
                  open(eval_out / "heldout.json", "w")); return 0

    per_seq = {}
    for sq in seqs:
        s = sq.name.replace("seq_", "")
        traj = art / f"traj_{s}.txt"               # produced by infer_heldout.py (GT-free job)
        if not traj.exists():
            per_seq[sq.name] = {"ate_rmse": 1e9, "rpe_trans": 1e9, "scale": 1.0,
                                "error": "no trajectory produced"}; continue
        est = np.loadtxt(traj).reshape(-1, 3)
        gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3)
        per_seq[sq.name] = score(est, gt)

    ate = float(np.mean([s["ate_rmse"] for s in per_seq.values()]))
    rpe = float(np.mean([s["rpe_trans"] for s in per_seq.values()]))
    scale_err = float(np.mean([abs(s["scale"] - 1.0) for s in per_seq.values()]))
    out = {"ate_rmse": ate, "rpe_trans": rpe, "scale_err": scale_err,
           "scale_ok": bool(scale_err <= 0.25),     # SE3 metric: scale should be ~1 (no freebie)
           "n_seqs": len(seqs), "align": "se3", "per_seq": per_seq,
           "caveats": [
               "SE(3) metric alignment (no scale correction) — scale_err is the residual metric error",
               "rpe_trans is a centre-only, POST-alignment translational drift proxy, NOT the "
               "standard Sturm et al. SE(3) RPE (trajectories store camera centres, not orientations)",
               "global ATE over the materialized frame window; for KITTI this is NOT the official "
               "segment-based metric, so it is not comparable to published KITTI numbers",
           ]}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"mean held-out ATE={ate:.4f} m  RPE={rpe:.4f}  scale_err={scale_err:.3f}  "
          f"over {len(seqs)} seqs: { {k: round(v['ate_rmse'],3) for k,v in per_seq.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
