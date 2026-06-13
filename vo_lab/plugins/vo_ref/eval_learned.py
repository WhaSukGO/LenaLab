"""Independent grader for LEARNED VO — HARNESS-OWNED (the solver never writes this).

The train+infer job writes $LAB_ARTIFACTS/traj_<s>.txt (camera centres) for each held-out
test sequence <s>. This grader reads those, aligns each to the held-out GT
($LAB_DATA/seq_<s>/gt.txt — the secret labels, never in the trainer's input) with Sim(3)
(MONOCULAR scale is unobservable, so scale-corrected — the honest metric, same policy as the
classical monocular track), and reports the mean ATE-RMSE + RPE (local drift) across test
sequences. Higher-is-better vo_score = 1/(1+ATE) for the menu/loop."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def umeyama_sim3(src, dst):
    """s,R,t minimizing ||dst - (s R src + t)|| (Umeyama 1991, with scale)."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    var_s = (Xs ** 2).sum() / n
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    return s, R, mu_d - s * R @ mu_s


def score(est, gt):
    m = min(len(est), len(gt))
    if m < 3:
        return {"ate_rmse": 1e9, "rpe_trans": 1e9}
    est, gt = est[:m], gt[:m]
    var_e = ((est - est.mean(0)) ** 2).sum() / m
    if var_e < 1e-9:                                    # degenerate (static) -> GT spread
        return {"ate_rmse": float(np.sqrt(((gt - gt.mean(0)) ** 2).sum(1).mean())),
                "rpe_trans": float(np.sqrt((np.diff(gt, axis=0) ** 2).sum(1).mean()))}
    s, R, t = umeyama_sim3(est, gt)
    aligned = (s * (R @ est.T)).T + t
    ate = float(np.sqrt(((gt - aligned) ** 2).sum(1).mean()))
    de, dg = aligned[1:] - aligned[:-1], gt[1:] - gt[:-1]      # local relative translations
    rpe = float(np.sqrt(((de - dg) ** 2).sum(1).mean()))
    return {"ate_rmse": ate, "rpe_trans": rpe}


def main() -> int:
    heldout = Path(os.environ["LAB_DATA"])             # seq_*/gt.txt (grader-only)
    art = Path(os.environ["LAB_ARTIFACTS"])            # traj_<s>.txt from the train+infer job
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)

    per_seq = {}
    for sq in sorted(heldout.glob("seq_*")):
        s = sq.name.replace("seq_", "")
        traj = art / f"traj_{s}.txt"
        if not traj.exists():
            per_seq[sq.name] = {"ate_rmse": 1e9, "rpe_trans": 1e9, "error": "no traj"}; continue
        est = np.loadtxt(traj).reshape(-1, 3)
        gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3)
        per_seq[sq.name] = score(est, gt)

    ate = float(np.mean([v["ate_rmse"] for v in per_seq.values()])) if per_seq else 1e9
    rpe = float(np.mean([v["rpe_trans"] for v in per_seq.values()])) if per_seq else 1e9
    vo_score = 1.0 / (1.0 + ate)
    out = {"ate_rmse": ate, "rpe_trans": rpe, "vo_score": vo_score, "n_seqs": len(per_seq),
           "align": "sim3", "per_seq": per_seq,
           "caveats": [
               "monocular Sim(3): scale-corrected -> SHAPE accuracy, not metric scale",
               "rpe_trans is a centre-only POST-alignment drift proxy, not the standard Sturm RPE",
               "global ATE on a truncated frame window (NOT KITTI's official segment metric)",
           ]}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"mean held-out ATE = {ate:.4f} m  vo_score = {vo_score:.4f}  "
          f"over {len(per_seq)} seqs: { {k: round(v['ate_rmse'],2) for k,v in per_seq.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
