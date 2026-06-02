"""Independent VO grader — HARNESS-OWNED (the solver can never write or edit this).

Runs as the evaluator's JOB in a separate context. Reads:
  - the solver's estimated trajectory at $LAB_ARTIFACTS/traj.txt
  - the HELD-OUT ground-truth trajectory at $LAB_DATA/gt.txt  (evaluator-only split)
Computes ATE-RMSE after Sim(3) Umeyama alignment (scale-corrected), the standard metric
for MONOCULAR VO whose absolute scale is unobservable. Writes the measured metric to
$LAB_EVAL_OUT/heldout.json; ScriptEvaluator (ver2) reads it and applies the fixed oracle.

The alignment policy is FIXED here on purpose: the solver cannot choose how it is scored
(prevents the scale/alignment silent-wrong gaming vector). This is a self-contained,
textbook Umeyama (1991) — equivalent to evo's `ape --align --correct_scale`."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Find s,R,t minimizing || dst - (s R src + t) ||^2 (Umeyama 1991, with scale)."""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    Xs = src - mu_s
    Xd = dst - mu_d
    var_s = (Xs ** 2).sum() / n
    cov = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t, var_s


def ate_rmse(est: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    # Degenerate estimate (no shape, e.g. all-zeros negative control): cannot be aligned,
    # so score it as the GT's own spread about its centroid — a large, honest error.
    var_e = ((est - est.mean(axis=0)) ** 2).sum() / est.shape[0]
    if var_e < 1e-9:
        spread = np.sqrt(((gt - gt.mean(axis=0)) ** 2).sum(axis=1).mean())
        return float(spread), 1.0
    s, R, t, _ = umeyama_sim3(est, gt)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(gt - aligned, axis=1)
    return float(np.sqrt((err ** 2).mean())), s


def main() -> int:
    artifacts = Path(os.environ["LAB_ARTIFACTS"])
    gt_dir = Path(os.environ["LAB_DATA"])          # held-out split (evaluator-only)
    eval_out = Path(os.environ["LAB_EVAL_OUT"])
    eval_out.mkdir(parents=True, exist_ok=True)

    est = np.loadtxt(artifacts / "traj.txt").reshape(-1, 3)
    gt = np.loadtxt(gt_dir / "gt.txt").reshape(-1, 3)
    m = min(len(est), len(gt))
    if m < 3:
        print("ERROR: trajectories too short to score", file=sys.stderr)
        json.dump({"ate_rmse": 1e9, "n": m}, open(eval_out / "heldout.json", "w"))
        return 0
    est, gt = est[:m], gt[:m]

    rmse, scale = ate_rmse(est, gt)
    # vo_score is a HIGHER-IS-BETTER restatement of ATE (monotone decreasing, in (0,1]).
    # ver2's Menu/loop/history all assume "higher is better", so the committee path and
    # goal_metric use vo_score; calibration uses ate_rmse<=bar directly. Same bar, two views.
    vo_score = 1.0 / (1.0 + rmse)
    out = {"ate_rmse": rmse, "vo_score": vo_score, "n": int(m), "align": "sim3",
           "recovered_scale": scale}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"ATE-RMSE (sim3-aligned) = {rmse:.4f} | vo_score = {vo_score:.4f} "
          f"over {m} poses; scale={scale:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
