"""KITTI-style segment metric grader — HARNESS-OWNED, GT-isolated (reads produced trajectories
+ held-out GT; never runs the solver). Reports the LENGTH-NORMALIZED translational drift used
by the KITTI odometry leaderboard, so numbers are comparable to PUBLISHED results — not the
truncated global ATE the external review flagged as non-comparable.

t_err(%) = mean over sub-sequence lengths L in {100,200,...,800} m (those that fit) and over
all start frames, of || Δest_aligned − Δgt || / L, where Δ is the i→j displacement and j is the
first frame whose cumulative GT path length from i reaches L. A single global SE(3) alignment is
applied first (CAVEAT: the official KITTI metric re-aligns each segment's START POSE, which needs
per-frame orientations — our trajectories store camera CENTRES only, so this is the translational
drift and NOT the official rotational error r_err). ATE is kept for continuity."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]   # KITTI sub-sequence lengths (metres)


def umeyama_se3(src, dst):
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd((Xd.T @ Xs) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    return (R @ src.T).T + (mu_d - R @ mu_s)


def _cum_lengths(gt):
    seg = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def segment_terr(est, gt):
    """KITTI-style translational drift as a FRACTION (×100 for %). Returns (t_err, per_len, total_m)."""
    m = min(len(est), len(gt))
    est, gt = est[:m], gt[:m]
    if m < 5:
        return 1e9, {}, 0.0
    aligned = umeyama_se3(est, gt)
    cl = _cum_lengths(gt)
    total = float(cl[-1])
    per_len = {}
    for L in LENGTHS:
        if L > total * 0.9:
            break
        errs = []
        for i in range(m):
            # first j>i whose GT path length from i reaches L
            target = cl[i] + L
            j = np.searchsorted(cl, target)
            if j >= m:
                continue
            d_est = aligned[j] - aligned[i]
            d_gt = gt[j] - gt[i]
            errs.append(np.linalg.norm(d_est - d_gt) / L)
        if errs:
            per_len[L] = float(np.mean(errs))
    t_err = float(np.mean(list(per_len.values()))) if per_len else 1e9
    return t_err, per_len, total


def ate_se3(est, gt):
    m = min(len(est), len(gt)); est, gt = est[:m], gt[:m]
    aligned = umeyama_se3(est, gt)
    return float(np.sqrt(((gt - aligned) ** 2).sum(1).mean()))


# ---- OFFICIAL KITTI metric (needs full 6-DoF poses; re-aligns each segment's start pose) ----

def _se3(p12):
    T = np.eye(4); T[:3] = np.asarray(p12, float).reshape(3, 4); return T


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t; return Ti


def official_segment_errors(est_poses, gt_poses):
    """KITTI devkit metric from full cam->world poses. Returns (t_err frac, r_err deg/m, len_m)."""
    n = min(len(est_poses), len(gt_poses))
    if n < 5:
        return 1e9, 1e9, 0.0
    E = [_se3(est_poses[i]) for i in range(n)]
    G = [_se3(gt_poses[i]) for i in range(n)]
    centres = np.array([G[i][:3, 3] for i in range(n)])
    cl = _cum_lengths(centres); total = float(cl[-1])
    t_per, r_per = {}, {}
    for L in LENGTHS:
        if L > total * 0.9:
            break
        terr, rerr = [], []
        for i in range(n):
            j = int(np.searchsorted(cl, cl[i] + L))
            if j >= n:
                continue
            dg = _inv(G[i]) @ G[j]
            de = _inv(E[i]) @ E[j]
            err = _inv(dg) @ de                       # residual relative pose
            terr.append(np.linalg.norm(err[:3, 3]) / L)
            rerr.append(float(np.arccos(np.clip((np.trace(err[:3, :3]) - 1) / 2, -1, 1))) / L)
        if terr:
            t_per[L] = float(np.mean(terr)); r_per[L] = float(np.mean(rerr))
    t_err = float(np.mean(list(t_per.values()))) if t_per else 1e9
    r_err = float(np.mean(list(r_per.values()))) if r_per else 1e9   # rad/m
    return t_err, r_err * 180.0 / np.pi, total       # r_err -> deg/m


def main() -> int:
    heldout = Path(os.environ["LAB_DATA"])
    art = Path(os.environ["LAB_ARTIFACTS"])           # traj_<n>.txt from the GT-free infer job
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in heldout.glob("seq_*") if (p / "gt.txt").exists())

    per_seq = {}
    mode = "centre_approx"
    for sq in seqs:
        s = sq.name.replace("seq_", "")
        traj = art / f"traj_{s}.txt"
        if not traj.exists():
            per_seq[sq.name] = {"t_err_pct": 1e9, "r_err_deg_m": 1e9, "ate_rmse": 1e9,
                                "error": "no trajectory"}; continue
        est = np.loadtxt(traj).reshape(-1, 3)
        gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3)
        est_poses_f, gt_poses_f = art / f"poses_{s}.txt", sq / "gt_poses.txt"
        if est_poses_f.exists() and gt_poses_f.exists():     # OFFICIAL metric (full poses)
            mode = "official"
            ep = np.loadtxt(est_poses_f).reshape(-1, 12)
            gp = np.loadtxt(gt_poses_f).reshape(-1, 12)
            t_err, r_err, total = official_segment_errors(ep, gp)
            per_seq[sq.name] = {"t_err_pct": t_err * 100.0, "r_err_deg_m": r_err,
                                "ate_rmse": ate_se3(est, gt), "path_len_m": round(total, 1)}
        else:                                                # centre-based approximation
            t_err, per_len, total = segment_terr(est, gt)
            per_seq[sq.name] = {"t_err_pct": t_err * 100.0, "r_err_deg_m": None,
                                "ate_rmse": ate_se3(est, gt), "path_len_m": round(total, 1),
                                "per_len": {str(k): round(v * 100, 3) for k, v in per_len.items()}}

    valid = [v for v in per_seq.values() if v["t_err_pct"] < 1e8]
    t_err_pct = float(np.mean([v["t_err_pct"] for v in valid])) if valid else 1e9
    ate = float(np.mean([v["ate_rmse"] for v in valid])) if valid else 1e9
    r_vals = [v["r_err_deg_m"] for v in valid if v.get("r_err_deg_m") is not None]
    r_err = float(np.mean(r_vals)) if r_vals else None
    if mode == "official":
        caveats = [
            "OFFICIAL KITTI odometry metric from full 6-DoF poses: length-normalized translational "
            "error t_err (%) AND rotational error r_err (deg/m), per-segment start-pose alignment.",
            "Comparable in form to the KITTI leaderboard; short/partial sequences use fewer of the "
            "100-800 m lengths, so it is INDICATIVE (a full-sequence run is a formal submission).",
        ]
    else:
        caveats = [
            "CENTRE-APPROX t_err only (no per-frame poses): one global SE(3) fit instead of "
            "per-segment alignment; r_err unavailable. Provide poses.txt for the official metric.",
        ]
    out = {"t_err_pct": t_err_pct, "r_err_deg_m": r_err, "ate_rmse": ate, "metric_mode": mode,
           "n_seqs": len(seqs), "metric": "kitti_t_err", "per_seq": per_seq, "caveats": caveats}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    rtxt = f" r_err {r_err:.5f} deg/m" if r_err is not None else ""
    print(f"KITTI [{mode}] t_err = {t_err_pct:.3f}%{rtxt}  (ATE {ate:.3f} m) over {len(seqs)} seqs: "
          f"{ {k: round(v['t_err_pct'],2) for k,v in per_seq.items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
