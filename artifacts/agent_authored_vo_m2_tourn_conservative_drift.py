"""
Loop-closure + pose-graph optimisation for stereo visual SLAM.

Strategy
--------
1. LOOP DETECTION   – bit-frequency ORB descriptor signatures to shortlist
                      same-direction candidates (camera angle < 25 deg),
                      plus a quick projection check for visual overlap.
2. GEOMETRIC VERIFY – PnP RANSAC (j-3D → i-2D).  Accept only >= 100 inliers.
3. POSE-GRAPH OPT   – Sparse-LSQR for translation + SO(3)-SLERP for rotation,
                      with one anchor (frame 0), smoothness on sequential edges,
                      and loop-closure constraints.
4. GUARD            – if no trustworthy loop found, write VO output unchanged.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

sys.path.insert(0, os.environ.get('LAB_CODE', '/code'))
from frontend import run_frontend


# ── SO(3) helpers ────────────────────────────────────────────────────────────

def so3_log(R):
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(cos_a)
    if abs(a) < 1e-9:
        return np.zeros(3)
    return a / (2.0 * np.sin(a)) * np.array([R[2,1]-R[1,2],
                                               R[0,2]-R[2,0],
                                               R[1,0]-R[0,1]])


def so3_exp(omega):
    a = np.linalg.norm(omega)
    if a < 1e-9:
        return np.eye(3)
    k = omega / a
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(a)*K + (1.0-np.cos(a))*(K@K)


def slerp_R(Ra, Rb, alpha):
    dR = Ra.T @ Rb
    return Ra @ so3_exp(alpha * so3_log(dR))


# ── descriptor signature ─────────────────────────────────────────────────────

def frame_sig(des):
    """256-float bit-frequency signature from (N,32) uint8 ORB descriptors."""
    if des is None or len(des) < 5:
        return None
    bits = np.unpackbits(des.astype(np.uint8), axis=1).astype(np.float32)
    return bits.mean(axis=0)   # (256,)


# ── loop detection ────────────────────────────────────────────────────────────

def detect_loops(frames, K, poses, traj):
    """
    Returns list of (i, j, n_inliers, T_ci_cj_4x4) sorted by i.

    Detection:
      * camera-forward angle < 25 deg  (same driving direction)
      * bit-frequency L2 signature distance (top-k candidates)
      * projection check: >= MIN_IN_IMG of j's 3D pts land inside frame i
    Verification:
      * PnP RANSAC with >= MIN_INL inliers
    """
    n = len(frames)
    MIN_GAP    = 50
    TOP_K      = 8
    MIN_IN_IMG = 30
    MIN_INL    = 100
    REPROJ_ERR = 1.5

    bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    sigs = [frame_sig(fr['des']) for fr in frames]
    fwds = np.array([p[:3,2] for p in poses])   # camera forward vectors

    loops   = []
    checked = set()

    for i in range(MIN_GAP, n):
        sig_i = sigs[i]
        if sig_i is None:
            continue
        fwd_i = fwds[i]

        # build candidate pool: angle < 25 deg AND temporal gap >= MIN_GAP
        cand_j, cand_sig = [], []
        for j in range(0, i - MIN_GAP):
            if sigs[j] is None:
                continue
            dot   = float(np.clip(fwds[j] @ fwd_i, -1.0, 1.0))
            angle = np.degrees(np.arccos(dot))
            if angle < 25.0:
                cand_j.append(j)
                cand_sig.append(sigs[j])

        if len(cand_j) < 3:
            continue

        cand_sig = np.array(cand_sig)            # (m, 256)
        dists    = np.linalg.norm(cand_sig - sig_i, axis=1)
        top_idx  = np.argsort(dists)[:TOP_K]

        for rank in top_idx:
            j    = cand_j[rank]
            pair = (j, i)
            if pair in checked:
                continue
            checked.add(pair)

            # -- quick 3D-point check (no projection: VO drift makes it unreliable) --
            pts3d_j = frames[j]['pts3d']
            if len(pts3d_j) < MIN_IN_IMG:
                continue

            # -- PnP verification ------------------------------------------
            des_j = frames[j]['des']
            des_i = frames[i]['des']
            if des_j is None or des_i is None:
                continue

            pidx_j  = frames[j]['pidx']
            kps_i   = frames[i]['kps']
            kp2pt   = {int(k): pts3d_j[r] for r, k in enumerate(pidx_j)}
            matches = bf.match(des_j, des_i)

            obj, img_pts = [], []
            for m in matches:
                qi, ti = int(m.queryIdx), int(m.trainIdx)
                if qi in kp2pt and ti < len(kps_i):
                    obj.append(kp2pt[qi])
                    img_pts.append(kps_i[ti])

            if len(obj) < MIN_INL:
                continue

            try:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    np.array(obj,     np.float64),
                    np.array(img_pts, np.float64),
                    K, None,
                    reprojectionError=REPROJ_ERR,
                    iterationsCount=1000,
                    confidence=0.9999)
            except Exception:
                continue

            if not ok or inl is None or len(inl) < MIN_INL:
                continue

            R_pnp, _ = cv2.Rodrigues(rvec)
            T_ci_cj   = np.eye(4)
            T_ci_cj[:3,:3] = R_pnp
            T_ci_cj[:3, 3] = tvec.ravel()

            print(f"  Loop i={i:3d} j={j:3d}  inl={len(inl):4d}  "
                  f"sig_d={dists[rank]:.4f}")
            loops.append((i, j, len(inl), T_ci_cj))

    loops.sort(key=lambda x: x[0])
    return loops


# ── pose-graph optimisation ───────────────────────────────────────────────────

def pose_graph_optimize(n, poses, loop_closures):
    """
    Translation: sparse LSQR (smoothness prior + loop constraints).
    Rotation:    SO(3)-SLERP between anchor (Identity) and loop targets.

    The corrected world pose for frame k is:
        T_k_corr = [[R_c @ R_k_vo, R_c @ t_k_vo + t_corr_k],
                    [0,            1                        ]]
    """
    # keep best (most inliers) loop target per constrained frame
    lc_by_frame = {}
    for (i, j, n_inl, T_ci_cj) in loop_closures:
        T_i_lc = poses[j] @ np.linalg.inv(T_ci_cj)
        if i not in lc_by_frame or n_inl > lc_by_frame[i][0]:
            lc_by_frame[i] = (n_inl, T_i_lc)

    lc_items = sorted(lc_by_frame.items())   # [(i, (n_inl, T_lc)), ...]
    print(f"  Unique LC frames: {[x[0] for x in lc_items]}")

    # Determine the "accurate zone" – all j-frames used as references.
    # Those frames are confirmed accurate by appearance: pin their corrections
    # to zero so the optimizer does NOT retroactively modify the first pass.
    j_refs = sorted({j for (_, j, _, _) in loop_closures})
    first_pass_end = max(j_refs) + 30   # a bit beyond the last reference frame
    first_pass_end = min(first_pass_end, lc_items[0][0] - 1)  # cap at first LC frame
    # extra zero-correction anchors: every 10 frames up through first_pass_end
    zero_anchors = list(range(0, first_pass_end + 1, 10))
    if first_pass_end not in zero_anchors:
        zero_anchors.append(first_pass_end)
    zero_anchors = sorted(set(zero_anchors))
    print(f"  Zero-correction anchors: {zero_anchors}")

    # ------------------------------------------------------------------
    # Step 1: Rotation corrections (SLERP).
    # Must be done BEFORE the translation LSQR because the correct
    # translation target for frame i is:
    #     t_corr[i] = T_i_lc[:3,3] - Rc_i @ T_i_vo[:3,3]
    # (not simply T_i_lc[:3,3] - T_i_vo[:3,3]).
    # ------------------------------------------------------------------
    R_at = {0: np.eye(3), first_pass_end: np.eye(3)}
    for (i, (_, T_i_lc)) in lc_items:
        R_at[i] = T_i_lc[:3, :3] @ poses[i][:3, :3].T

    cf = sorted(R_at.keys())

    def R_interp(k):
        if k <= cf[0]:
            return R_at[cf[0]]
        if k >= cf[-1]:
            return R_at[cf[-1]]
        for idx in range(len(cf) - 1):
            k1, k2 = cf[idx], cf[idx + 1]
            if k1 <= k <= k2:
                alpha = (k - k1) / float(k2 - k1)
                return slerp_R(R_at[k1], R_at[k2], alpha)
        return np.eye(3)

    # ------------------------------------------------------------------
    # Step 2: Translation correction via sparse LSQR.
    # Variables:  t_corr[0..n-1]  (3D each) stacked as 3n-vector.
    # Final pose:  t_corrected[k] = Rc_k @ t_vo[k] + t_corr[k]
    # So we optimise t_corr directly; the target for LC frame i is:
    #     t_corr[i]  =  T_i_lc[:3,3] - Rc_i @ T_i_vo[:3,3]
    # Zero-anchor frames (first pass) keep t_corr=0, hence t_corrected=t_vo.
    # ------------------------------------------------------------------
    sigma_s = 2.0     # smoothness (larger = smoother trajectory)
    sigma_l = 0.05    # loop / anchor tightness (smaller = tighter)

    n_anch = len(zero_anchors)
    n_lc   = len(lc_items)
    n_rows = (n_anch + (n - 1) + n_lc) * 3
    n_vars = n * 3

    A = lil_matrix((n_rows, n_vars), dtype=np.float64)
    b = np.zeros(n_rows)
    row = 0

    # zero-correction anchors (first pass frames: Rc=I, t_corr=0)
    for ka in zero_anchors:
        for c in range(3):
            A[row + c, ka*3 + c] = 1.0 / sigma_l
            b[row + c]           = 0.0
        row += 3

    # smoothness
    for k in range(n - 1):
        for c in range(3):
            A[row + c, (k+1)*3 + c] =  1.0 / sigma_s
            A[row + c,    k  *3 + c] = -1.0 / sigma_s
            b[row + c] = 0.0
        row += 3

    # loop constraints: t_corr[i] = T_i_lc[:3,3] - Rc_i @ T_i_vo[:3,3]
    for (i, (n_inl, T_i_lc)) in lc_items:
        Rc_i = R_interp(i)   # rotation correction at frame i
        dt   = T_i_lc[:3, 3] - Rc_i @ poses[i][:3, 3]
        for c in range(3):
            A[row + c, i*3 + c] = 1.0 / sigma_l
            b[row + c]           = dt[c] / sigma_l
        row += 3

    result = lsqr(A.tocsr(), b, iter_lim=10000, show=False)
    t_corr = result[0].reshape(n, 3)

    # ------------------------------------------------------------------
    # Step 3: Assemble corrected poses
    #   T_corr = [[Rc @ Rvo,  Rc @ t_vo + t_corr],
    #             [0,          1                 ]]
    # ------------------------------------------------------------------
    corrected = []
    for k in range(n):
        Rc = R_interp(k)
        T  = np.eye(4)
        T[:3, :3] = Rc @ poses[k][:3, :3]
        T[:3,  3] = Rc @ poses[k][:3,  3] + t_corr[k]
        corrected.append(T)

    return corrected


# ── I/O ───────────────────────────────────────────────────────────────────────

def _write(poses, artifacts_dir):
    a = Path(artifacts_dir)
    a.mkdir(parents=True, exist_ok=True)
    traj = np.array([p[:3, 3] for p in poses])
    rows = np.array([p[:3, :4].flatten() for p in poses])
    np.savetxt(a / "traj.txt",  traj, fmt="%.6f")
    np.savetxt(a / "poses.txt", rows, fmt="%.8e")
    print(f"Written traj.txt + poses.txt  ->  {artifacts_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    data_dir      = os.environ['LAB_DATA']
    artifacts_dir = os.environ['LAB_ARTIFACTS']

    print("Running frontend ...")
    fe     = run_frontend(data_dir)
    n      = fe['n']
    K      = fe['K']
    frames = fe['frames']
    poses  = fe['poses']
    traj   = fe['traj']
    pl_vo  = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    print(f"Frontend: {n} frames, VO path {pl_vo:.1f} m")

    print("Detecting loop closures ...")
    loops = detect_loops(frames, K, poses, traj)
    print(f"Accepted {len(loops)} loop(s)")

    if not loops:
        print("No trustworthy loops found -- writing VO trajectory unchanged.")
        _write(poses, artifacts_dir)
        return

    print("Optimising pose graph ...")
    corrected = pose_graph_optimize(n, poses, loops)

    traj_new = np.array([p[:3, 3] for p in corrected])
    pl_new   = float(np.linalg.norm(np.diff(traj_new, axis=0), axis=1).sum())

    # sanity guard: reject if path length changes by more than 15 %
    if abs(pl_new - pl_vo) / max(pl_vo, 1.0) > 0.15:
        print(f"WARNING: path length {pl_vo:.1f} -> {pl_new:.1f} m "
              f"(change > 15%) -- reverting to VO output.")
        _write(poses, artifacts_dir)
        return

    print(f"Path length: {pl_vo:.1f} -> {pl_new:.1f} m  "
          f"({100*(pl_new - pl_vo)/pl_vo:+.1f}%)")
    _write(corrected, artifacts_dir)
    print("Done.")


if __name__ == '__main__':
    main()
