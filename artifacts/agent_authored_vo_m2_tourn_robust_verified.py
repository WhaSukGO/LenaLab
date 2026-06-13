"""
Loop-closure + pose-graph optimisation on top of the locked stereo-VO front-end.

Loop detection : cosine-similarity shortlist → BFMatcher → PnP RANSAC (strict)
Correction     : Left-multiply rubber-band for dominant loop, then Jacobi
                 iterations for all loops simultaneously.
                 Left-multiply is geometrically exact for the anchor loop:
                   corrected[k] = C_k @ poses[k]
                   C_k = xi_to_mat(alpha * xi_Ci),  alpha = (k-j)/(i-j)
                   C_i = corrected[j] @ inv(T_meas) @ inv(corrected[i])
"""
import os, sys, time
import numpy as np
import cv2

# ── SE(3) helpers ───────────────────────────────────────────────────────────────

def mat_to_xi(T):
    rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
    return np.concatenate([rvec, T[:3, 3]])

def xi_to_mat(xi):
    R = cv2.Rodrigues(xi[:3])[0]
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = xi[3:]
    return M

def se3_inv(T):
    R = T[:3, :3].T
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ T[:3, 3]
    return M

def T_rel(Ti, Tj):
    return se3_inv(Ti) @ Tj


# ── Loop detection ──────────────────────────────────────────────────────────────

def detect_loops(frames, K,
                 min_gap=30, max_cands=4, sim_thresh=0.70,
                 min_full_matches=30, min_inliers=20,
                 reproj_err=3.0, max_t=50.0):
    """
    Returns list of (i, j, T_meas, n_inliers) with i > j.
    T_meas = T_{i←j} : transforms a point in j-cam coords to i-cam coords.
    (Equivalently: T_rel(Twc_i, Twc_j) = inv(Twc_i) @ Twc_j.)
    """
    n = len(frames)
    print(f"[LC] Building signatures for {n} frames …", flush=True)

    sigs, valid = [None]*n, []
    for idx, f in enumerate(frames):
        d = f['des']
        if d is not None and len(d) >= 20:
            sigs[idx] = d.astype(np.float32).mean(axis=0)
            valid.append(idx)

    if len(valid) < 2:
        return []

    valid_arr = np.array(valid, dtype=int)
    S         = np.vstack([sigs[v] for v in valid]).astype(np.float32)
    S        /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    CS        = S @ S.T

    bf         = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    loop_edges = []
    confirmed  = set()

    for qi in range(len(valid)):
        i = valid_arr[qi]
        if i < min_gap:
            continue
        mask = valid_arr < (i - min_gap)
        if not np.any(mask):
            continue

        sims = CS[qi][mask]
        cand = np.where(mask)[0]
        top_n = 0

        for ci_order in np.argsort(-sims):
            if top_n >= max_cands:
                break
            sim = float(sims[ci_order])
            if sim < sim_thresh:
                break
            j = int(valid_arr[cand[ci_order]])
            if (i, j) in confirmed or (j, i) in confirmed:
                continue

            fi, fj = frames[i], frames[j]
            try:
                matches = bf.match(fi['des'], fj['des'])
            except Exception:
                top_n += 1; continue
            if len(matches) < min_full_matches:
                top_n += 1; continue

            fj_kp2row = {int(fj['pidx'][r]): r for r in range(len(fj['pidx']))}
            obj3d, img2d = [], []
            for mm in matches:
                if mm.trainIdx in fj_kp2row:
                    obj3d.append(fj['pts3d'][fj_kp2row[mm.trainIdx]])
                    img2d.append(fi['kps'][mm.queryIdx])

            top_n += 1
            if len(obj3d) < 15:
                continue

            obj3d = np.array(obj3d, np.float64)
            img2d = np.array(img2d, np.float64)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj3d, img2d, K, None,
                reprojectionError=reproj_err, iterationsCount=500,
                confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or inl is None or len(inl) < min_inliers:
                continue
            t_dist = float(np.linalg.norm(tvec))
            if t_dist > max_t:
                continue

            n_inl = int(len(inl))
            R_pnp, _ = cv2.Rodrigues(rvec)
            T_meas = np.eye(4)
            T_meas[:3, :3] = R_pnp
            T_meas[:3, 3]  = tvec.ravel()

            print(f"  [LC] {j}<->{i}  inl={n_inl}  sim={sim:.3f}  t={t_dist:.1f}m",
                  flush=True)
            confirmed.add((i, j))
            loop_edges.append((i, j, T_meas, n_inl))

    print(f"[LC] {len(loop_edges)} loops.", flush=True)
    return loop_edges


# ── Left-multiply rubber-band loop correction ───────────────────────────────────

def rubber_band_left(corrected, i, j, T_meas, n, full_tail=True):
    """
    One rubber-band pass for loop (i, j), i > j.
    j is the anchor (alpha=0), i is the endpoint (alpha=1).
    corrected[k] = C_k @ corrected[k]  (left-multiply, world-frame correction).
    C_k = xi_to_mat(alpha * xi_Ci)
    C_i = corrected[j] @ inv(T_meas) @ inv(corrected[i])

    If full_tail=True, frames k > i also get the full C_i correction.
    """
    C_i = corrected[j] @ se3_inv(T_meas) @ se3_inv(corrected[i])
    xi_Ci = mat_to_xi(C_i)

    span = i - j  # positive
    for k in range(j + 1, n):    # j itself is anchor (alpha=0, no change)
        alpha = min((k - j) / span, 1.0)
        C_k = xi_to_mat(alpha * xi_Ci)
        corrected[k] = C_k @ corrected[k]

    return corrected


def jacobi_left(poses_in, loop_edges, n_iter=15):
    """
    Jacobi iterations for all loops simultaneously using left-multiply.
    Each iteration:
      - Compute C_i for every loop based on current poses
      - For each frame k, accumulate weighted alpha * xi_Ci from all loops [j,i] it belongs to
      - Apply average correction C_k @ corrected[k]
    """
    n = len(poses_in)
    corrected = [T.copy() for T in poses_in]

    for it in range(n_iter):
        corr_xi  = np.zeros((n, 6))
        weights  = np.zeros(n)
        max_err  = 0.0

        for (i, j, T_meas, n_inl) in loop_edges:
            # Current residual
            T_ij_est = T_rel(corrected[i], corrected[j])
            T_err    = se3_inv(T_meas) @ T_ij_est
            xi_err   = mat_to_xi(T_err)
            max_err  = max(max_err, float(np.linalg.norm(xi_err)))

            # Left-multiply correction for endpoint i
            C_i    = corrected[j] @ se3_inv(T_meas) @ se3_inv(corrected[i])
            xi_Ci  = mat_to_xi(C_i)

            w    = float(n_inl) / 30.0
            span = i - j

            # Vectorised accumulation
            ks     = np.arange(j + 1, i + 1)
            alphas = (ks - j) / span
            corr_xi[ks] += w * alphas[:, None] * xi_Ci[None, :]
            weights[ks] += w

        # Apply simultaneous corrections (left-multiply) to current corrected
        for k in range(1, n):
            if weights[k] > 0:
                C_k = xi_to_mat(corr_xi[k] / weights[k])
                corrected[k] = C_k @ corrected[k]

        print(f"  [JL] iter {it+1:2d}  max_err={max_err:.4f}", flush=True)
        if max_err < 0.05:
            break

    return corrected


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    data_dir      = os.environ['LAB_DATA']
    artifacts_dir = os.environ['LAB_ARTIFACTS']
    os.makedirs(artifacts_dir, exist_ok=True)

    print("[FE] Running stereo VO front-end …", flush=True)
    sys.path.insert(0, os.environ['LAB_CODE'])
    from frontend import run_frontend
    fe     = run_frontend(data_dir)
    n      = fe['n']
    K      = fe['K']
    frames = fe['frames']
    poses  = fe['poses']
    print(f"[FE] n={n}  t={time.time()-t0:.1f}s", flush=True)

    loop_edges = detect_loops(
        frames, K,
        min_gap=30, max_cands=4, sim_thresh=0.70,
        min_full_matches=30, min_inliers=20,
        reproj_err=3.0, max_t=50.0)
    print(f"[LC] t={time.time()-t0:.1f}s", flush=True)

    if not loop_edges:
        print("[COR] No loops — using front-end poses.", flush=True)
        final = poses
    else:
        corrected = [T.copy() for T in poses]

        # ── Step 1: Main rubber-band (highest inlier loop) ─────────────────
        best = max(loop_edges, key=lambda e: e[3])
        bi, bj, T_best, n_best = best
        print(f"[COR] Main rubber-band: {bj}<->{bi}  inl={n_best}", flush=True)
        corrected = rubber_band_left(corrected, bi, bj, T_best, n, full_tail=True)

        # ── Step 2: Direct tail correction ────────────────────────────────────
        # Each secondary loop (i, j) has i in the tail (> bi) and j in the
        # main segment (≤ bi).  For each tail frame k, pick the single best
        # loop (highest inliers) and set corrected[k] DIRECTLY to the target
        # position corrected[j] @ inv(T_meas).  This avoids xi-averaging
        # artefacts for large corrections.  Frames with no direct measurement
        # are filled by VO-propagation from the previous (already-corrected)
        # tail frame.
        seg_anchor = bi
        sec_loops = [(i, j, Tm, ni) for (i, j, Tm, ni) in loop_edges
                     if i > seg_anchor and j <= seg_anchor]

        # Build best-loop-per-tail-frame map
        best_for_k = {}   # k → (j, T_meas, n_inl)
        for (i, j, T_meas, n_inl) in sec_loops:
            if i not in best_for_k or n_inl > best_for_k[i][2]:
                best_for_k[i] = (j, T_meas, n_inl)

        n_direct = len(best_for_k)
        n_tail   = n - seg_anchor - 1
        print(f"[COR] Tail correction: {n_direct}/{n_tail} frames direct, "
              f"{n_tail - n_direct} VO-propagated", flush=True)

        # Single forward pass: direct set or VO-propagate
        for k in range(seg_anchor + 1, n):
            if k in best_for_k:
                j_b, T_b, _ = best_for_k[k]
                # corrected[k] := corrected[j_b] @ inv(T_meas)
                corrected[k] = corrected[j_b] @ se3_inv(T_b)
            else:
                # Propagate from previous frame via original VO relative step
                VO_step = T_rel(poses[k - 1], poses[k])   # locally accurate
                corrected[k] = corrected[k - 1] @ VO_step

        # Measure residual after tail correction
        err_tail = 0.0
        for (i, j, T_meas, n_inl) in sec_loops:
            T_err = se3_inv(T_meas) @ T_rel(corrected[i], corrected[j])
            err_tail = max(err_tail, float(np.linalg.norm(mat_to_xi(T_err))))
        print(f"  [TC] max_err after tail correction = {err_tail:.4f}", flush=True)

        # ── Step 3: Light Jacobi for residual smoothing ───────────────────
        print("[COR] Light Jacobi …", flush=True)
        corrected = jacobi_left(corrected, loop_edges, n_iter=5)

        print(f"[COR] t={time.time()-t0:.1f}s", flush=True)

        # Sanity: trajectory length ratio
        def tlen(ps):
            pts = np.array([p[:3, 3] for p in ps])
            return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

        r_ratio = tlen(corrected) / (tlen(poses) + 1e-6)
        print(f"[COR] Traj length ratio = {r_ratio:.4f}", flush=True)

        if 0.8 < r_ratio < 1.25:
            final = corrected
        else:
            print("[COR] Sanity FAILED — reverting to front-end.", flush=True)
            final = poses

    traj_out  = np.array([p[:3, 3]      for p in final])
    poses_out = np.array([p[:3, :].ravel() for p in final])
    np.savetxt(os.path.join(artifacts_dir, 'traj.txt'),  traj_out,  fmt='%.6f')
    np.savetxt(os.path.join(artifacts_dir, 'poses.txt'), poses_out, fmt='%.8e')
    print(f"[DONE] {len(traj_out)} frames.  t={time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
