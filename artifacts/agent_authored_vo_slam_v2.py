"""
RGB-D SLAM with Loop Closure
────────────────────────────
VO front-end: ORB features + PnP-RANSAC (nfeatures=500 for speed)
Loop closure: brute-force descriptor matching + PnP verification
Pose graph:   TRANSLATION-ONLY linear system (rotations fixed from VO)
              → sparse LSQR, guaranteed no divergence, < 1 s
Fallback:     VO-only if no LC detected or system is ill-conditioned
"""
import numpy as np
import cv2
import os
import glob
from scipy.sparse     import lil_matrix
from scipy.sparse.linalg import lsqr as sp_lsqr

# ──────────── SE(3) helpers ────────────
def invert_T(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def T_from_Rt(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t).ravel()
    return T

def rel_T(Ti, Tj):
    return invert_T(Ti) @ Tj

# ──────────── feature extraction ────────────
def extract_kps(orb, img, dep, fx, fy, cx, cy, depth_scale,
                min_d=0.15, max_d=7.0):
    """Detect ORB keypoints; keep only those with valid depth."""
    kps, des = orb.detectAndCompute(img, None)
    if not kps or des is None:
        return [], None, np.zeros((0, 3), np.float32)
    h, w = dep.shape
    vidx, pts3d = [], []
    for j, kp in enumerate(kps):
        u = int(round(kp.pt[0])); v = int(round(kp.pt[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        d = dep[v, u] / depth_scale
        if d < min_d or d > max_d:
            continue
        pts3d.append([(u - cx) * d / fx, (v - cy) * d / fy, d])
        vidx.append(j)
    if not vidx:
        return [], None, np.zeros((0, 3), np.float32)
    vi = np.array(vidx, np.int32)
    return [kps[j] for j in vi], des[vi], np.array(pts3d, np.float32)

# ──────────── PnP relative pose ────────────
def pnp_rel_pose(pts3d_src, kps_dst, des_src, des_dst,
                 K, matcher, min_inliers=8, reproj_err=4.0):
    """
    3D(src) → 2D(dst) PnP.  Returns (T_rel, n_inliers) s.t.
    T_dst = T_src @ T_rel, or (None, 0) on failure.
    """
    if des_src is None or des_dst is None:
        return None, 0
    if len(des_src) < min_inliers or len(des_dst) < min_inliers:
        return None, 0
    try:
        pairs = matcher.knnMatch(des_src, des_dst, k=2)
    except Exception:
        return None, 0
    good = [m for p in pairs if len(p) == 2
            for m, n in [(p[0], p[1])] if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, 0
    obj = np.array([pts3d_src[m.queryIdx] for m in good], np.float32)
    img = np.array([kps_dst[m.trainIdx].pt  for m in good], np.float32)
    ok_mask = obj[:, 2] > 0
    obj, img = obj[ok_mask], img[ok_mask]
    if len(obj) < min_inliers:
        return None, 0
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, None,
            iterationsCount=300, reprojectionError=reproj_err,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        return None, 0
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, 0
    R, _ = cv2.Rodrigues(rvec)
    # PnP: p_dst = R @ p_src + t  →  T_src_dst = [R|t]
    # T_dst = T_src @ inv(T_src_dst)
    return invert_T(T_from_Rt(R, tvec.ravel())), len(inliers)

# ──────────── linear pose-graph solve ────────────
def solve_translation_graph(edges, kf_init, n_kf):
    """
    Fix rotations from kf_init; solve for translations (positions) only.
    Constraint per edge (ki, kj, T_meas, wt, wr):
        p_kj - p_ki  =  R_ki @ T_meas[:3,3]
    Variables: p_1 ... p_{n_kf-1}  (p_0 = 0 gauge-fixed).
    Returns optimised position array shape (n_kf, 3).
    """
    n_vars = (n_kf - 1) * 3
    n_cons = len(edges) * 3
    A = lil_matrix((n_cons, n_vars))
    b = np.zeros(n_cons)

    for eidx, (ki, kj, T_meas, wt, _wr) in enumerate(edges):
        R_ki   = kf_init[ki][:3, :3]
        t_rel  = R_ki @ T_meas[:3, 3]     # relative translation in world frame
        row    = eidx * 3
        if ki > 0:
            ci = (ki - 1) * 3
            A[row,     ci]     = -wt
            A[row + 1, ci + 1] = -wt
            A[row + 2, ci + 2] = -wt
        if kj > 0:
            cj = (kj - 1) * 3
            A[row,     cj]     =  wt
            A[row + 1, cj + 1] =  wt
            A[row + 2, cj + 2] =  wt
        b[row:row + 3] = wt * t_rel

    Acsr = A.tocsr()
    res  = sp_lsqr(Acsr, b, atol=1e-10, btol=1e-10, iter_lim=2000)
    x    = res[0]

    p_opt = np.zeros((n_kf, 3))
    for k in range(1, n_kf):
        p_opt[k] = x[(k - 1) * 3: k * 3]
    return p_opt

# ──────────── main ────────────
def main():
    data_dir      = os.environ.get('LAB_DATA',      '/data')
    artifacts_dir = os.environ.get('LAB_ARTIFACTS', '/artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    vals = open(os.path.join(data_dir, 'intrinsics.txt')).read().split()
    fx, fy, cx, cy, depth_scale = [float(v) for v in vals]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    frames   = sorted(glob.glob(os.path.join(data_dir, 'frame_*.png')))
    n_frames = len(frames)
    print(f"[SLAM] {n_frames} frames,  depth_scale={depth_scale}")

    orb     = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ── 1. Feature extraction ──
    print("[SLAM] Extracting features...")
    kps_all = []; des_all = []; pts3d_all = []
    for i in range(n_frames):
        img = cv2.imread(os.path.join(data_dir, f'frame_{i:04d}.png'),
                         cv2.IMREAD_GRAYSCALE)
        dep = cv2.imread(os.path.join(data_dir, f'depth_{i:04d}.png'),
                         cv2.IMREAD_ANYDEPTH).astype(np.float32)
        kps, des, pts3d = extract_kps(orb, img, dep, fx, fy, cx, cy, depth_scale)
        kps_all.append(kps); des_all.append(des); pts3d_all.append(pts3d)
        if i % 100 == 0:
            print(f"  frame {i}: {len(kps)} valid kps")

    # ── 2. Frame-to-frame VO ──
    print("[SLAM] Running VO...")
    poses = [np.eye(4)]
    n_fail = 0
    for i in range(1, n_frames):
        T_rel, _ = pnp_rel_pose(
            pts3d_all[i-1], kps_all[i],
            des_all[i-1], des_all[i], K, matcher, min_inliers=8)
        if T_rel is not None:
            poses.append(poses[-1] @ T_rel)
        else:
            poses.append(poses[-1].copy())
            n_fail += 1

    traj_vo = np.array([p[:3, 3] for p in poses])
    vo_span = float(np.ptp(traj_vo, axis=0).max())
    vo_len  = float(np.sum(np.linalg.norm(np.diff(traj_vo, axis=0), axis=1)))
    print(f"[SLAM] VO: span={vo_span:.3f} m  path={vo_len:.3f} m  failures={n_fail}")

    # ── 3. Keyframe selection ──
    KF_TRANS, KF_ROT = 0.15, 0.15   # m, rad (~8.6°)
    kf_idx = [0]
    for i in range(1, n_frames):
        Tr    = rel_T(poses[kf_idx[-1]], poses[i])
        trans = np.linalg.norm(Tr[:3, 3])
        rot   = np.linalg.norm(cv2.Rodrigues(Tr[:3, :3].astype(np.float64))[0])
        if trans > KF_TRANS or rot > KF_ROT:
            kf_idx.append(i)
    if kf_idx[-1] != n_frames - 1:
        kf_idx.append(n_frames - 1)
    n_kf = len(kf_idx)
    print(f"[SLAM] {n_kf} keyframes")

    # ── 4. Sequential pose-graph edges ──
    SEQ_WT = 50.0
    LC_WT  = 10.0

    edges = []   # (ki, kj, T_meas, wt, wr)  ki→kj
    for k in range(n_kf - 1):
        fi, fj = kf_idx[k], kf_idx[k+1]
        edges.append((k, k+1, rel_T(poses[fi], poses[fj]), SEQ_WT, SEQ_WT))

    # ── 5. Loop closure detection ──
    print("[SLAM] Detecting loop closures...")
    MIN_GAP   = 15       # min keyframe index gap
    MAX_BACK  = n_kf     # search ALL prior keyframes (unlimited)
    MIN_INL   = 20       # inlier threshold (lowered slightly for better recall)
    LC_REPROJ = 3.5      # reprojection error threshold (px)
    MAX_DRIFT = max(2.0, 0.5 * vo_len)

    n_lc = 0
    for ki in range(MIN_GAP, n_kf):
        fi = kf_idx[ki]
        if des_all[fi] is None or len(des_all[fi]) < MIN_INL:
            continue
        lo = max(0, ki - MAX_BACK)
        hi = ki - MIN_GAP
        for kj in range(hi, lo - 1, -1):
            fj = kf_idx[kj]
            if des_all[fj] is None or len(des_all[fj]) < MIN_INL:
                continue

            T_lc, n_inl = pnp_rel_pose(
                pts3d_all[fi], kps_all[fj],
                des_all[fi], des_all[fj],
                K, matcher, min_inliers=MIN_INL, reproj_err=LC_REPROJ)
            if T_lc is None:
                continue

            # Sanity: LC relative translation vs VO accumulated
            T_vo_rel = rel_T(poses[fi], poses[fj])
            dt = np.linalg.norm(T_lc[:3, 3] - T_vo_rel[:3, 3])
            if dt > MAX_DRIFT:
                continue
            # Rotation consistency
            R_diff = T_lc[:3, :3].T @ T_vo_rel[:3, :3]
            dr = np.linalg.norm(cv2.Rodrigues(R_diff.astype(np.float64))[0])
            if dr > 1.5:          # >~86° → reject
                continue

            edges.append((ki, kj, T_lc, LC_WT, LC_WT))
            n_lc += 1
            if n_lc <= 20:
                print(f"  LC kf{ki}(f{fi})<->kf{kj}(f{fj}) "
                      f"inl={n_inl} dt={dt:.3f}m")

    print(f"[SLAM] {n_lc} loop closures. Total edges: {len(edges)}")

    # ── 6. Linear pose-graph optimisation (translation only) ──
    kf_init = [poses[kf_idx[k]].copy() for k in range(n_kf)]

    if n_lc > 0:
        print("[SLAM] Solving linear translation graph (sparse LSQR)...")
        p_opt = solve_translation_graph(edges, kf_init, n_kf)

        # Sanity check: RMS deviation from VO
        p_vo  = np.array([poses[kf_idx[k]][:3, 3] for k in range(n_kf)])
        rms   = float(np.sqrt(np.mean(np.sum((p_opt - p_vo)**2, axis=1))))
        print(f"[SLAM] RMS position shift from VO: {rms:.4f} m")

        if rms > max(5.0, 1.5 * vo_span):
            print("[SLAM] WARNING: shift too large → using VO")
            p_opt = p_vo

        # Build optimised keyframe poses (keep VO rotations, replace translations)
        kf_poses_opt = []
        for k in range(n_kf):
            T_opt = kf_init[k].copy()
            T_opt[:3, 3] = p_opt[k]
            kf_poses_opt.append(T_opt)
    else:
        print("[SLAM] No LC found → using VO")
        kf_poses_opt = kf_init

    # ── 7. Propagate to all frames ──
    all_poses = [None] * n_frames
    for k in range(n_kf):
        all_poses[kf_idx[k]] = kf_poses_opt[k]

    for k in range(n_kf - 1):
        fi      = kf_idx[k]
        fj      = kf_idx[k + 1]
        T_fi_opt = kf_poses_opt[k]
        T_fi_vo  = poses[fi]
        for f in range(fi + 1, fj):
            # Keep relative displacement from VO anchor
            all_poses[f] = T_fi_opt @ rel_T(T_fi_vo, poses[f])

    for i in range(n_frames):
        if all_poses[i] is None:
            all_poses[i] = poses[i]

    # ── 8. Write traj.txt ──
    out = os.path.join(artifacts_dir, 'traj.txt')
    with open(out, 'w') as fh:
        for T in all_poses:
            t = T[:3, 3]
            fh.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")

    traj = np.array([T[:3, 3] for T in all_poses])
    print(f"[SLAM] Wrote {n_frames} poses → {out}")
    print(f"[SLAM] span={np.ptp(traj, axis=0)}")
    print(f"[SLAM] path={np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)):.3f} m")

if __name__ == '__main__':
    main()
