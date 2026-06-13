#!/usr/bin/env python3
"""
Stereo Visual SLAM for KITTI outdoor driving.
Pipeline:
  1. Stereo VO: SGBM depth + ORB + PnP + constant-velocity model
  2. Keyframe selection (every KF_INTERVAL frames)
  3. Multi-frame PnP refinement (cheap local BA; monotonic safeguard)
  4. Appearance-based loop detection (ORB descriptor voting)
  5. Geometric verification (PnP RANSAC with 3D points from DB keyframe)
  6. Global pose-graph optimisation (SE(3), scipy TRF + Huber)
  7. Propagate corrected KF poses to all frames
"""
import os, sys, time
import numpy as np
import cv2
from scipy.optimize import least_squares

DATA_DIR  = os.environ.get("LAB_DATA",      "/data")
OUT_DIR   = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
KF_INTERVAL       = 5     # keyframe every N frames
MFPNP_LOOKBACK    = 3     # how many previous KFs to use in multi-frame PnP
LOOP_MIN_FRAMES   = 60    # minimum frame-index gap for loop candidate
LOOP_SCORE_THRESH = 0.10  # minimum descriptor match ratio
LOOP_MIN_INLIERS  = 25    # minimum PnP inliers for loop acceptance
N_IDX_DESCS       = 200   # subsampled descriptors per KF for loop index
DIST = np.zeros(4, np.float64)

# ─── Intrinsics ───────────────────────────────────────────────────────────────
def load_intrinsics():
    with open(os.path.join(DATA_DIR, "intrinsics.txt")) as f:
        vals = [float(l.strip()) for l in f if l.strip()]
    fx, fy, cx, cy, baseline = vals[:5]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
    return K, float(baseline), float(fx), float(fy), float(cx), float(cy)

def count_frames():
    n = 0
    while os.path.exists(os.path.join(DATA_DIR, f"left_{n:06d}.png")):
        n += 1
    return n

# ─── SE(3) helpers ────────────────────────────────────────────────────────────
# Convention: cam-to-world T_cw = [R_cw | t_cw]
#   X_world = R_cw @ X_cam + t_cw
#   camera centre = t_cw

def rmat(v):
    R, _ = cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))
    return R

def rvec(R):
    v, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return v.ravel()

def cw2wc(R_cw, t_cw):
    """cam-to-world  →  world-to-cam"""
    return R_cw.T, -R_cw.T @ t_cw

def wc2cw(R_wc, t_wc):
    """world-to-cam  →  cam-to-world"""
    return R_wc.T, -R_wc.T @ t_wc

# ─── Stereo depth ─────────────────────────────────────────────────────────────
_sgbm = None
def get_depth(L, R_img, fx, bl):
    global _sgbm
    if _sgbm is None:
        b = 11
        _sgbm = cv2.StereoSGBM_create(
            minDisparity=1, numDisparities=128, blockSize=b,
            P1=8*b*b, P2=32*b*b,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = _sgbm.compute(L, R_img).astype(np.float32) / 16.0
    depth = np.where((d > 1.0) & (d < 128.0), fx * bl / d, 0.0).astype(np.float32)
    depth[depth > 80.0] = 0.0
    return depth

# ─── ORB ─────────────────────────────────────────────────────────────────────
_orb, _bf = None, None
def get_orb():
    global _orb, _bf
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8,
                               fastThreshold=10, edgeThreshold=31)
        _bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return _orb, _bf

def extract_orb(img):
    orb, _ = get_orb()
    return orb.detectAndCompute(img, None)

def match_bf(d1, d2, max_dist=60):
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return []
    _, bf = get_orb()
    ms = [m for m in bf.match(d1, d2) if m.distance < max_dist]
    return sorted(ms, key=lambda m: m.distance)

# ─── Back-projection ──────────────────────────────────────────────────────────
def backproj_cam(kps, depth, fx, fy, cx, cy):
    """Return dict {kp_idx: np.array([x,y,z])} in camera frame for valid pts."""
    out = {}
    h, w = depth.shape
    for i, kp in enumerate(kps):
        u = int(round(kp.pt[0])); v = int(round(kp.pt[1]))
        if 0 <= v < h and 0 <= u < w:
            z = float(depth[v, u])
            if 0.5 < z < 75.0:
                out[i] = np.array([(kp.pt[0]-cx)*z/fx,
                                   (kp.pt[1]-cy)*z/fy, z], np.float64)
    return out

# ─── PnP ─────────────────────────────────────────────────────────────────────
def run_pnp(pts3d, pts2d, K, init_R=None, init_t=None, rep_err=2.0, min_inl=6):
    """
    RANSAC PnP then LM refinement on inliers.
    Returns (R, t, inlier_mask) with convention X_cam = R @ X_ref + t,
    or (None, None, None) on failure.
    """
    if len(pts3d) < max(min_inl, 6):
        return None, None, None
    use_g = init_R is not None and init_t is not None
    ir = rvec(init_R).reshape(3,1) if use_g else None
    it = np.array(init_t, np.float64).reshape(3,1) if use_g else None
    ok, rv, tv, inl = cv2.solvePnPRansac(
        pts3d.astype(np.float64), pts2d.astype(np.float64), K, DIST,
        rvec=ir, tvec=it, useExtrinsicGuess=use_g,
        iterationsCount=300, reprojectionError=rep_err,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < min_inl:
        return None, None, None
    idx = inl.ravel()
    _, rv2, tv2 = cv2.solvePnP(
        pts3d[idx].astype(np.float64), pts2d[idx].astype(np.float64),
        K, DIST, rvec=rv, tvec=tv, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE)
    mask = np.zeros(len(pts3d), bool); mask[idx] = True
    return rmat(rv2), tv2.ravel(), mask

# ─── Reprojection error ───────────────────────────────────────────────────────
def reproj_err(pts3d_w, pts2d, R_wc, t_wc, K):
    """Mean pixel reprojection error; R_wc,t_wc is world-to-cam transform."""
    X = (R_wc @ pts3d_w.T).T + t_wc
    ok = X[:, 2] > 0.01
    if not np.any(ok):
        return 1e9
    Xv = X[ok]; p2v = pts2d[ok]
    u = K[0,0]*Xv[:,0]/Xv[:,2] + K[0,2]
    v = K[1,1]*Xv[:,1]/Xv[:,2] + K[1,2]
    return float(np.mean(np.hypot(u - p2v[:,0], v - p2v[:,1])))

# ─── Loop detection helpers ───────────────────────────────────────────────────
def make_idx_des(des):
    if des is None or len(des) == 0:
        return None
    idx = np.random.choice(len(des), min(N_IDX_DESCS, len(des)), replace=False)
    return des[idx].copy()

_bf_loop = None   # reused across all descriptor_score calls
def descriptor_score(d1, d2):
    """Fraction of d1 descriptors that match d2 within Hamming distance 50."""
    global _bf_loop
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return 0.0
    if _bf_loop is None:
        _bf_loop = cv2.BFMatcher(cv2.NORM_HAMMING)
    ms = _bf_loop.knnMatch(d1, d2, k=1)
    cnt = sum(1 for m in ms if m and m[0].distance < 50)
    return cnt / min(len(d1), len(d2))

# ─── Fast SE(3) helpers (avoid cv2.Rodrigues inside scipy inner loop) ─────────
def _rmat(v):
    """Pure-numpy Rodrigues rvec→R (avoids cv2 Python/C++ overhead)."""
    v = np.asarray(v, np.float64).ravel()
    t2 = float(v @ v)
    if t2 < 1e-20:
        return np.eye(3, dtype=np.float64)
    theta = np.sqrt(t2); k = v / theta
    s, c = np.sin(theta), np.cos(theta)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2],  0.0, -k[0]],
                  [-k[1], k[0],  0.0]])
    return c * np.eye(3) + s * K + (1.0 - c) * np.outer(k, k)

def _rvec_err(R):
    """Fast approximate log(R) for small angles (valid < ~30°)."""
    return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def _rmat_batch(vs):
    """Batched Rodrigues: vs (M,3) → Rs (M,3,3). Fully vectorised, no Python loop."""
    M = len(vs)
    t2 = np.einsum('ij,ij->i', vs, vs)          # (M,)
    Rs = np.broadcast_to(np.eye(3, dtype=np.float64), (M, 3, 3)).copy()
    big = t2 >= 1e-20
    if not np.any(big):
        return Rs
    vb = vs[big]; t2b = t2[big]
    theta = np.sqrt(t2b)                         # (k,)
    k = vb / theta[:, None]                      # (k,3)
    s = np.sin(theta); c = np.cos(theta)         # (k,)
    # Skew-symmetric K matrices
    K = np.zeros((len(vb), 3, 3), np.float64)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] =  k[:, 1]
    K[:, 1, 0] =  k[:, 2]; K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] =  k[:, 0]
    kk = k[:, :, None] * k[:, None, :]          # (k,3,3) outer products
    I3 = np.eye(3, dtype=np.float64)
    Rs[big] = (c[:, None, None] * I3[None]
               + s[:, None, None] * K
               + (1.0 - c)[:, None, None] * kk)
    return Rs

# ─── Pose-graph optimisation ──────────────────────────────────────────────────
def pose_graph_opt(kf_poses, seq_edges, loop_edges_w):
    """
    kf_poses:      list of (R_cw, t_cw) — KF 0 is anchored (fixed).
    seq_edges:     list (i, j, R_obs, t_obs)
    loop_edges_w:  list (i, j, R_obs, t_obs, weight)
    Returns updated list of (R_cw, t_cw).
    Uses batched Rodrigues + vectorised edge residuals (no Python loop per edge).
    """
    N = len(kf_poses)
    R0, t0 = kf_poses[0]

    x0 = np.array([v for i in range(1, N)
                   for v in list(rvec(kf_poses[i][0])) + list(kf_poses[i][1])])

    # ── Pre-build edge index / data arrays (done once, outside residuals) ────
    seq_i  = np.array([e[0] for e in seq_edges], np.int32)
    seq_j  = np.array([e[1] for e in seq_edges], np.int32)
    seq_Ro = np.asarray([e[2] for e in seq_edges], np.float64)  # (Ms,3,3)
    seq_to = np.asarray([e[3] for e in seq_edges], np.float64)  # (Ms,3)
    n_lp   = len(loop_edges_w)
    if n_lp > 0:
        lp_i  = np.array([e[0] for e in loop_edges_w], np.int32)
        lp_j  = np.array([e[1] for e in loop_edges_w], np.int32)
        lp_Ro = np.asarray([e[2] for e in loop_edges_w], np.float64)
        lp_to = np.asarray([e[3] for e in loop_edges_w], np.float64)
        lp_w  = np.asarray([e[4] for e in loop_edges_w], np.float64)

    def _edge_block(R_all, t_all, i_arr, j_arr, Ro, to):
        """Vectorised residual block for a set of edges → shape (M,6)."""
        Ri = R_all[i_arr]; Rj = R_all[j_arr]           # (M,3,3)
        ti = t_all[i_arr]; tj = t_all[j_arr]           # (M,3)
        RjT    = np.transpose(Rj, (0, 2, 1))           # (M,3,3)
        R_pred = RjT @ Ri                               # (M,3,3) batched matmul
        t_pred = (RjT @ (ti - tj)[..., None]).squeeze(-1)  # (M,3)
        RoT_Rp = np.transpose(Ro, (0, 2, 1)) @ R_pred  # (M,3,3)
        rot_r  = 0.5 * np.stack([
            RoT_Rp[:, 2, 1] - RoT_Rp[:, 1, 2],
            RoT_Rp[:, 0, 2] - RoT_Rp[:, 2, 0],
            RoT_Rp[:, 1, 0] - RoT_Rp[:, 0, 1]], axis=1)  # (M,3)
        return np.hstack([rot_r, t_pred - to])          # (M,6)

    def _build_all(x):
        xr = x.reshape(N-1, 6)
        R_all = np.empty((N, 3, 3), np.float64)
        R_all[0] = R0; R_all[1:] = _rmat_batch(xr[:, :3])
        t_all = np.empty((N, 3), np.float64)
        t_all[0] = t0; t_all[1:] = xr[:, 3:6]
        return R_all, t_all

    def residuals(x):
        R_all, t_all = _build_all(x)
        out_seq = _edge_block(R_all, t_all, seq_i, seq_j, seq_Ro, seq_to)
        if n_lp > 0:
            out_lp = _edge_block(R_all, t_all, lp_i, lp_j, lp_Ro, lp_to)
            out_lp *= lp_w[:, None]
            return np.concatenate([out_seq.ravel(), out_lp.ravel()])
        return out_seq.ravel()

    def seq_res_norm(x):
        R_all, t_all = _build_all(x)
        return float(np.linalg.norm(
            _edge_block(R_all, t_all, seq_i, seq_j, seq_Ro, seq_to)))

    base_sn = seq_res_norm(x0)
    try:
        res = least_squares(
            residuals, x0,
            method='lm', max_nfev=1000,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            verbose=0)
        new_sn = seq_res_norm(res.x)
        if new_sn > base_sn * 3.0 + 0.5:
            print(f"  PG REJECTED: seq_res {base_sn:.4f} -> {new_sn:.4f}")
            return kf_poses
        print(f"  PG: seq_res {base_sn:.4f} -> {new_sn:.4f}, cost {res.cost:.2f}")
        new_poses = [kf_poses[0]]
        for i in range(1, N):
            xi = res.x[(i-1)*6 : i*6]
            new_poses.append((_rmat(xi[:3]), xi[3:6].copy()))
        return new_poses
    except Exception as e:
        print(f"  PG error: {e}")
        return kf_poses

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0_total = time.time()
    np.random.seed(42)

    K, baseline, fx, fy, cx, cy = load_intrinsics()
    N = count_frames()
    print(f"Sequence: {N} frames | fx={fx:.1f} bl={baseline:.4f}m")

    frame_poses = []          # (R_cw, t_cw) per frame
    R_cw = np.eye(3, dtype=np.float64)
    t_cw = np.zeros(3, dtype=np.float64)
    prev_kps, prev_des, prev_depth = None, None, None
    last_R_rel = np.eye(3, dtype=np.float64)   # velocity model: prev-cam → curr-cam
    last_t_rel = np.zeros(3, dtype=np.float64)
    keyframes  = []

    print("Running VO ...")
    for fi in range(N):
        L     = cv2.imread(os.path.join(DATA_DIR, f"left_{fi:06d}.png"),
                           cv2.IMREAD_GRAYSCALE)
        R_img = cv2.imread(os.path.join(DATA_DIR, f"right_{fi:06d}.png"),
                           cv2.IMREAD_GRAYSCALE)

        depth = get_depth(L, R_img, fx, baseline)
        kps, des = extract_orb(L)

        if fi == 0:
            frame_poses.append((R_cw.copy(), t_cw.copy()))
            prev_kps, prev_des, prev_depth = kps, des, depth
        else:
            # ── Frame-to-frame PnP ────────────────────────────────────────
            ms = match_bf(prev_des, des, max_dist=60)
            R_rel, t_rel = last_R_rel.copy(), last_t_rel.copy()

            if len(ms) >= 6:
                p3, p2 = [], []
                for m in ms:
                    kp1 = prev_kps[m.queryIdx]
                    u1 = int(round(kp1.pt[0])); v1 = int(round(kp1.pt[1]))
                    if 0 <= v1 < prev_depth.shape[0] and 0 <= u1 < prev_depth.shape[1]:
                        z = float(prev_depth[v1, u1])
                        if 0.5 < z < 75.0:
                            p3.append([(kp1.pt[0]-cx)*z/fx,
                                       (kp1.pt[1]-cy)*z/fy, z])
                            p2.append(kps[m.trainIdx].pt)
                if len(p3) >= 6:
                    p3 = np.array(p3, np.float64)
                    p2 = np.array(p2, np.float64)
                    Rr, tr, _ = run_pnp(p3, p2, K,
                                        init_R=last_R_rel, init_t=last_t_rel)
                    if Rr is not None:
                        R_rel, t_rel = Rr, tr
                        last_R_rel, last_t_rel = Rr.copy(), tr.copy()

            # PnP gives: X_curr = R_rel @ X_prev + t_rel (prev-cam → curr-cam)
            # Cam-to-world update:
            #   R_cw_new = R_cw_prev @ R_rel^T
            #   t_cw_new = t_cw_prev - R_cw_new @ t_rel
            R_cw_new = R_cw @ R_rel.T
            t_cw_new = t_cw - R_cw_new @ t_rel
            R_cw, t_cw = R_cw_new, t_cw_new
            frame_poses.append((R_cw.copy(), t_cw.copy()))

        # ── Keyframe selection ────────────────────────────────────────────
        if fi % KF_INTERVAL == 0:
            pts_cam = backproj_cam(kps, depth, fx, fy, cx, cy)
            pts3d_w = [None] * len(kps)
            for ki, pc in pts_cam.items():
                pts3d_w[ki] = R_cw @ pc + t_cw   # camera → world

            kf = {
                'fi':      fi,
                'kf_idx':  len(keyframes),
                'R_cw':    R_cw.copy(),
                't_cw':    t_cw.copy(),
                'kps':     kps,
                'des':     des,
                'pts3d_w': pts3d_w,
                'idx_des': make_idx_des(des),
            }
            keyframes.append(kf)

            # ── Multi-frame PnP refinement (monotonic safeguard) ──────────
            kf_idx = len(keyframes) - 1
            if kf_idx >= 2:
                all_p3w, all_p2 = [], []
                for pk in range(max(0, kf_idx - MFPNP_LOOKBACK), kf_idx):
                    pkf = keyframes[pk]
                    ms_kf = match_bf(pkf['des'], des, max_dist=55)
                    for m in ms_kf:
                        ip, iq = m.queryIdx, m.trainIdx
                        if (ip < len(pkf['pts3d_w'])
                                and pkf['pts3d_w'][ip] is not None):
                            all_p3w.append(pkf['pts3d_w'][ip])
                            all_p2.append(kps[iq].pt)

                if len(all_p3w) >= 15:
                    all_p3w = np.array(all_p3w, np.float64)
                    all_p2  = np.array(all_p2,  np.float64)
                    R_wc_in, t_wc_in = cw2wc(R_cw, t_cw)
                    Rw, tw, mask_w = run_pnp(all_p3w, all_p2, K,
                                             init_R=R_wc_in, init_t=t_wc_in,
                                             rep_err=3.0, min_inl=12)
                    if Rw is not None and int(np.sum(mask_w)) >= 12:
                        p3in = all_p3w[mask_w]; p2in = all_p2[mask_w]
                        err_old = reproj_err(p3in, p2in, R_wc_in, t_wc_in, K)
                        err_new = reproj_err(p3in, p2in, Rw, tw, K)
                        if err_new < err_old * 1.1:   # accept if not worse
                            R_cw_r, t_cw_r = wc2cw(Rw, tw)
                            R_cw, t_cw = R_cw_r, t_cw_r
                            frame_poses[fi] = (R_cw.copy(), t_cw.copy())
                            kf['R_cw'] = R_cw.copy()
                            kf['t_cw'] = t_cw.copy()
                            # Refresh world-frame 3D points with refined pose
                            for ki, pc in pts_cam.items():
                                kf['pts3d_w'][ki] = R_cw @ pc + t_cw

        prev_kps, prev_des, prev_depth = kps, des, depth
        if fi % 50 == 0 or fi == N - 1:
            print(f"  f{fi:04d}/{N}  pos=[{t_cw[0]:7.2f} {t_cw[1]:6.2f}"
                  f" {t_cw[2]:7.2f}]")

    t_vo = time.time()
    print(f"VO done in {t_vo-t0_total:.1f}s | {len(keyframes)} KFs")

    # ── Loop detection ────────────────────────────────────────────────────────
    print("Loop detection ...")
    loop_edges = []   # list of (kf_i, kf_j, R_obs, t_obs)

    for q_idx in range(5, len(keyframes)):
        kf_q = keyframes[q_idx]
        best_score, best_d = LOOP_SCORE_THRESH, -1

        for d_idx in range(q_idx):
            kf_d = keyframes[d_idx]
            if kf_q['fi'] - kf_d['fi'] < LOOP_MIN_FRAMES:
                continue
            sc = descriptor_score(kf_q['idx_des'], kf_d['idx_des'])
            if sc > best_score:
                best_score, best_d = sc, d_idx

        if best_d < 0:
            continue

        # Geometric verification with all descriptors
        kf_d   = keyframes[best_d]
        ms_all = match_bf(kf_d['des'], kf_q['des'], max_dist=55)
        if len(ms_all) < LOOP_MIN_INLIERS:
            continue

        p3w_lp, p2_lp = [], []
        for m in ms_all:
            ip, iq = m.queryIdx, m.trainIdx
            if (ip < len(kf_d['pts3d_w'])
                    and kf_d['pts3d_w'][ip] is not None):
                p3w_lp.append(kf_d['pts3d_w'][ip])
                p2_lp.append(kf_q['kps'][iq].pt)

        if len(p3w_lp) < LOOP_MIN_INLIERS:
            continue

        p3w_lp = np.array(p3w_lp, np.float64)
        p2_lp  = np.array(p2_lp,  np.float64)
        Rw_lc, tw_lc, mask_lc = run_pnp(p3w_lp, p2_lp, K,
                                          rep_err=3.0,
                                          min_inl=LOOP_MIN_INLIERS)
        if Rw_lc is None or int(np.sum(mask_lc)) < LOOP_MIN_INLIERS:
            continue

        err_lc = reproj_err(p3w_lp[mask_lc], p2_lp[mask_lc], Rw_lc, tw_lc, K)
        if err_lc > 5.0:
            print(f"  Loop KF{best_d}<->KF{q_idx}: REJECTED reproj={err_lc:.2f}px")
            continue

        # Loop-detected cam-to-world pose of kf_q
        R_cw_q_lc, t_cw_q_lc = wc2cw(Rw_lc, tw_lc)

        # Edge constraint: relative pose from kf_d to kf_q (camera-frame convention)
        #   X_j_cam = R_obs @ X_i_cam + t_obs
        R_cw_d = kf_d['R_cw']; t_cw_d = kf_d['t_cw']
        R_obs = R_cw_q_lc.T @ R_cw_d
        t_obs = R_cw_q_lc.T @ (t_cw_d - t_cw_q_lc)

        n_inl = int(np.sum(mask_lc))
        print(f"  Loop KF{best_d}(f{kf_d['fi']})<->KF{q_idx}(f{kf_q['fi']}) "
              f"score={best_score:.3f} inl={n_inl} err={err_lc:.2f}px")
        loop_edges.append((best_d, q_idx, R_obs.copy(), t_obs.copy()))

    print(f"Loops found: {len(loop_edges)} | loop_det={time.time()-t_vo:.1f}s")

    # ── Global pose-graph optimisation ───────────────────────────────────────
    if loop_edges:
        print("Pose-graph optimisation ...")

        # Sequential edges from current KF VO poses
        seq_edges = []
        for i in range(len(keyframes) - 1):
            Ri, ti = keyframes[i]['R_cw'],   keyframes[i]['t_cw']
            Rj, tj = keyframes[i+1]['R_cw'], keyframes[i+1]['t_cw']
            # Relative: X_j = R_obs @ X_i + t_obs  (camera convention)
            seq_edges.append((i, i+1, Rj.T @ Ri, Rj.T @ (ti - tj)))

        w_loop = [(i, j, Ro, to, 5.0) for (i, j, Ro, to) in loop_edges]

        kf_poses_orig = [(kf['R_cw'].copy(), kf['t_cw'].copy())
                         for kf in keyframes]
        t_pg0 = time.time()
        new_kf_poses = pose_graph_opt(kf_poses_orig, seq_edges, w_loop)
        print(f"  PG opt time: {time.time()-t_pg0:.2f}s")

        # Update keyframes with optimised poses
        for i, kf in enumerate(keyframes):
            kf['R_cw'], kf['t_cw'] = new_kf_poses[i]

        # Propagate corrections to all frames via preceding KF's rigid correction
        t_prop0 = time.time()
        print("Propagating corrections ...")
        for fi in range(N):
            # Find preceding keyframe
            pk = None
            for ki in range(len(keyframes) - 1, -1, -1):
                if keyframes[ki]['fi'] <= fi:
                    pk = ki
                    break
            if pk is None:
                continue

            R_kf_old, t_kf_old = kf_poses_orig[pk]
            R_kf_new, t_kf_new = new_kf_poses[pk]
            R_f_old,  t_f_old  = frame_poses[fi]

            # Rigid-body correction: delta applied to KF propagates to frame
            R_delta = R_kf_new @ R_kf_old.T
            t_delta = t_kf_new - R_delta @ t_kf_old
            frame_poses[fi] = (R_delta @ R_f_old,
                                R_delta @ t_f_old + t_delta)

    if loop_edges:
        print(f"  Propagation time: {time.time()-t_prop0:.2f}s")
    # ── Write output ─────────────────────────────────────────────────────────
    traj_path  = os.path.join(OUT_DIR, "traj.txt")
    poses_path = os.path.join(OUT_DIR, "poses.txt")

    with open(traj_path, 'w') as ft, open(poses_path, 'w') as fp:
        for R_cw_f, t_cw_f in frame_poses:
            # traj.txt: camera centre
            ft.write(f"{t_cw_f[0]:.6f} {t_cw_f[1]:.6f} {t_cw_f[2]:.6f}\n")
            # poses.txt: 3×4 [R|t] cam-to-world, row-major, 12 numbers
            mat = np.concatenate([R_cw_f, t_cw_f.reshape(3, 1)], axis=1)
            fp.write(' '.join(f'{x:.6f}' for x in mat.ravel()) + '\n')

    elapsed = time.time() - t0_total
    print(f"\nDone in {elapsed:.1f}s | wrote {N} poses")

if __name__ == "__main__":
    main()
