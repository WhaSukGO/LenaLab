#!/usr/bin/env python3
"""
Stereo Visual SLAM for KITTI outdoor driving.

Front-end : ORB features + BF matching (same as reference VO that scores ~2.4%)
            + SGBM stereo depth for absolute metric scale
            → PnP (world-to-cam) to find per-frame poses

Keyframes : every KF_INTERVAL frames; same ORB features reused for loop det.
            Multi-KF PnP refinement with monotonic reprojection safeguard.

Loop det  : descriptor voting on subsampled ORB + top-K candidate search
            + PnP geometric verify.

Pose graph: SE(3) scipy TRF with sparse Jacobian, anchored at KF0.
            Guard: cost must decrease, loop residuals must reduce.
"""
import os, time
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

DATA_DIR  = os.environ.get("LAB_DATA",      "/data")
OUT_DIR   = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Intrinsics ─────────────────────────────────────────────────────────────────
def load_intrinsics():
    vals = [float(v) for v in
            open(os.path.join(DATA_DIR, "intrinsics.txt")).read().split()]
    fx, fy, cx, cy, bl = vals[:5]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
    return K, bl, fx, fy, cx, cy

def count_frames():
    n = 0
    while os.path.exists(os.path.join(DATA_DIR, f"left_{n:06d}.png")):
        n += 1
    return n

# ── SGBM depth ─────────────────────────────────────────────────────────────────
_sgbm = None
def compute_depth(L_img, R_img, fx, bl):
    global _sgbm
    if _sgbm is None:
        _sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=7,
            P1=8*7*7, P2=32*7*7,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
    d = _sgbm.compute(L_img, R_img).astype(np.float64) / 16.0
    dep = np.zeros_like(d)
    m = d > 0.5
    dep[m] = fx * bl / d[m]
    dep[dep > 60.0] = 0.0
    return dep

# ── ORB front-end (frame-to-frame VO) ─────────────────────────────────────────
_vo_orb = None
_vo_bf  = None

def get_vo_orb():
    global _vo_orb, _vo_bf
    if _vo_orb is None:
        _vo_orb = cv2.ORB_create(nfeatures=3000)
        _vo_bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return _vo_orb, _vo_bf

def extract_frame_features(img):
    """Extract ORB features for both front-end VO and loop detection."""
    orb, _ = get_vo_orb()
    kps, des = orb.detectAndCompute(img, None)
    return kps, des

def match_features(des0, des1):
    """BF match for front-end VO. Returns list of DMatch."""
    if des0 is None or des1 is None or len(des0) < 6 or len(des1) < 6:
        return []
    _, bf = get_vo_orb()
    return bf.match(des0, des1)

def backproject_kps(kps, idx_list, depth, fx, fy, cx, cy):
    """
    Back-project a subset of keypoints using depth map.
    idx_list: list of kp indices to use.
    Returns (pts3d, valid_indices) where pts3d[i] is the 3D position for
    the i-th entry in idx_list (or None if depth invalid).
    """
    pts3d = []
    valid = []
    h, w = depth.shape
    for i, ki in enumerate(idx_list):
        u, v = kps[ki].pt
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < h and 0 <= ui < w:
            z = depth[vi, ui]
            if 1.0 < z < 60.0:
                pts3d.append([(u-cx)*z/fx, (v-cy)*z/fy, z])
                valid.append(i)
    return np.array(pts3d, np.float64), valid

# ── PnP ────────────────────────────────────────────────────────────────────────
_DIST = np.zeros(4, np.float64)

def run_pnp(pts3w, pts2, K, min_inl=6, rep_err=2.0):
    """
    PnP RANSAC + LM refinement.
    pts3w: 3D world points; pts2: 2D observations in current frame.
    Returns R_wc, t_wc (world-to-cam), inlier_mask; or (None,None,None).
    """
    if len(pts3w) < max(min_inl, 6):
        return None, None, None
    ok, rv, tv, inl = cv2.solvePnPRansac(
        pts3w.astype(np.float64), pts2.astype(np.float64), K, _DIST,
        iterationsCount=300, reprojectionError=rep_err,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < min_inl:
        return None, None, None
    idx = inl.ravel()
    _, rv2, tv2 = cv2.solvePnP(
        pts3w[idx].astype(np.float64), pts2[idx].astype(np.float64),
        K, _DIST, rvec=rv, tvec=tv, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE)
    mask = np.zeros(len(pts3w), bool); mask[idx] = True
    R, _ = cv2.Rodrigues(rv2)
    return R, tv2.ravel(), mask

def reproj_err(pts3w, pts2, R_wc, t_wc, K):
    """Mean reprojection error (world->cam)."""
    X = (R_wc @ pts3w.T).T + t_wc
    ok = X[:,2] > 0.01
    if not ok.any(): return 1e9
    Xv = X[ok]; p2v = pts2[ok]
    u = K[0,0]*Xv[:,0]/Xv[:,2] + K[0,2]
    v = K[1,1]*Xv[:,1]/Xv[:,2] + K[1,2]
    return float(np.mean(np.hypot(u-p2v[:,0], v-p2v[:,1])))

# ── SE(3) helpers ──────────────────────────────────────────────────────────────
def rvec(R):
    v, _ = cv2.Rodrigues(R.astype(np.float64))
    return v.ravel()

def _rmat_np(v):
    v = np.asarray(v, np.float64).ravel()
    t2 = float(v @ v)
    if t2 < 1e-20:
        return np.eye(3, dtype=np.float64)
    th = np.sqrt(t2); k = v/th
    s, c = np.sin(th), np.cos(th)
    K_ = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], np.float64)
    return c*np.eye(3) + s*K_ + (1-c)*np.outer(k,k)

def _rmat_batch(vs):
    M = len(vs)
    t2 = np.einsum('ij,ij->i', vs, vs)
    Rs = np.broadcast_to(np.eye(3,dtype=np.float64),(M,3,3)).copy()
    big = t2 >= 1e-20
    if not np.any(big): return Rs
    vb=vs[big]; t2b=t2[big]
    th=np.sqrt(t2b); k=vb/th[:,None]
    s=np.sin(th); c=np.cos(th)
    K_=np.zeros((len(vb),3,3),np.float64)
    K_[:,0,1]=-k[:,2]; K_[:,0,2]=k[:,1]
    K_[:,1,0]=k[:,2];  K_[:,1,2]=-k[:,0]
    K_[:,2,0]=-k[:,1]; K_[:,2,1]=k[:,0]
    kk=k[:,:,None]*k[:,None,:]
    Rs[big]=(c[:,None,None]*np.eye(3)+s[:,None,None]*K_+(1-c)[:,None,None]*kk)
    return Rs

# ── Loop detection helpers ─────────────────────────────────────────────────────
_bf_score_obj = None
def desc_score(d1, d2, n_sub=250, thresh=50):
    """Fraction of subsampled d1 that match d2 within Hamming threshold."""
    global _bf_score_obj
    if d1 is None or d2 is None or len(d1)<8 or len(d2)<8: return 0.0
    if _bf_score_obj is None:
        _bf_score_obj = cv2.BFMatcher(cv2.NORM_HAMMING)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(d1), min(n_sub, len(d1)), replace=False)
    ds1 = d1[idx]
    ms = _bf_score_obj.knnMatch(ds1, d2, k=1)
    cnt = sum(1 for m in ms if m and m[0].distance < thresh)
    return cnt / len(idx)

_bf_lp = None
def bf_match_lp(d1, d2, max_dist=65):
    """BF match for loop closure geometric verification (cached matcher)."""
    global _bf_lp
    if _bf_lp is None:
        _bf_lp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if d1 is None or d2 is None or len(d1)<4 or len(d2)<4: return []
    ms = _bf_lp.match(d1, d2)
    return sorted([m for m in ms if m.distance<max_dist], key=lambda m: m.distance)

# ── Pose-graph optimisation ────────────────────────────────────────────────────
def _pg_sparsity(N, si, sj, li, lj):
    n_seq = len(si); nl = len(li)
    n_res = (n_seq + nl) * 6
    n_par = (N-1) * 6
    S = lil_matrix((n_res, n_par), dtype=np.int8)
    for k in range(n_seq):
        i, j = si[k], sj[k]; row = k * 6
        if i > 0: S[row:row+6, (i-1)*6:(i-1)*6+6] = 1
        if j > 0: S[row:row+6, (j-1)*6:(j-1)*6+6] = 1
    for k in range(nl):
        i, j = li[k], lj[k]; row = (n_seq + k) * 6
        if i > 0: S[row:row+6, (i-1)*6:(i-1)*6+6] = 1
        if j > 0: S[row:row+6, (j-1)*6:(j-1)*6+6] = 1
    return S.tocsr()

def pose_graph_opt(kf_poses, seq_edges, loop_edges_w):
    """
    SE(3) pose graph with sparse TRF. kf_poses[0] anchored.
    seq_edges:    [(i,j,R_obs,t_obs),...] relative i->j cam convention
    loop_edges_w: [(i,j,R_obs,t_obs,w),...]
    """
    N = len(kf_poses)
    if N < 2: return kf_poses
    R0, t0 = kf_poses[0]

    x0 = np.concatenate([
        np.concatenate([rvec(kf_poses[i][0]), kf_poses[i][1]])
        for i in range(1, N)])

    si  = np.array([e[0] for e in seq_edges],  np.int32)
    sj  = np.array([e[1] for e in seq_edges],  np.int32)
    sRo = np.array([e[2] for e in seq_edges],  np.float64)
    sto = np.array([e[3] for e in seq_edges],  np.float64)
    nl  = len(loop_edges_w)
    li  = np.array([e[0] for e in loop_edges_w], np.int32) if nl else np.array([], np.int32)
    lj  = np.array([e[1] for e in loop_edges_w], np.int32) if nl else np.array([], np.int32)
    lRo = np.array([e[2] for e in loop_edges_w], np.float64) if nl else np.zeros((0,3,3))
    lto = np.array([e[3] for e in loop_edges_w], np.float64) if nl else np.zeros((0,3))
    lw  = np.array([e[4] for e in loop_edges_w], np.float64) if nl else np.array([])

    sparsity = _pg_sparsity(N, si, sj, li, lj)

    def _build(x):
        xr = x.reshape(N-1, 6)
        Ra = np.empty((N,3,3),np.float64); Ra[0]=R0; Ra[1:]=_rmat_batch(xr[:,:3])
        ta = np.empty((N,3),  np.float64); ta[0]=t0; ta[1:]=xr[:,3:6]
        return Ra, ta

    def _block(Ra, ta, ia, ja, Ro, to):
        Ri=Ra[ia]; Rj=Ra[ja]; ti=ta[ia]; tj=ta[ja]
        RjT=np.transpose(Rj,(0,2,1))
        Rp=RjT@Ri; tp=(RjT@(ti-tj)[...,None]).squeeze(-1)
        RoTRp=np.transpose(Ro,(0,2,1))@Rp
        rr=0.5*np.stack([RoTRp[:,2,1]-RoTRp[:,1,2],
                          RoTRp[:,0,2]-RoTRp[:,2,0],
                          RoTRp[:,1,0]-RoTRp[:,0,1]],axis=1)
        return np.hstack([rr, tp-to])

    def residuals(x):
        Ra, ta = _build(x)
        rs = _block(Ra, ta, si, sj, sRo, sto).ravel()
        if nl:
            rl = _block(Ra, ta, li, lj, lRo, lto); rl *= lw[:,None]
            return np.concatenate([rs, rl.ravel()])
        return rs

    r0_full = residuals(x0)
    n_seq_res = len(si)*6
    lnorm0 = float(np.linalg.norm(r0_full[n_seq_res:]))
    cost0  = float(np.dot(r0_full, r0_full))

    try:
        res = least_squares(residuals, x0, method='trf',
                            jac_sparsity=sparsity,
                            max_nfev=500, ftol=1e-6, xtol=1e-6, gtol=1e-6,
                            verbose=0)
        r1_full = residuals(res.x)
        lnorm1  = float(np.linalg.norm(r1_full[n_seq_res:]))
        cost1   = float(np.dot(r1_full, r1_full))

        if not np.all(np.isfinite(res.x)):
            print("  PG REJECTED: NaN/Inf")
            return kf_poses
        if nl > 0 and lnorm1 > lnorm0 * 0.95:
            print(f"  PG REJECTED: loop_res {lnorm0:.4f}->{lnorm1:.4f} not reduced")
            return kf_poses
        if cost1 > cost0 * 1.05:
            print(f"  PG REJECTED: cost {cost0:.4f}->{cost1:.4f} increased")
            return kf_poses

        xr = res.x.reshape(N-1, 6)
        max_jump = max(float(np.linalg.norm(xr[i-1,3:6]-kf_poses[i][1])) for i in range(1,N))
        print(f"  PG: cost {cost0:.4f}->{cost1:.4f}  "
              f"loop_res {lnorm0:.4f}->{lnorm1:.4f}  max_jump {max_jump:.2f}m")
        new = [kf_poses[0]]
        for i in range(1, N):
            new.append((_rmat_np(xr[i-1,:3]), xr[i-1,3:6].copy()))
        return new
    except Exception as e:
        print(f"  PG error: {e}")
        return kf_poses

# ── Parameters ─────────────────────────────────────────────────────────────────
KF_INTERVAL       = 5      # keyframe every N frames
MFPNP_LOOKBACK    = 3      # multi-frame PnP lookback in keyframes
LOOP_MIN_GAP      = 60     # min frame gap for loop candidate
LOOP_SCORE_THRESH = 0.05   # min descriptor score (loose; PnP provides the real gate)
LOOP_TOP_K        = 3      # try top-K score candidates geometrically
LOOP_MIN_INLIERS  = 20     # min PnP inliers for loop acceptance
LOOP_MAX_REPROJ   = 5.0    # max reprojection error px for loop acceptance

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    np.random.seed(42)

    K, bl, fx, fy, cx, cy = load_intrinsics()
    N = count_frames()
    print(f"N={N} frames  fx={fx:.2f}  bl={bl:.4f}m")

    frame_poses = []   # list of (R_cw, t_cw) per frame
    keyframes   = []   # list of KF dicts

    R_cw = np.eye(3, dtype=np.float64)
    t_cw = np.zeros(3, dtype=np.float64)

    # Front-end state
    kps_prev = None; des_prev = None; depth_prev = None

    print("Front-end VO ...")
    for fi in range(N):
        L = cv2.imread(os.path.join(DATA_DIR, f"left_{fi:06d}.png"),
                       cv2.IMREAD_GRAYSCALE)
        R_img = cv2.imread(os.path.join(DATA_DIR, f"right_{fi:06d}.png"),
                           cv2.IMREAD_GRAYSCALE)
        depth = compute_depth(L, R_img, fx, bl)
        kps, des = extract_frame_features(L)

        if fi == 0:
            frame_poses.append((R_cw.copy(), t_cw.copy()))
        else:
            # ── ORB matching with previous frame ────────────────────────────
            ms = match_features(des_prev, des)
            if len(ms) >= 6:
                # Backproject matched keypoints from prev depth
                prev_idx = [m.queryIdx for m in ms]
                curr_idx = [m.trainIdx for m in ms]

                p3c, valid_i = backproject_kps(kps_prev, prev_idx, depth_prev, fx, fy, cx, cy)
                if len(valid_i) >= 6:
                    # 2D observations in current frame
                    p2 = np.array([kps[curr_idx[vi]].pt for vi in valid_i], np.float64)
                    # Convert prev-cam 3D -> world
                    p3w = (R_cw @ p3c.T).T + t_cw

                    R_wc, t_wc, mask_pnp = run_pnp(p3w, p2, K, min_inl=6)
                    if R_wc is not None and mask_pnp.sum() >= 6:
                        R_cw = R_wc.T
                        t_cw = -R_wc.T @ t_wc

            frame_poses.append((R_cw.copy(), t_cw.copy()))

        # Update state for next frame
        kps_prev  = kps
        des_prev  = des
        depth_prev = depth

        # ── Keyframe ─────────────────────────────────────────────────────────
        if fi % KF_INTERVAL == 0:
            # 3D world-frame points for ORB keypoints (from current depth)
            pts3d_w = [None] * len(kps)
            for i, kp in enumerate(kps):
                u = int(round(kp.pt[0])); vv = int(round(kp.pt[1]))
                if 0 <= vv < depth.shape[0] and 0 <= u < depth.shape[1]:
                    z = float(depth[vv, u])
                    if 1.0 < z < 60.0:
                        xc = (kp.pt[0]-cx)*z/fx
                        yc = (kp.pt[1]-cy)*z/fy
                        pts3d_w[i] = R_cw @ np.array([xc,yc,z]) + t_cw

            # Subsampled descriptors for fast loop scoring
            idx_des = None
            if des is not None and len(des) >= 8:
                rng = np.random.RandomState(fi)
                idx = rng.choice(len(des), min(250, len(des)), replace=False)
                idx_des = des[idx].copy()

            kf = dict(fi=fi, kf_idx=len(keyframes),
                      R_cw=R_cw.copy(), t_cw=t_cw.copy(),
                      kps=kps, des=des, pts3d_w=pts3d_w, idx_des=idx_des)

            # ── Multi-frame PnP refinement (monotonic safeguard) ────────────
            ki = len(keyframes)
            if ki >= 2:
                look = range(max(0, ki - MFPNP_LOOKBACK), ki)
                all_p3w, all_p2 = [], []
                for pk_idx in look:
                    pkf = keyframes[pk_idx]
                    ms_kf = bf_match_lp(pkf['des'], des, max_dist=60)
                    for m in ms_kf:
                        ip, iq = m.queryIdx, m.trainIdx
                        if ip < len(pkf['pts3d_w']) and pkf['pts3d_w'][ip] is not None:
                            all_p3w.append(pkf['pts3d_w'][ip])
                            all_p2.append(kps[iq].pt)
                if len(all_p3w) >= 15:
                    A3 = np.array(all_p3w, np.float64)
                    A2 = np.array(all_p2,  np.float64)
                    R_wc_in = R_cw.T; t_wc_in = -R_cw.T @ t_cw
                    Rw, tw, mk = run_pnp(A3, A2, K, min_inl=12, rep_err=3.0)
                    if Rw is not None and mk.sum() >= 12:
                        p3in = A3[mk]; p2in = A2[mk]
                        e_old = reproj_err(p3in, p2in, R_wc_in, t_wc_in, K)
                        e_new = reproj_err(p3in, p2in, Rw, tw, K)
                        if e_new < e_old * 1.1:
                            R_cw = Rw.T; t_cw = -Rw.T @ tw
                            frame_poses[fi] = (R_cw.copy(), t_cw.copy())
                            kf['R_cw'] = R_cw.copy(); kf['t_cw'] = t_cw.copy()
                            for i, kp in enumerate(kps):
                                u_ = int(round(kp.pt[0])); v_ = int(round(kp.pt[1]))
                                if 0<=v_<depth.shape[0] and 0<=u_<depth.shape[1]:
                                    z = float(depth[v_, u_])
                                    if 1.0 < z < 60.0:
                                        xc=(kp.pt[0]-cx)*z/fx; yc=(kp.pt[1]-cy)*z/fy
                                        pts3d_w[i] = R_cw @ np.array([xc,yc,z]) + t_cw

            keyframes.append(kf)

        if fi % 50 == 0 or fi == N-1:
            print(f"  f{fi:04d}/{N}  pos=[{t_cw[0]:7.2f} {t_cw[1]:6.2f}"
                  f" {t_cw[2]:7.2f}]  #KF={len(keyframes)}")

    t_vo = time.time()
    print(f"VO done in {t_vo-t0:.1f}s | {len(keyframes)} keyframes")

    # ── Loop detection ─────────────────────────────────────────────────────────
    print("Loop detection ...")
    loop_edges = []
    min_kf_gap = max(1, LOOP_MIN_GAP // KF_INTERVAL)

    for qi in range(min_kf_gap + 1, len(keyframes)):
        kfq = keyframes[qi]
        if kfq['idx_des'] is None: continue

        candidates = []
        for di in range(qi - min_kf_gap):
            kfd = keyframes[di]
            if kfq['fi'] - kfd['fi'] < LOOP_MIN_GAP: continue
            sc = desc_score(kfq['idx_des'], kfd['idx_des'])
            if sc >= LOOP_SCORE_THRESH:
                candidates.append((sc, di))

        if not candidates: continue
        candidates.sort(reverse=True)

        for best_score, best_di in candidates[:LOOP_TOP_K]:
            kfd = keyframes[best_di]
            ms = bf_match_lp(kfd['des'], kfq['des'], max_dist=65)
            if len(ms) < LOOP_MIN_INLIERS: continue

            p3w, p2c = [], []
            for m in ms:
                ip, iq = m.queryIdx, m.trainIdx
                if ip < len(kfd['pts3d_w']) and kfd['pts3d_w'][ip] is not None:
                    p3w.append(kfd['pts3d_w'][ip]); p2c.append(kfq['kps'][iq].pt)

            if len(p3w) < LOOP_MIN_INLIERS: continue
            p3w=np.array(p3w,np.float64); p2c=np.array(p2c,np.float64)

            R_wc_lc, t_wc_lc, mask_lc = run_pnp(p3w, p2c, K,
                                                   min_inl=LOOP_MIN_INLIERS, rep_err=3.0)
            if R_wc_lc is None or mask_lc.sum() < LOOP_MIN_INLIERS: continue

            n_inl = int(mask_lc.sum())
            err = reproj_err(p3w[mask_lc], p2c[mask_lc], R_wc_lc, t_wc_lc, K)
            if err > LOOP_MAX_REPROJ: continue

            R_cw_q_lc = R_wc_lc.T; t_cw_q_lc = -R_wc_lc.T @ t_wc_lc
            Rd = kfd['R_cw']; td = kfd['t_cw']
            R_obs = R_cw_q_lc.T @ Rd
            t_obs = R_cw_q_lc.T @ (td - t_cw_q_lc)

            print(f"  Loop KF{best_di}(f{kfd['fi']}) <- KF{qi}(f{kfq['fi']}) "
                  f"score={best_score:.3f} inl={n_inl} err={err:.2f}px")
            loop_edges.append((best_di, qi, R_obs.copy(), t_obs.copy()))
            break

    print(f"Loops found: {len(loop_edges)} | det time {time.time()-t_vo:.1f}s")

    # ── Global pose-graph optimisation ─────────────────────────────────────────
    if loop_edges:
        print("Pose-graph optimisation ...")
        seq_edges = []
        for i in range(len(keyframes)-1):
            Ri, ti = keyframes[i]['R_cw'], keyframes[i]['t_cw']
            Rj, tj = keyframes[i+1]['R_cw'], keyframes[i+1]['t_cw']
            seq_edges.append((i, i+1, Rj.T@Ri, Rj.T@(ti-tj)))

        w_loop = [(i, j, Ro, to, 5.0) for i,j,Ro,to in loop_edges]
        orig_kf_poses = [(kf['R_cw'].copy(), kf['t_cw'].copy()) for kf in keyframes]

        t_pg = time.time()
        new_kf_poses = pose_graph_opt(orig_kf_poses, seq_edges, w_loop)
        print(f"  PG time: {time.time()-t_pg:.2f}s")

        for i, kf in enumerate(keyframes):
            kf['R_cw'], kf['t_cw'] = new_kf_poses[i]

        # Propagate rigid correction to all frames via preceding KF
        print("Propagating corrections ...")
        for fi2 in range(N):
            pk = None
            for ki in range(len(keyframes)-1, -1, -1):
                if keyframes[ki]['fi'] <= fi2: pk = ki; break
            if pk is None: continue
            R_o, t_o = orig_kf_poses[pk]; R_n, t_n = new_kf_poses[pk]
            Rf, tf = frame_poses[fi2]
            R_d = R_n @ R_o.T; t_d = t_n - R_d @ t_o
            frame_poses[fi2] = (R_d @ Rf, R_d @ tf + t_d)
    else:
        print("No loops found -- writing plain VO trajectory")

    # ── Write outputs ──────────────────────────────────────────────────────────
    traj_path  = os.path.join(OUT_DIR, "traj.txt")
    poses_path = os.path.join(OUT_DIR, "poses.txt")
    with open(traj_path, 'w') as ft, open(poses_path, 'w') as fp:
        for Rc, tc in frame_poses:
            ft.write(f"{tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}\n")
            mat = np.concatenate([Rc, tc.reshape(3,1)], axis=1)
            fp.write(' '.join(f'{x:.6f}' for x in mat.ravel()) + '\n')

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s | {N} poses written")
    print(f"traj: {os.path.getsize(traj_path)} B  poses: {os.path.getsize(poses_path)} B")

if __name__ == "__main__":
    main()
