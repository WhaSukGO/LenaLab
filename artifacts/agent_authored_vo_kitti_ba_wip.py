#!/usr/bin/env python3
"""
Stereo Visual Odometry + sliding-window local Bundle Adjustment  (KITTI).

Key design:
  • Frame-to-frame: relative PnP with PREV-frame stereo depth (metric scale)
  • BA: triangulate 3-D from keyframe pairs, then jointly optimise poses+pts
  • Triangulated init guarantees near-zero reprojection residuals to start
  • SE(3) interpolation of non-KF poses after each BA run
"""
import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

DATA_DIR = os.environ.get('LAB_DATA', '/data')
ART_DIR  = os.environ.get('LAB_ARTIFACTS', '/artifacts')
os.makedirs(ART_DIR, exist_ok=True)

MAX_FEAT    = 600
MIN_FEAT    = 150
KF_EVERY    = 5
BA_WINDOW   = 12
BA_EVERY    = 1
MIN_OBS     = 2
MIN_DEPTH   = 0.5
MAX_DEPTH   = 60.0
MAX_T_FRAME = 8.0
FB_THRESH   = 2.0
TRIANG_REPROJ = 3.0   # px: max reprojection at triangulation views

# ── intrinsics ────────────────────────────────────────────────────────────────
def load_intrinsics():
    with open(os.path.join(DATA_DIR, 'intrinsics.txt')) as f:
        vals = [float(l.strip()) for l in f if l.strip()]
    fx, fy, cx, cy, b = vals
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    return K, b

# ── stereo depth ──────────────────────────────────────────────────────────────
_sgbm = None
def get_depth(iL, iR, fx, b):
    global _sgbm
    if _sgbm is None:
        _sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=11,
            P1=8*3*121, P2=32*3*121,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = _sgbm.compute(iL, iR).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    m = disp > 1.0
    depth[m] = fx * b / disp[m]
    return depth

# ── geometry ──────────────────────────────────────────────────────────────────
def backproj(pts2d, depth, K):
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = depth.shape
    pi = np.round(pts2d).astype(int)
    ok = (pi[:,0]>=0)&(pi[:,0]<w)&(pi[:,1]>=0)&(pi[:,1]<h)
    z  = np.where(ok, depth[np.clip(pi[:,1],0,h-1),np.clip(pi[:,0],0,w-1)], 0.0)
    ok &= (z >= MIN_DEPTH) & (z <= MAX_DEPTH)
    X  = np.where(ok, (pts2d[:,0]-cx)*z/fx, 0.0)
    Y  = np.where(ok, (pts2d[:,1]-cy)*z/fy, 0.0)
    return np.c_[X,Y,z], ok

def cam2world(p3c, Rcw, tcw):
    return (Rcw.T @ (p3c - tcw).T).T

def compose(R_rel, t_rel, Rp, tp):
    return R_rel @ Rp, R_rel @ tp + t_rel

def rvec(R):
    r,_ = cv2.Rodrigues(R.astype(np.float64)); return r.flatten()

def rmat(r):
    R,_ = cv2.Rodrigues(np.asarray(r,dtype=np.float64).reshape(3,1)); return R

# ── PnP ───────────────────────────────────────────────────────────────────────
def run_pnp(p3d, p2d, K):
    if len(p3d) < 6:
        return None
    ok, rv, tv, inl = cv2.solvePnPRansac(
        p3d.astype(np.float64), p2d.astype(np.float64), K, None,
        None, None, False, 200, 2.5, 0.999, None, cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return None
    inl = inl.flatten()
    rv, tv = cv2.solvePnPRefineLM(
        p3d[inl].astype(np.float64), p2d[inl].astype(np.float64),
        K, None, rv, tv)
    R = rmat(rv); t = tv.flatten()
    if np.linalg.norm(t) > MAX_T_FRAME:
        return None
    return R, t, inl

# ── feature tracking ──────────────────────────────────────────────────────────
def detect_pts(gray, max_n=MAX_FEAT):
    det = cv2.GFTTDetector_create(
        maxCorners=max_n, qualityLevel=0.01, minDistance=10, blockSize=5)
    kps = det.detect(gray)
    if not kps:
        return np.zeros((0,2), dtype=np.float32)
    return np.array([k.pt for k in kps], dtype=np.float32)

def lk_track_fb(pg, cg, ppts, h, w):
    if len(ppts) == 0:
        return np.zeros((0,2),np.float32), np.zeros(0,bool)
    ppts = ppts.astype(np.float32)
    lk = dict(winSize=(21,21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    cpts, sf, _ = cv2.calcOpticalFlowPyrLK(pg, cg, ppts, None, **lk)
    back, sb, _ = cv2.calcOpticalFlowPyrLK(cg, pg, cpts, None, **lk)
    ok = (sf.flatten()==1) & (sb.flatten()==1)
    ok &= np.linalg.norm(ppts - back, axis=1) < FB_THRESH
    ok &= (cpts[:,0]>=0)&(cpts[:,0]<w)&(cpts[:,1]>=0)&(cpts[:,1]<h)
    return cpts, ok

_orb = None; _bf = None
def orb_match(pg, cg, prev_depth, K):
    global _orb, _bf
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=1000)
        _bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp1, des1 = _orb.detectAndCompute(pg, None)
    kp2, des2 = _orb.detectAndCompute(cg, None)
    if des1 is None or des2 is None or len(kp1) < 8:
        return None, None
    matches = [m for m in _bf.match(des1, des2) if m.distance < 64]
    if len(matches) < 8:
        return None, None
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
    p3c, valid = backproj(pts1, prev_depth, K)
    if valid.sum() < 8:
        return None, None
    return p3c[valid], pts2[valid]

# ── local BA with triangulated initialisation ─────────────────────────────────
def triangulate_lm_init(kf_window, K):
    """Compute initial 3-D world positions via triangulation from KF pairs."""
    # Gather all observations per landmark
    lm_obs = {}
    for ki, kf in enumerate(kf_window):
        for lid, u, v in kf['obs']:
            lm_obs.setdefault(lid, []).append((ki, float(u), float(v)))

    # Keep only multi-view landmarks
    lm_obs = {lid: obs for lid, obs in lm_obs.items() if len(obs) >= MIN_OBS}

    lm_world = {}
    for lid, obs_list in lm_obs.items():
        # Triangulate from first and last observation
        ki1, u1, v1 = obs_list[0]
        ki2, u2, v2 = obs_list[-1]
        R1, t1 = kf_window[ki1]['R'], kf_window[ki1]['t']
        R2, t2 = kf_window[ki2]['R'], kf_window[ki2]['t']

        P1 = K @ np.hstack([R1, t1.reshape(3,1)])
        P2 = K @ np.hstack([R2, t2.reshape(3,1)])
        pt4d = cv2.triangulatePoints(
            P1, P2,
            np.array([[u1],[v1]], dtype=np.float64),
            np.array([[u2],[v2]], dtype=np.float64))
        if abs(pt4d[3,0]) < 1e-10:
            continue
        pt3d = pt4d[:3,0] / pt4d[3,0]

        # Depth sanity in both views
        pc1 = R1 @ pt3d + t1
        pc2 = R2 @ pt3d + t2
        if pc1[2] < MIN_DEPTH or pc2[2] < MIN_DEPTH:
            continue
        if pc1[2] > MAX_DEPTH*2 or pc2[2] > MAX_DEPTH*2:
            continue

        # Reprojection sanity: triangulated point must re-project close to observations
        fx_,fy_,cx_,cy_ = K[0,0],K[1,1],K[0,2],K[1,2]
        u1p = fx_*pc1[0]/pc1[2]+cx_; v1p = fy_*pc1[1]/pc1[2]+cy_
        u2p = fx_*pc2[0]/pc2[2]+cx_; v2p = fy_*pc2[1]/pc2[2]+cy_
        if ((u1p-u1)**2+(v1p-v1)**2) > TRIANG_REPROJ**2:
            continue
        if ((u2p-u2)**2+(v2p-v2)**2) > TRIANG_REPROJ**2:
            continue

        lm_world[lid] = pt3d

    return lm_world, lm_obs

def local_ba(kf_window, K):
    lm_world, lm_obs = triangulate_lm_init(kf_window, K)
    if len(lm_world) < 10:
        return False

    lids  = sorted(lm_world)
    lidx  = {l:i for i,l in enumerate(lids)}
    n_kf  = len(kf_window)
    n_lm  = len(lids)

    # Build obs arrays (only landmarks successfully triangulated)
    ki_arr=[]; lmi_arr=[]; u_arr=[]; v_arr=[]
    for lid in lids:
        for ki,u,v in lm_obs[lid]:
            ki_arr.append(ki); lmi_arr.append(lidx[lid])
            u_arr.append(u);   v_arr.append(v)
    ki_arr  = np.array(ki_arr,  np.int32)
    lmi_arr = np.array(lmi_arr, np.int32)
    u_arr   = np.array(u_arr)
    v_arr   = np.array(v_arr)
    n_obs   = len(ki_arr)

    x0 = np.concatenate(
        [np.r_[rvec(kf['R']), kf['t']] for kf in kf_window] +
        [lm_world[lid] for lid in lids])

    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]

    def residuals(x):
        Rs  = np.array([rmat(x[ki*6:ki*6+3]) for ki in range(n_kf)])
        ts  = x[:n_kf*6].reshape(n_kf, 6)[:, 3:]
        pws = x[n_kf*6:].reshape(n_lm, 3)
        pc  = np.einsum('ijk,ik->ij', Rs[ki_arr], pws[lmi_arr]) + ts[ki_arr]
        valid = pc[:,2] > 0.1
        iz    = np.where(valid, 1.0/np.where(valid, pc[:,2], 1.0), 0.0)
        r = np.empty(2*n_obs)
        r[0::2] = np.where(valid, fx*pc[:,0]*iz + cx - u_arr, 100.0)
        r[1::2] = np.where(valid, fy*pc[:,1]*iz + cy - v_arr, 100.0)
        return r

    sp = lil_matrix((2*n_obs, len(x0)), dtype=int)
    for i in range(n_obs):
        ki=ki_arr[i]; lmi=lmi_arr[i]
        sp[2*i:2*i+2, ki*6:ki*6+6]                  = 1
        sp[2*i:2*i+2, n_kf*6+lmi*3:n_kf*6+lmi*3+3] = 1

    lo = np.full(len(x0),-np.inf); hi = np.full(len(x0),np.inf)
    lo[:6] = x0[:6]-1e-12; hi[:6] = x0[:6]+1e-12

    cost0 = 0.5*np.sum(residuals(x0)**2)
    try:
        res = least_squares(residuals, x0, jac_sparsity=sp.tocsr(),
                            method='trf', loss='huber', f_scale=1.5,
                            max_nfev=200, ftol=1e-4, xtol=1e-4, gtol=1e-8,
                            bounds=(lo,hi), verbose=0)
        xo = res.x
    except Exception as e:
        print(f"BA err:{e}"); return False

    cost1 = 0.5*np.sum(residuals(xo)**2)
    if cost1 >= cost0:
        print(f"BA no improvement ({cost0:.1f}->{cost1:.1f})")
        return False

    # Reject if any non-fixed KF moved too far (guards against BA divergence)
    max_move = 0.0
    for ki, kf in enumerate(kf_window):
        old_C = -kf['R'].T @ kf['t']
        new_R = rmat(xo[ki*6:ki*6+3])
        new_t = xo[ki*6+3:ki*6+6]
        new_C = -new_R.T @ new_t
        max_move = max(max_move, np.linalg.norm(new_C - old_C))
    if max_move > 15.0:
        print(f"BA rejected: max KF move {max_move:.1f}m")
        return False

    for ki,kf in enumerate(kf_window):
        kf['R'] = rmat(xo[ki*6:ki*6+3])
        kf['t'] = xo[ki*6+3:ki*6+6].copy()

    print(f"{n_lm}lm/{n_obs}obs  {cost0:.0f}->{cost1:.0f}  max_move={max_move:.2f}m")
    return True

# ── pose interpolation ────────────────────────────────────────────────────────
def interp_pose(R0, t0, R1, t1, alpha):
    """Interpolate two world-to-cam poses (R_cw, t_cw).
    Rotation: Rodrigues LERP.
    Translation: interpolate camera CENTRES (world positions), then
    convert back to t_cw = -R_interp @ C_interp.
    Interpolating t_cw directly is incorrect when R changes (car turns)."""
    rv = (1-alpha)*rvec(R0) + alpha*rvec(R1)
    R_interp = rmat(rv)
    C0 = -R0.T @ t0          # camera centre in world frame
    C1 = -R1.T @ t1
    C_interp = (1-alpha)*C0 + alpha*C1
    t_interp = -R_interp @ C_interp   # back to t_cw
    return R_interp, t_interp

def patch_nkf_poses(all_R, all_t, kf_fis):
    for i in range(len(kf_fis)-1):
        fi1, fi2 = kf_fis[i], kf_fis[i+1]
        for fj in range(fi1+1, fi2):
            a = (fj-fi1)/(fi2-fi1)
            all_R[fj], all_t[fj] = interp_pose(
                all_R[fi1], all_t[fi1], all_R[fi2], all_t[fi2], a)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    K, b = load_intrinsics()
    fx = K[0,0]

    imgs = sorted(f for f in os.listdir(DATA_DIR)
                  if f.startswith('left_') and f.endswith('.png'))
    N = len(imgs)
    print(f"Frames: {N},  fx={fx:.2f}  b={b:.4f}m")

    all_R = []; all_t = []

    R_cw = np.eye(3); t_cw = np.zeros(3)
    R_vel = np.eye(3); t_vel = np.zeros(3)
    consec_fail = 0

    prev_gray  = None
    prev_depth = None
    prev_pts   = np.zeros((0,2), np.float32)
    prev_ids   = np.zeros(0, np.int64)
    img_h = img_w = None
    next_lid = 0

    kf_window  = []
    kf_count   = 0
    all_kf_fis = []   # ALL keyframe frame indices, for final global patch

    for fi in range(N):
        iL = cv2.imread(os.path.join(DATA_DIR, f"left_{fi:06d}.png"),  cv2.IMREAD_GRAYSCALE)
        iR = cv2.imread(os.path.join(DATA_DIR, f"right_{fi:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if iL is None or iR is None:
            print(f"  Missing {fi}"); break
        if img_h is None:
            img_h, img_w = iL.shape

        depth = get_depth(iL, iR, fx, b)

        # ── frame 0 ──────────────────────────────────────────────────────
        if fi == 0:
            prev_gray = iL; prev_depth = depth
            pts = detect_pts(iL)
            ids = np.arange(next_lid, next_lid+len(pts), dtype=np.int64)
            next_lid += len(pts)
            prev_pts, prev_ids = pts, ids
            all_R.append(R_cw.copy()); all_t.append(t_cw.copy())
            continue

        # ── LK tracking ───────────────────────────────────────────────────
        curr_pts_all, ok_lk = lk_track_fb(prev_gray, iL, prev_pts, img_h, img_w)
        prev_pts_lk = prev_pts[ok_lk]
        curr_pts_lk = curr_pts_all[ok_lk]
        curr_ids_lk = prev_ids[ok_lk]

        # ── PnP: relative (prev_cam 3D → curr 2D) ────────────────────────
        pnp_ok = False
        if len(curr_pts_lk) >= 8:
            p3c_prev, valid3d = backproj(prev_pts_lk, prev_depth, K)
            if valid3d.sum() >= 6:
                result = run_pnp(p3c_prev[valid3d], curr_pts_lk[valid3d], K)
                if result is not None:
                    R_rel, t_rel, _ = result
                    R_cw, t_cw = compose(R_rel, t_rel, all_R[-1], all_t[-1])
                    R_vel = R_rel.copy(); t_vel = t_rel.copy()
                    consec_fail = 0; pnp_ok = True

        if not pnp_ok:
            p3d_orb, p2d_orb = orb_match(prev_gray, iL, prev_depth, K)
            if p3d_orb is not None:
                result = run_pnp(p3d_orb, p2d_orb, K)
                if result is not None:
                    R_rel, t_rel, _ = result
                    R_cw, t_cw = compose(R_rel, t_rel, all_R[-1], all_t[-1])
                    R_vel = R_rel.copy(); t_vel = t_rel.copy()
                    consec_fail = 0; pnp_ok = True
                    curr_pts_lk = np.zeros((0,2), np.float32)
                    curr_ids_lk = np.zeros(0, np.int64)

        if not pnp_ok:
            consec_fail += 1
            R_cw, t_cw = compose(R_vel, t_vel, all_R[-1], all_t[-1])
            if consec_fail % 3 == 0:
                print(f"  Frame {fi}: {consec_fail} PnP fail")

        all_R.append(R_cw.copy()); all_t.append(t_cw.copy())

        # ── re-detect if few features ──────────────────────────────────
        curr_pts = curr_pts_lk; curr_ids = curr_ids_lk
        if len(curr_pts) < MIN_FEAT:
            new_pts = detect_pts(iL, MAX_FEAT)
            new_ids = np.arange(next_lid, next_lid+len(new_pts), dtype=np.int64)
            next_lid += len(new_pts)
            curr_pts = (np.vstack([curr_pts, new_pts]) if len(curr_pts)>0 else new_pts)
            curr_ids = (np.concatenate([curr_ids, new_ids]) if len(curr_ids)>0 else new_ids)

        # ── keyframe + BA ─────────────────────────────────────────────
        if fi % KF_EVERY == 0:
            # Build observations using CURRENT depth (consistent with current pose)
            p3c_kf, ok_kf = backproj(curr_pts, depth, K)
            obs = []
            for i in range(len(curr_pts)):
                if ok_kf[i]:
                    obs.append((int(curr_ids[i]),
                                float(curr_pts[i,0]),
                                float(curr_pts[i,1])))

            kf_window.append({'R': R_cw.copy(), 't': t_cw.copy(),
                               'obs': obs, 'fi': fi})
            all_kf_fis.append(fi)
            kf_count += 1
            if len(kf_window) > BA_WINDOW:
                kf_window.pop(0)

            if kf_count % BA_EVERY == 0 and len(kf_window) >= 2:
                print(f"  Frame {fi}: BA({len(kf_window)}kf) ", end='')
                improved = local_ba(kf_window, K)
                if improved:
                    R_cw = kf_window[-1]['R'].copy()
                    t_cw = kf_window[-1]['t'].copy()
                    for kf in kf_window:
                        all_R[kf['fi']] = kf['R'].copy()
                        all_t[kf['fi']] = kf['t'].copy()
                    kf_fis = sorted(kf['fi'] for kf in kf_window)
                    patch_nkf_poses(all_R, all_t, kf_fis)

        prev_gray  = iL
        prev_depth = depth
        prev_pts   = curr_pts
        prev_ids   = curr_ids

        if fi % 50 == 0:
            C = -R_cw.T @ t_cw
            print(f"Frame {fi}/{N}  pos=({C[0]:.1f},{C[1]:.1f},{C[2]:.1f})")

    # ── final global re-patch of all non-KF poses ─────────────────────────────
    # BAs update KF poses many times; later BAs may move a KF after the
    # non-KF frames adjacent to it were last patched. One final global pass
    # using the FINAL KF positions guarantees smooth, consistent interpolation.
    if len(all_kf_fis) >= 2:
        patch_nkf_poses(all_R, all_t, all_kf_fis)
    print(f"Final global patch done ({len(all_kf_fis)} keyframes)")

    # ── write outputs ─────────────────────────────────────────────────────────
    with open(os.path.join(ART_DIR, 'traj.txt'), 'w') as f:
        for R,t in zip(all_R, all_t):
            C = -R.T @ t
            f.write(f"{C[0]:.6f} {C[1]:.6f} {C[2]:.6f}\n")

    with open(os.path.join(ART_DIR, 'poses.txt'), 'w') as f:
        for R,t in zip(all_R, all_t):
            Rwc = R.T; twc = (-R.T @ t).reshape(3,1)
            M = np.hstack([Rwc, twc])
            f.write(' '.join(f"{v:.8f}" for v in M.flatten()) + '\n')

    print(f"Done: {len(all_R)} poses")

if __name__ == '__main__':
    main()
