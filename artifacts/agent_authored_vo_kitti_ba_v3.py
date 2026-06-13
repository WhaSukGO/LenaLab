#!/usr/bin/env python3
"""
Stereo VO for KITTI outdoor driving — v5: 3-view BA.

Tracking : Shi-Tomasi + Lucas-Kanade optical flow (forward-backward check).
Depth    : SGBM  →  Z = fx*baseline/disp  (metric).
Pose     : 3D-2D PnP + RANSAC + LM refinement.
BA       : Sliding window FULL BA (poses + free 3-D points).
           USES 3-VIEW LANDMARKS to avoid 2-view degeneracy:
             features tracked across KF[-3]→KF[-2]→current
             give 3 projections per 3-D point → 3 net constraints per point
             → genuine cross-window correction without free-point absorption.
           Consecutive 2-view obs also included for tight consecutive constraints.
           Monotonic safeguard + max-displacement safeguard.
Multi-seq: detects seq number from LAB_DATA path.
"""

import os, sys, re
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("LAB_DATA", "/data")
OUT_DIR  = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

_bn = os.path.basename(DATA_DIR.rstrip("/"))
_m  = re.match(r"seq_?(\d+)", _bn)
SEQ = _m.group(1) if _m else None

def _out(name):
    if SEQ is not None:
        base, ext = os.path.splitext(name)
        return os.path.join(OUT_DIR, f"{base}_{SEQ}{ext}")
    return os.path.join(OUT_DIR, name)

# ── Intrinsics ────────────────────────────────────────────────────────────────
vals     = open(os.path.join(DATA_DIR, "intrinsics.txt")).read().split()
fx, fy, cx, cy, baseline = [float(v) for v in vals]
K        = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
dist     = np.zeros(5)

# ── SGBM ─────────────────────────────────────────────────────────────────────
_bw = 5
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=_bw,
    P1=8*3*_bw**2, P2=32*3*_bw**2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
    preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def compute_depth(L, R):
    d = sgbm.compute(L, R).astype(np.float32) / 16.0
    dep = np.zeros_like(d)
    m = d > 0.5
    dep[m] = fx * baseline / d[m]
    return dep

MAX_DEPTH = 40.0

# ── LK parameters ────────────────────────────────────────────────────────────
_NFEAT = 600
_NMIN  = 80
_lk = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
_lk_w = dict(winSize=(31, 31), maxLevel=4,
             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))

def detect(img):
    p = cv2.goodFeaturesToTrack(img, maxCorners=_NFEAT, qualityLevel=0.01,
                                minDistance=7, blockSize=5)
    return p.reshape(-1, 2) if p is not None else np.empty((0, 2), np.float32)

def track_ok(img0, img1, pts0, wide=False):
    """Returns (pts0_survived, pts1_survived, ok_mask) after forward-backward LK."""
    if len(pts0) == 0:
        return (np.empty((0,2),np.float32), np.empty((0,2),np.float32),
                np.zeros(0, bool))
    prm = _lk_w if wide else _lk
    p0 = pts0.reshape(-1, 1, 2).astype(np.float32)
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **prm)
    pb, sb, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **prm)
    fb  = np.abs(p0 - pb).reshape(-1, 2).max(axis=1)
    ok  = (s1.ravel() == 1) & (sb.ravel() == 1) & (fb < 1.5)
    return pts0[ok], p1.reshape(-1, 2)[ok], ok

def backproject(pts2d, dmap):
    h, w = dmap.shape
    u  = pts2d[:, 0]; v  = pts2d[:, 1]
    ui = np.round(u).astype(int).clip(0, w-1)
    vi = np.round(v).astype(int).clip(0, h-1)
    z  = dmap[vi, ui]
    ok = (z > 0.5) & (z < MAX_DEPTH)
    p3 = np.column_stack([(u-cx)*z/fx, (v-cy)*z/fy, z]).astype(np.float64)
    return p3, ok

def pnp(pts3, pts2):
    if len(pts3) < 8:
        return None, None, None
    ok, rv, tv, inl = cv2.solvePnPRansac(
        pts3.astype(np.float64), pts2.astype(np.float64), K, dist,
        iterationsCount=300, reprojectionError=2.0,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 8:
        return None, None, None
    inl = inl.ravel()
    rv, tv = cv2.solvePnPRefineLM(
        pts3[inl].astype(np.float64), pts2[inl].astype(np.float64),
        K, dist, rv, tv)
    R, _ = cv2.Rodrigues(rv)
    return R, tv.ravel(), inl

# ── Batch Rodrigues ───────────────────────────────────────────────────────────
def brod(rvecs):
    if len(rvecs) == 0:
        return np.zeros((0, 3, 3))
    th = np.linalg.norm(rvecs, axis=1, keepdims=True)
    k  = rvecs / np.where(th > 1e-10, th, 1.0)
    s, c = np.sin(th), np.cos(th)
    N = len(rvecs); Km = np.zeros((N, 3, 3))
    Km[:,0,1]=-k[:,2]; Km[:,0,2]= k[:,1]
    Km[:,1,0]= k[:,2]; Km[:,1,2]=-k[:,0]
    Km[:,2,0]=-k[:,1]; Km[:,2,1]= k[:,0]
    K2 = np.einsum('nij,njk->nik', Km, Km)
    return np.eye(3)[None] + s[:,:,None]*Km + (1-c[:,:,None])*K2

def r2v(R):
    rv, _ = cv2.Rodrigues(R); return rv.ravel()

# ── Full BA (free 3D points + free poses) ────────────────────────────────────
def _ba_res(x, nk, np_, opi, oli, o2d):
    poses = x[:nk*6].reshape(nk, 6)
    pts   = x[nk*6:].reshape(np_, 3)
    Rb    = brod(poses[opi, :3])
    pc    = np.einsum('nij,nj->ni', Rb, pts[oli]) + poses[opi, 3:]
    z     = pc[:, 2]; sf = z > 0.1; zs = np.where(sf, z, 1.0)
    u     = np.where(sf, fx*pc[:,0]/zs + cx, o2d[:,0])
    v     = np.where(sf, fy*pc[:,1]/zs + cy, o2d[:,1])
    return np.column_stack([u-o2d[:,0], v-o2d[:,1]]).ravel()

def _rmse(x, nk, np_, opi, oli, o2d):
    return float(np.sqrt(np.mean(_ba_res(x, nk, np_, opi, oli, o2d)**2)))

def _sparsity(nk, np_, opi, oli):
    m = len(opi)*2; n = nk*6 + np_*3
    A = lil_matrix((m, n), dtype=np.uint8)
    rows = np.arange(len(opi))
    for d in range(6): A[2*rows, opi*6+d]=1; A[2*rows+1, opi*6+d]=1
    for d in range(3):
        c = nk*6 + oli*3 + d; A[2*rows, c]=1; A[2*rows+1, c]=1
    return A.tocsr()

WSIZE  = 14
KT     = 0.3
KR     = 0.04

def run_ba(kfw, f_scale=3.0, max_nfev=500):
    nk = len(kfw)
    if nk < 3: return
    kf2i = {id(kf): i for i, kf in enumerate(kfw)}
    t_before = [kf['t'].copy() for kf in kfw]

    opi_l, oli_l, o2d_l, pts_w_all = [], [], [], []
    na = 0

    for ci, kfc in enumerate(kfw):
        # 2-view consecutive observations (accurate, from PnP inliers)
        for obs in kfc.get('obs2', []):
            ri = kf2i.get(id(obs['rkf']))
            if ri is None: continue
            Rr = obs['rkf']['R']; tr = obs['rkf']['t']
            pw = (Rr @ obs['p3'].T).T + tr
            n  = len(pw); idx = na + np.arange(n, dtype=np.int32); na += n
            pts_w_all.append(pw)
            opi_l.append(np.full(n, ri, dtype=np.int32)); oli_l.append(idx); o2d_l.append(obs['pr'])
            opi_l.append(np.full(n, ci, dtype=np.int32)); oli_l.append(idx); o2d_l.append(obs['pc'])

        # 3-view cross-window observations (KF[-3]→KF[-2]→current)
        for obs3 in kfc.get('obs3', []):
            ri2 = kf2i.get(id(obs3['rkf2']))
            ri1 = kf2i.get(id(obs3['rkf1']))
            if ri2 is None or ri1 is None: continue
            Rr2 = obs3['rkf2']['R']; tr2 = obs3['rkf2']['t']
            pw = (Rr2 @ obs3['p3'].T).T + tr2
            n  = len(pw); idx = na + np.arange(n, dtype=np.int32); na += n
            pts_w_all.append(pw)
            opi_l.append(np.full(n, ri2, dtype=np.int32)); oli_l.append(idx); o2d_l.append(obs3['pr2'])
            opi_l.append(np.full(n, ri1, dtype=np.int32)); oli_l.append(idx); o2d_l.append(obs3['pr1'])
            opi_l.append(np.full(n, ci,  dtype=np.int32)); oli_l.append(idx); o2d_l.append(obs3['pc'])

    if not pts_w_all or na < 10: return

    p3   = np.vstack(pts_w_all)
    opi  = np.concatenate(opi_l); oli = np.concatenate(oli_l)
    o2d  = np.vstack(o2d_l).astype(np.float64); np_ = len(p3)

    # Minimum overdetermination: 2× (poses only)
    if len(opi) < 2 * nk * 6:
        return

    x0p = np.zeros((nk, 6))
    for i, kf in enumerate(kfw):
        Rc = kf['R'].T; tc = -Rc @ kf['t']
        x0p[i, :3] = r2v(Rc); x0p[i, 3:] = tc
    x0  = np.concatenate([x0p.ravel(), p3.ravel()])
    sp  = _sparsity(nk, np_, opi, oli)
    eb  = _rmse(x0, nk, np_, opi, oli, o2d)

    try:
        res = least_squares(_ba_res, x0,
                            args=(nk, np_, opi, oli, o2d),
                            method='trf', loss='huber', f_scale=f_scale,
                            jac_sparsity=sp, max_nfev=max_nfev,
                            ftol=1e-5, xtol=1e-6, gtol=1e-7, verbose=0)
    except Exception as e:
        print(f"  [BA] err: {e}", file=sys.stderr); return

    ea = _rmse(res.x, nk, np_, opi, oli, o2d)
    if ea >= eb - 1e-6:
        print(f"  [BA] rej {eb:.3f}->{ea:.3f}", file=sys.stderr); return

    # Displacement check
    po  = res.x[:nk*6].reshape(nk, 6)
    t_prop = []
    for i in range(1, nk):
        Rc = brod(po[i:i+1, :3])[0]; tc = po[i, 3:]
        t_prop.append(-Rc.T @ tc)
    max_disp = max(np.linalg.norm(t_prop[i-1] - t_before[i]) for i in range(1, nk))
    if max_disp > 0.5:
        print(f"  [BA] rej_disp {max_disp:.3f}m", file=sys.stderr); return

    print(f"  [BA] acc {eb:.3f}->{ea:.3f} ({nk}kf {np_}pts disp={max_disp:.3f}m)", file=sys.stderr)
    for i in range(1, nk):
        Rc = brod(po[i:i+1, :3])[0]; tc = po[i, 3:]
        kfw[i]['R'] = Rc.T; kfw[i]['t'] = -Rc.T @ tc

# ── 3-view observation builder ────────────────────────────────────────────────
def make_3view_obs(kf2, kf1, img_cur):
    """
    Track kf2 features → kf1 → img_cur.
    Returns obs3 dict or None.
    3D points from kf2's depth → NOT degenerate in BA (3 projections per point).
    """
    pts2 = kf2['pts']
    if len(pts2) < 8: return None

    # Step 1: kf2 → kf1
    _, p1m, ok1 = track_ok(kf2['img'], kf1['img'], pts2, wide=True)
    if ok1.sum() < 8: return None
    pts2_ok = pts2[ok1]; p1_ok = p1m

    # Step 2: kf1 → cur
    _, pcc, ok2 = track_ok(kf1['img'], img_cur, p1_ok, wide=True)
    if ok2.sum() < 5: return None
    pr2 = pts2_ok[ok2]; pr1 = p1_ok[ok2]; pcc = pcc

    # F-RANSAC filter on pr1-pcc pair (skip-1 geometry)
    if len(pr1) >= 8:
        F, mask = cv2.findFundamentalMat(pr1, pcc, cv2.FM_RANSAC, 2.0, 0.999)
        if F is not None and mask is not None:
            gm = mask.ravel().astype(bool)
            if gm.sum() < 5: return None
            pr2, pr1, pcc = pr2[gm], pr1[gm], pcc[gm]

    # Backproject from kf2
    p3, ok3d = backproject(pr2, kf2['depth'])
    if ok3d.sum() < 5: return None
    return {
        'rkf2': kf2, 'rkf1': kf1,
        'p3':  p3[ok3d].astype(np.float64),
        'pr2': pr2[ok3d].astype(np.float64),
        'pr1': pr1[ok3d].astype(np.float64),
        'pc':  pcc[ok3d].astype(np.float64),
    }

# ── Process one sequence ─────────────────────────────────────────────────────
def process_sequence(data_dir, seq_label):
    """Run stereo VO on a single sequence directory. seq_label: '05', '07', or None."""
    global fx, fy, cx, cy, baseline, K, sgbm  # may be re-read per-sequence

    # Re-read intrinsics if a per-sequence intrinsics.txt exists
    intr_path = os.path.join(data_dir, "intrinsics.txt")
    if os.path.exists(intr_path):
        vals2 = open(intr_path).read().split()
        fx2, fy2, cx2, cy2, b2 = [float(v) for v in vals2]
        # Only reinitialise globals if different from module-level
        if abs(fx2 - fx) > 0.01:
            # Different intrinsics: rebuild K and SGBM
            import types
            fx, fy, cx, cy, baseline = fx2, fy2, cx2, cy2, b2
            K[:] = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
            bw = 5
            sgbm = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=128, blockSize=bw,
                P1=8*3*bw**2, P2=32*3*bw**2, disp12MaxDiff=1,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
                preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    lfs = sorted(f for f in os.listdir(data_dir)
                 if f.startswith("left_") and f.endswith(".png"))
    N   = len(lfs)
    if N == 0:
        print(f"[VO] No frames in {data_dir}", file=sys.stderr)
        return

    # Output paths
    def out(name):
        if seq_label is not None:
            base, ext = os.path.splitext(name)
            return os.path.join(OUT_DIR, f"{base}_{seq_label}{ext}")
        return os.path.join(OUT_DIR, name)

    print(f"[VO] {N} frames  seq={seq_label}  fx={fx:.1f} b={baseline:.4f}", file=sys.stderr)

    poses    = [None] * N
    poses[0] = (np.eye(3), np.zeros(3))

    I0   = cv2.imread(os.path.join(data_dir, "left_000000.png"),  cv2.IMREAD_GRAYSCALE)
    IR   = cv2.imread(os.path.join(data_dir, "right_000000.png"), cv2.IMREAD_GRAYSCALE)
    dep0 = compute_depth(I0, IR)
    pts0 = detect(I0)

    kfw = [{'R': np.eye(3), 't': np.zeros(3), 'fi': 0,
            'img': I0, 'pts': pts0, 'depth': dep0, 'obs2': [], 'obs3': []}]

    img_prev   = I0
    pts_prev   = pts0
    depth_prev = dep0
    last_R     = np.eye(3)
    last_t     = np.zeros(3)

    for fi in range(1, N):
        Ic  = cv2.imread(os.path.join(data_dir, f"left_{fi:06d}.png"),  cv2.IMREAD_GRAYSCALE)
        Irc = cv2.imread(os.path.join(data_dir, f"right_{fi:06d}.png"), cv2.IMREAD_GRAYSCALE)
        dc  = compute_depth(Ic, Irc)

        # Frame-to-frame LK tracking
        pp, pct, ok_lk = track_ok(img_prev, Ic, pts_prev)

        Rr = tr = None; pts3d_inl = pts2d_p_inl = pts2d_c_inl = None
        if len(pp) >= 8:
            p3a, ok3d = backproject(pp, depth_prev)
            if ok3d.sum() >= 8:
                p3ok = p3a[ok3d]; p2ok = pct[ok3d]; ppok = pp[ok3d]
                Rr, tr, inl_ = pnp(p3ok, p2ok)
                if Rr is not None:
                    pts3d_inl   = p3ok[inl_]
                    pts2d_p_inl = ppok[inl_].astype(np.float64)
                    pts2d_c_inl = p2ok[inl_].astype(np.float64)

        good_R, good_t = last_R.copy(), last_t.copy()  # last accepted motion

        if Rr is None:
            Rr, tr = good_R, good_t          # dead reckoning
        else:
            # Check step BEFORE committing last_R/last_t
            Rp0, tp0 = poses[fi-1]
            tn_test = tp0 - Rp0 @ Rr.T @ tr
            if np.linalg.norm(tn_test - tp0) > 5.0:
                Rr, tr = good_R, good_t      # reject outlier PnP
            else:
                last_R, last_t = Rr.copy(), tr.copy()   # accept

        Rp, tp = poses[fi-1]
        Rn = Rp @ Rr.T
        tn = tp - Rp @ Rr.T @ tr
        poses[fi] = (Rn.copy(), tn.copy())

        pts_cur = detect(Ic) if len(pct) < _NMIN else pct

        # Keyframe selection
        dt    = np.linalg.norm(tn - kfw[-1]['t'])
        rv_, _ = cv2.Rodrigues(kfw[-1]['R'].T @ Rn)
        is_kf  = (dt > KT) or (np.linalg.norm(rv_) > KR)

        if is_kf:
            nkf = {'R': Rn.copy(), 't': tn.copy(), 'fi': fi,
                   'img': Ic, 'pts': pts_cur, 'depth': dc, 'obs2': [], 'obs3': []}

            # 2-view consecutive (PnP inliers — highly accurate)
            if pts3d_inl is not None and len(pts3d_inl) >= 5:
                nkf['obs2'].append({'rkf': kfw[-1],
                                    'p3':  pts3d_inl,
                                    'pr':  pts2d_p_inl,
                                    'pc':  pts2d_c_inl})

            # 3-view cross-window: KF[-3]→KF[-2]→current
            if len(kfw) >= 3:
                obs3 = make_3view_obs(kfw[-3], kfw[-2], Ic)
                if obs3 is not None:
                    nkf['obs3'].append(obs3)
            # Also KF[-2]→KF[-1]→current
            if len(kfw) >= 2:
                obs3b = make_3view_obs(kfw[-2], kfw[-1], Ic)
                if obs3b is not None:
                    nkf['obs3'].append(obs3b)

            kfw.append(nkf)
            if len(kfw) > WSIZE:
                kfw.pop(0)

            if len(kfw) >= 3:
                run_ba(kfw)
                poses[fi] = (kfw[-1]['R'].copy(), kfw[-1]['t'].copy())

        img_prev   = Ic
        pts_prev   = pts_cur
        depth_prev = dc

        if fi % 50 == 0:
            print(f"  frame {fi}/{N}  pos={poses[fi][1].round(2)}", file=sys.stderr)

    if len(kfw) >= 3:
        run_ba(kfw)

    traj_path  = out("traj.txt")
    poses_path = out("poses.txt")
    with open(traj_path, 'w') as ft, open(poses_path, 'w') as fp:
        for i in range(N):
            R, t = poses[i] or (np.eye(3), np.zeros(3))
            ft.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")
            fp.write(" ".join(f"{v:.8e}" for v in [
                R[0,0],R[0,1],R[0,2],t[0],
                R[1,0],R[1,1],R[1,2],t[1],
                R[2,0],R[2,1],R[2,2],t[2]]) + "\n")

    print(f"[VO] wrote {traj_path} and {poses_path}  ({N} frames)", file=sys.stderr)


# ── Main entry point ──────────────────────────────────────────────────────────
def main():
    # Check for multi-sequence mode: LAB_DATA contains seq_* subdirs
    seq_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if re.match(r'seq_\d+', d) and
           os.path.isdir(os.path.join(DATA_DIR, d)) and
           any(f.startswith('left_') for f in os.listdir(os.path.join(DATA_DIR, d))))

    if seq_dirs:
        # Process each sequence in turn
        for sd in seq_dirs:
            m = re.match(r'seq_?(\d+)', sd)
            sl = m.group(1) if m else sd
            print(f"[VO] Processing {sd} ...", file=sys.stderr)
            process_sequence(os.path.join(DATA_DIR, sd), sl)
    else:
        # Single sequence mode
        process_sequence(DATA_DIR, SEQ)

    print("DONE")

if __name__ == "__main__":
    main()
