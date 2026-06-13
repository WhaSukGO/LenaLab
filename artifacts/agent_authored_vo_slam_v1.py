#!/usr/bin/env python3
"""
RGB-D SLAM with Loop Closure.
Uses:
  1. ORB + PnP RGB-D odometry
  2. ORB descriptor voting + geometric verification for LC detection
  3. Sparse linear translation optimiser + iterative SE(3) pose graph refinement
"""

import numpy as np
import cv2
import os
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr as sparse_lsqr
from scipy.optimize import least_squares

DATA_DIR  = os.environ.get('LAB_DATA',      '/data')
ARTIFACTS = os.environ.get('LAB_ARTIFACTS', '/artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)

# ─── SE(3) helpers ────────────────────────────────────────────────────────────

def rvt2T(rv, tv):
    R, _ = cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3))
    T = np.eye(4); T[:3, :3] = R
    T[:3,  3] = np.asarray(tv, np.float64).ravel()
    return T

def T2rvt(T):
    rv, _ = cv2.Rodrigues(T[:3, :3].astype(np.float64))
    return rv.ravel().copy(), T[:3, 3].copy()

def invT(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -(R.T @ t)
    return Ti

def cam_center(T_cw):
    return -(T_cw[:3, :3].T @ T_cw[:3, 3])

def interp_T(Ta, Tb, alpha):
    """Linear axis-angle + translation interpolation."""
    ra, ta = T2rvt(Ta); rb, tb = T2rvt(Tb)
    return rvt2T(ra + alpha * (rb - ra), ta + alpha * (tb - ta))

# ─── RGB-D odometry ───────────────────────────────────────────────────────────

def rel_pose(rgb1, d1, rgb2, d2, fx, fy, cx, cy, ds,
             nfeat=1500, reproj=3.0):
    """p_cam2 = T @ p_cam1_hom.  Returns (T, n_inliers) or (None,0)."""
    g1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeat)
    kp1, dc1 = orb.detectAndCompute(g1, None)
    kp2, dc2 = orb.detectAndCompute(g2, None)
    if dc1 is None or dc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        mts = bf.knnMatch(dc1, dc2, k=2)
    except Exception:
        return None, 0
    good = [m for m, n in mts if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0
    p3, p2 = [], []
    h, w = d1.shape
    for m in good:
        u1, v1 = kp1[m.queryIdx].pt; u2, v2 = kp2[m.trainIdx].pt
        r, c = int(round(v1)), int(round(u1))
        r = max(0, min(h-1, r)); c = max(0, min(w-1, c))
        dv = d1[r, c] / ds
        if 0.1 < dv < 8.0:
            p3.append([(u1-cx)*dv/fx, (v1-cy)*dv/fy, dv])
            p2.append([u2, v2])
    if len(p3) < 6:
        return None, 0
    p3 = np.float32(p3); p2 = np.float32(p2)
    K = np.float32([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    try:
        ret, rv, tv, inl = cv2.solvePnPRansac(
            p3, p2, K, None,
            iterationsCount=200, reprojectionError=reproj,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        return None, 0
    if not ret or inl is None or len(inl) < 6:
        return None, 0
    nin = len(inl.ravel())
    try:
        _, rv2, tv2 = cv2.solvePnP(
            p3[inl.ravel()], p2[inl.ravel()], K, None,
            rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        rv, tv = rv2, tv2
    except Exception:
        pass
    return rvt2T(rv, tv), nin

# ─── Pose-graph optimisation ──────────────────────────────────────────────────

def pgo_linear_translations(kf_poses, edges):
    """
    Fix rotations (from raw VO), optimise only translations via sparse linear LS.
    Constraint for edge (i,j,T_ij,w):  t_j - R_ij @ t_i = t_ij
    with t_0 = 0 fixed.
    """
    n = len(kf_poses)
    n_vars = 3 * (n - 1)
    n_rows = 3 * len(edges)
    A = lil_matrix((n_rows, n_vars))
    b = np.zeros(n_rows)
    for ei, (i, j, Tm, w) in enumerate(edges):
        row = ei * 3
        R_ij = Tm[:3, :3]
        t_ij = Tm[:3, 3]
        if j > 0:
            A[row:row+3, (j-1)*3:(j)*3] = w * np.eye(3)
        if i > 0:
            A[row:row+3, (i-1)*3:(i)*3] = -w * R_ij
        # t_0 = kf_poses[0][:3,3] = 0 (world origin)
        b[row:row+3] = w * t_ij
    sol = sparse_lsqr(A.tocsr(), b, iter_lim=5000, damp=1e-6)
    x = sol[0]
    opt = [kf_poses[0].copy()]
    for i in range(1, n):
        T = kf_poses[i].copy()
        T[:3, 3] = x[(i-1)*3:(i)*3]
        opt.append(T)
    return opt


def pgo_se3(kf_poses, edges, max_nfev=1000):
    """
    Full SE(3) pose-graph optimisation via scipy TRF with sparse Jacobian.
    Falls back gracefully if it diverges.
    """
    n = len(kf_poses)
    if n <= 1:
        return list(kf_poses)
    T0 = kf_poses[0].copy()
    x0 = np.zeros(6 * (n - 1))
    for i in range(1, n):
        rv, tv = T2rvt(kf_poses[i])
        x0[(i-1)*6:(i-1)*6+3] = rv
        x0[(i-1)*6+3:(i-1)*6+6] = tv

    initial_cost_ref = [None]  # track initial cost

    def getT(x, i):
        if i == 0: return T0
        v = x[(i-1)*6:(i-1)*6+6]
        return rvt2T(v[:3], v[3:])

    call_count = [0]
    def resid(x):
        call_count[0] += 1
        Ts  = [T0] + [getT(x, i) for i in range(1, n)]
        iTs = [invT(T) for T in Ts]
        out = []
        for i, j, Tm, w in edges:
            Te = invT(Tm) @ Ts[j] @ iTs[i]
            re, _ = cv2.Rodrigues(Te[:3, :3].astype(np.float64))
            out.extend((w * re.ravel()).tolist())
            out.extend((w * Te[:3, 3]).tolist())
        return np.array(out, np.float64)

    r0 = resid(x0)
    c0 = 0.5 * np.sum(r0**2)
    initial_cost_ref[0] = c0
    print(f"    SE3 initial cost={c0:.3f}", flush=True)

    n_p = 6*(n-1); n_r = 6*len(edges)
    sp = lil_matrix((n_r, n_p), dtype=np.int8)
    for ei, (i, j, _, _) in enumerate(edges):
        r0e = ei * 6
        if i > 0: sp[r0e:r0e+6, (i-1)*6:(i-1)*6+6] = 1
        if j > 0: sp[r0e:r0e+6, (j-1)*6:(j-1)*6+6] = 1

    try:
        res = least_squares(resid, x0, method='trf', jac_sparsity=sp,
                            max_nfev=max_nfev,
                            ftol=1e-6, xtol=1e-6, gtol=1e-6,
                            verbose=0)
        xo = res.x
        cf = res.cost
        print(f"    SE3 final  cost={cf:.3f}  nfev={res.nfev}", flush=True)
        # Accept only if cost decreased
        if cf > c0 * 2.0:
            print("    SE3 diverged – keeping linear solution", flush=True)
            xo = x0
    except Exception as e:
        print(f"    SE3 exception: {e}", flush=True)
        xo = x0

    return [T0] + [getT(xo, i) for i in range(1, n)]


def optimise(kf_poses, edges):
    """Two-stage: linear translations first, then SE3 refinement."""
    print("  Stage 1: linear translation optimisation ...", flush=True)
    lin = pgo_linear_translations(kf_poses, edges)
    print("  Stage 2: SE(3) refinement ...", flush=True)
    opt = pgo_se3(lin, edges, max_nfev=2000)
    return opt

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    with open(os.path.join(DATA_DIR, 'intrinsics.txt')) as f:
        vals = f.read().split()
    fx, fy, cx, cy, ds = map(float, vals)
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} ds={ds}", flush=True)

    n_fr = 0
    while os.path.exists(os.path.join(DATA_DIR, f'frame_{n_fr:04d}.png')):
        n_fr += 1
    print(f"{n_fr} frames", flush=True)

    # ── Phase 1: odometry ────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Phase 1: odometry ...", flush=True)
    KF_EVERY = 10

    raw_poses = []
    kf_idx, kf_poses, kf_rgb, kf_dep = [], [], [], []

    rgb0 = cv2.imread(os.path.join(DATA_DIR, 'frame_0000.png'))
    dep0 = cv2.imread(os.path.join(DATA_DIR, 'depth_0000.png'), cv2.IMREAD_ANYDEPTH)
    raw_poses.append(np.eye(4))
    kf_idx.append(0); kf_poses.append(np.eye(4))
    kf_rgb.append(rgb0); kf_dep.append(dep0)

    prv_rgb, prv_dep, prv_T = rgb0, dep0, np.eye(4)

    for i in range(1, n_fr):
        rgb = cv2.imread(os.path.join(DATA_DIR, f'frame_{i:04d}.png'))
        dep = cv2.imread(os.path.join(DATA_DIR, f'depth_{i:04d}.png'), cv2.IMREAD_ANYDEPTH)
        Tr, _ = rel_pose(prv_rgb, prv_dep, rgb, dep, fx, fy, cx, cy, ds)
        if Tr is None:
            Tr = np.eye(4)
        cur_T = Tr @ prv_T
        raw_poses.append(cur_T)
        if i % KF_EVERY == 0 or i == n_fr - 1:
            kf_idx.append(i); kf_poses.append(cur_T)
            kf_rgb.append(rgb); kf_dep.append(dep)
        prv_rgb, prv_dep, prv_T = rgb, dep, cur_T

    n_kf = len(kf_idx)
    print(f"[{time.time()-t0:.1f}s] {n_kf} keyframes", flush=True)

    # ── Phase 2: KF descriptors ───────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Phase 2: descriptors ...", flush=True)
    orb_lc = cv2.ORB_create(nfeatures=2000)
    kf_desc = []
    for k in range(n_kf):
        g = cv2.cvtColor(kf_rgb[k], cv2.COLOR_BGR2GRAY)
        _, d = orb_lc.detectAndCompute(g, None)
        kf_desc.append(d)

    # ── Phase 3: loop closure detection ──────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Phase 3: loop closure ...", flush=True)
    MIN_KF_GAP  = 4
    MIN_VOTES   = 20
    MIN_INLIERS = 15

    bf_lc = cv2.BFMatcher(cv2.NORM_HAMMING)
    lc_edges = []

    for j in range(MIN_KF_GAP, n_kf):
        dj = kf_desc[j]
        if dj is None or len(dj) < 20:
            continue
        for i in range(j - MIN_KF_GAP):
            di = kf_desc[i]
            if di is None or len(di) < 20:
                continue
            try:
                mts = bf_lc.knnMatch(di, dj, k=2)
                votes = sum(1 for m, n in mts if m.distance < 0.75 * n.distance)
            except Exception:
                continue
            if votes < MIN_VOTES:
                continue
            Tr, nin = rel_pose(
                kf_rgb[i], kf_dep[i], kf_rgb[j], kf_dep[j],
                fx, fy, cx, cy, ds, nfeat=2000, reproj=4.0)
            if Tr is not None and nin >= MIN_INLIERS:
                lc_edges.append((i, j, Tr, nin))
                print(f"  LC KF{i}(fr{kf_idx[i]})<->KF{j}(fr{kf_idx[j]}) "
                      f"in={nin}", flush=True)

    print(f"[{time.time()-t0:.1f}s] {len(lc_edges)} loop closures", flush=True)

    # ── Phase 4: pose-graph optimisation ─────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Phase 4: optimise ...", flush=True)

    # Sequential edges
    edges = []
    for j in range(1, n_kf):
        i = j - 1
        T_ij = kf_poses[j] @ invT(kf_poses[i])
        edges.append((i, j, T_ij, 1.0))

    # LC edges with higher weight
    for i, j, Tr, nin in lc_edges:
        w = min(4.0, nin / float(MIN_INLIERS))
        edges.append((i, j, Tr, w))

    if lc_edges:
        opt_kf = optimise(kf_poses, edges)
    else:
        print("  No LC – raw VO only", flush=True)
        opt_kf = list(kf_poses)
    print(f"[{time.time()-t0:.1f}s] optimisation done", flush=True)

    # ── Phase 5: propagate to all frames ─────────────────────────────────────
    all_T = [None] * n_fr
    for k, fi in enumerate(kf_idx):
        all_T[fi] = opt_kf[k]

    for k in range(n_kf - 1):
        fi = kf_idx[k]; fj = kf_idx[k + 1]
        if fj - fi <= 1:
            continue
        ci = opt_kf[k]   @ invT(raw_poses[fi])
        cj = opt_kf[k+1] @ invT(raw_poses[fj])
        ns = fj - fi
        for f in range(fi + 1, fj):
            a = (f - fi) / ns
            all_T[f] = interp_T(ci, cj, a) @ raw_poses[f]

    for i in range(n_fr):
        if all_T[i] is None:
            all_T[i] = raw_poses[i]

    # ── Output ───────────────────────────────────────────────────────────────
    out = os.path.join(ARTIFACTS, 'traj.txt')
    centres = []
    with open(out, 'w') as fout:
        for i in range(n_fr):
            c = cam_center(all_T[i])
            centres.append(c)
            fout.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
    centres = np.array(centres)
    plen = np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1))
    print(f"Wrote {n_fr} lines → {out}  path≈{plen:.2f}m", flush=True)
    print(f"Total: {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()
