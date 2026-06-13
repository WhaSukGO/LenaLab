"""
Loop-closure + pose-graph optimisation for stereo visual SLAM on KITTI.
Uses the locked frontend (frontend.py) for VO, adds BoW loop detection,
PnP RANSAC geometric verification, and SE(3) Gauss-Newton pose-graph opt.
"""
import os, sys
import numpy as np
import cv2
from scipy.spatial.distance import cdist

# ── helpers: Lie algebra / SE(3) ──────────────────────────────────────────────

def skew(v):
    v = v.ravel()
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def rodrigues(w):
    """so(3) → SO(3)  (w is a 3-vec)"""
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    n = w / theta
    K = skew(n)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def log_SO3(R):
    """SO(3) → so(3) axis-angle 3-vec"""
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if theta < 1e-9:
        return np.zeros(3)
    return theta/(2*np.sin(theta)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def pose_from_xi(xi):
    """se(3) 6-vec [rho, omega] → 4×4"""
    rho = xi[:3]; omega = xi[3:]
    R = rodrigues(omega)
    theta = np.linalg.norm(omega)
    if theta < 1e-9:
        V = np.eye(3)
    else:
        n = omega/theta; K = skew(n)
        V = (np.eye(3) + (1-np.cos(theta))/theta*K + (theta-np.sin(theta))/theta*(K@K))
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=V@rho
    return T

def log_SE3(T):
    """4×4 → se(3) 6-vec [rho, omega]"""
    R = T[:3,:3]; t = T[:3,3]
    omega = log_SO3(R)
    theta = np.linalg.norm(omega)
    if theta < 1e-9:
        V_inv = np.eye(3)
    else:
        n = omega/theta; K = skew(n)
        V_inv = (np.eye(3) - 0.5*theta*K +
                 (1 - theta/(2*np.tan(theta/2)))*(K@K))
    rho = V_inv @ t
    return np.concatenate([rho, omega])

# ── BoW vocabulary (flat k-means) ─────────────────────────────────────────────

def build_vocab(all_des, k=256, max_iter=30):
    all_des = all_des.astype(np.float32)
    N = len(all_des)
    if N < k:
        k = max(8, N//4)
    if N > 50000:
        idx = np.random.choice(N, 50000, replace=False)
        sample = all_des[idx]
    else:
        sample = all_des
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers  # (k, 32)

def compute_bow(des, centers, idf=None):
    if des is None or len(des) == 0:
        return np.zeros(len(centers))
    des_f = des.astype(np.float32)
    dists = cdist(des_f, centers.astype(np.float32), metric='euclidean')
    words = np.argmin(dists, axis=1)
    hist = np.bincount(words, minlength=len(centers)).astype(np.float64)
    if idf is not None:
        hist = hist * idf
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    return hist

def build_idf(frames, centers):
    k = len(centers)
    df = np.zeros(k)
    N = 0
    for fr in frames:
        des = fr['des']
        if des is None or len(des) == 0:
            continue
        N += 1
        des_f = des.astype(np.float32)
        dists = cdist(des_f, centers.astype(np.float32), metric='euclidean')
        words = np.argmin(dists, axis=1)
        df[np.unique(words)] += 1
    idf = np.log((N + 1) / (df + 1))
    return idf

# ── Loop detection ─────────────────────────────────────────────────────────────

def detect_loops(frames, centers, idf, min_gap=30, top_k=5):
    n = len(frames)
    bows = []
    for fr in frames:
        bows.append(compute_bow(fr['des'], centers, idf))
    bows = np.array(bows)  # (n, k)

    candidates = []
    for i in range(min_gap, n):
        diffs = np.abs(bows[i] - bows[:i-min_gap+1]).sum(axis=1)
        scores = 2.0 - diffs  # higher = more similar
        best_k_idx = np.argsort(scores)[::-1][:top_k]
        for j in best_k_idx:
            s = scores[j]
            if s > 1.5:  # L1 sim > 0.75 (since max is 2.0)
                candidates.append((i, int(j), float(s)))
    return candidates

# ── Geometric verification via PnP RANSAC ─────────────────────────────────────

def verify_loop(fr_i, fr_j, K, inlier_thresh=15):
    """
    Verify loop: fr_i is the query (later), fr_j is the match (earlier).
    Uses 3D-2D: 3D points from fr_j matched to 2D keypoints in fr_i.
    Returns (True, T_cam_ij, n_inliers) or (False, None, 0).
    T_cam_ij: inv(Twc_i) @ Twc_j  (maps 3D pts in j-cam to i-cam coords).
    """
    des_i = fr_i['des']; des_j = fr_j['des']
    if des_i is None or des_j is None:
        return False, None, 0
    if len(des_i) < 10 or len(des_j) < 10:
        return False, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_j, des_i)  # query=j, train=i
    if len(matches) < inlier_thresh:
        return False, None, 0

    pidx_j_set = {int(p): row for row, p in enumerate(fr_j['pidx'])}
    pts3d_j = fr_j['pts3d']
    kps_i   = fr_i['kps']

    obj_pts, img_pts = [], []
    for m in matches:
        qi = m.queryIdx  # kp index in j
        ti = m.trainIdx  # kp index in i
        if qi in pidx_j_set:
            obj_pts.append(pts3d_j[pidx_j_set[qi]])
            img_pts.append(kps_i[ti])

    if len(obj_pts) < inlier_thresh:
        return False, None, 0

    obj_pts = np.array(obj_pts, np.float64)
    img_pts = np.array(img_pts, np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, None,
        reprojectionError=3.0, iterationsCount=200,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)

    if not ok or inliers is None or len(inliers) < inlier_thresh:
        return False, None, 0

    R, _ = cv2.Rodrigues(rvec)
    # p_cam_i = R @ p_cam_j + t  => T_cam_ij maps j-cam pts to i-cam
    T_cam_ij = np.eye(4)
    T_cam_ij[:3,:3] = R
    T_cam_ij[:3,3]  = tvec.ravel()

    return True, T_cam_ij, int(len(inliers))

# ── Pose-graph optimisation (SE(3) Gauss-Newton with LM damping) ──────────────

def pose_graph_optimize(poses_init, odom_edges, loop_edges, n_iter=50):
    """
    poses_init : list of n 4×4 Twc matrices (cam->world)
    odom_edges : list of (i, j, T_ij)  where T_ij = inv(Twc_i) @ Twc_j
    loop_edges : list of (i, j, T_ij, weight)  same convention
    """
    n = len(poses_init)
    poses = [p.copy() for p in poses_init]

    odom_w     = 1.0
    loop_w_base = 5.0

    def build_system(poses):
        dim = 6*(n-1)
        H = np.zeros((dim, dim))
        b = np.zeros(dim)
        total_res = 0.0

        def add_edge(i, j, T_ij_meas, weight):
            nonlocal total_res
            T_ij_pred = np.linalg.inv(poses[i]) @ poses[j]
            T_err = np.linalg.inv(T_ij_meas) @ T_ij_pred
            err = log_SE3(T_err)
            total_res += weight * float(np.dot(err, err))
            w = weight
            if i > 0:
                ii = (i-1)*6
                H[ii:ii+6, ii:ii+6] += w * np.eye(6)
                b[ii:ii+6] += w * err
            if j > 0:
                jj = (j-1)*6
                H[jj:jj+6, jj:jj+6] += w * np.eye(6)
                b[jj:jj+6] -= w * err
            if i > 0 and j > 0:
                ii = (i-1)*6; jj = (j-1)*6
                H[ii:ii+6, jj:jj+6] -= w * np.eye(6)
                H[jj:jj+6, ii:ii+6] -= w * np.eye(6)

        for (i, j, T_ij) in odom_edges:
            add_edge(i, j, T_ij, odom_w)
        for (i, j, T_ij, wt) in loop_edges:
            add_edge(i, j, T_ij, loop_w_base * wt)

        return H, b, total_res

    lam = 1e-4
    prev_res = None

    for it in range(n_iter):
        H, b, res = build_system(poses)
        diag_H = np.diag(H).copy()
        H_damp = H + lam * np.diag(np.where(diag_H > 0, diag_H, 1.0))
        try:
            dx = np.linalg.solve(H_damp, b)
        except np.linalg.LinAlgError:
            print(f"  iter {it}: singular, stopping")
            break

        new_poses = [poses[0].copy()]
        for k in range(1, n):
            xi = dx[(k-1)*6 : k*6]
            dT = pose_from_xi(xi)
            new_poses.append(poses[k] @ dT)

        _, _, new_res = build_system(new_poses)
        if new_res < res - 1e-9:
            poses = new_poses
            lam = max(lam / 3.0, 1e-9)
            print(f"  iter {it:3d}: res {res:.4f} -> {new_res:.4f}  lam={lam:.2e}  ACCEPT")
            if prev_res is not None and abs(prev_res - new_res) < 1e-4 * abs(prev_res):
                print("  Converged.")
                break
            prev_res = new_res
        else:
            lam = min(lam * 5.0, 1e6)
            print(f"  iter {it:3d}: res {res:.4f}  lam={lam:.2e}  REJECT")
            if lam >= 1e6:
                print("  Lambda saturated, stopping.")
                break

    return poses

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.environ['LAB_DATA']
    art_dir  = os.environ['LAB_ARTIFACTS']
    os.makedirs(art_dir, exist_ok=True)

    print("Running frontend VO...")
    sys.path.insert(0, os.environ.get('LAB_CODE', '/code'))
    from frontend import run_frontend
    fe = run_frontend(data_dir)

    n      = fe['n']
    K      = fe['K']
    frames = fe['frames']
    poses  = fe['poses']   # list of n 4×4 Twc

    print(f"Frontend done: {n} frames")

    # ── 1. Build BoW vocabulary ────────────────────────────────────────────────
    print("Building BoW vocabulary...")
    all_des = []
    for fr in frames:
        if fr['des'] is not None and len(fr['des']) > 0:
            all_des.append(fr['des'])
    if not all_des:
        print("No descriptors – writing frontend output")
        _write_output(art_dir, poses)
        return
    all_des = np.vstack(all_des)
    print(f"  Total descriptors: {len(all_des)}")

    np.random.seed(42)
    centers = build_vocab(all_des, k=256)
    idf     = build_idf(frames, centers)
    print(f"  Vocabulary: {len(centers)} words")

    # ── 2. Loop detection ──────────────────────────────────────────────────────
    print("Detecting loop candidates...")
    candidates = detect_loops(frames, centers, idf, min_gap=30, top_k=5)
    print(f"  {len(candidates)} candidates before geometric verification")

    # ── 3. Geometric verification ──────────────────────────────────────────────
    print("Verifying loops...")
    loop_edges_raw = []
    seen = set()
    for (i, j, score) in sorted(candidates, key=lambda x: -x[2]):
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        ok, T_cam_ij, n_inl = verify_loop(frames[i], frames[j], K, inlier_thresh=15)
        if ok:
            weight = min(n_inl / 50.0, 3.0)
            loop_edges_raw.append((i, j, T_cam_ij, weight))
            print(f"  LOOP {j:4d}→{i:4d}: {n_inl} inliers, score={score:.3f}, w={weight:.2f}")

    print(f"  {len(loop_edges_raw)} verified loops")

    if len(loop_edges_raw) == 0:
        print("No loops verified – writing frontend output unchanged")
        _write_output(art_dir, poses)
        return

    # ── 4. Build edges ─────────────────────────────────────────────────────────
    odom_edges = []
    for i in range(n-1):
        T_ij = np.linalg.inv(poses[i]) @ poses[i+1]
        odom_edges.append((i, i+1, T_ij))

    # Convert loop edges:
    # T_cam_ij: p_cam_i = T_cam_ij @ p_cam_j  (3D pts from j to i cam frame)
    # We showed: T_cam_ij = inv(Twc_i) @ Twc_j
    # Edge convention: T_ij = inv(Twc_i) @ Twc_j
    # So for edge (j->i): T_{j,i} = inv(Twc_j) @ Twc_i = inv(T_cam_ij)
    # For edge (i->j) which we need since j < i:
    # Actually we can use edge (j, i, T_ji) where T_ji = inv(T_cam_ij)
    loop_edges = []
    for (i, j, T_cam_ij, w) in loop_edges_raw:
        # i is later frame, j is earlier. T_cam_ij = inv(Twc_i) @ Twc_j
        # We add edge from j to i: T_{j->i} = inv(Twc_j) @ Twc_i = inv(T_cam_ij)
        T_ji = np.linalg.inv(T_cam_ij)
        loop_edges.append((j, i, T_ji, w))

    # ── 5. Pose-graph optimisation ─────────────────────────────────────────────
    print(f"Optimising pose graph ({n} nodes, {len(odom_edges)} odom, {len(loop_edges)} loop)...")
    opt_poses = pose_graph_optimize(poses, odom_edges, loop_edges, n_iter=50)

    # ── 6. Write outputs ───────────────────────────────────────────────────────
    _write_output(art_dir, opt_poses)
    print("Done.")

def _write_output(art_dir, poses):
    traj_arr  = np.array([p[:3,3] for p in poses])
    poses_arr = np.array([p[:3,:4].ravel() for p in poses])
    np.savetxt(os.path.join(art_dir, 'traj.txt'),  traj_arr,  fmt='%.6f')
    np.savetxt(os.path.join(art_dir, 'poses.txt'), poses_arr, fmt='%.8e')
    print(f"Written {len(traj_arr)} rows to traj.txt and poses.txt")

if __name__ == '__main__':
    main()
