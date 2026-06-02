"""
RGB-D Visual Odometry — robust metric estimation using depth + PnP.

Pipeline per frame pair (in priority order):
  1. SIFT match (consecutive) → 3D-2D PnP RANSAC           [primary]
  2. KLT optical-flow (consecutive) → PnP                   [if SIFT gives <50 inliers]
  3. SIFT match against recent keyframes → PnP               [recovery fallback]

Depth back-projection provides absolute (metric) scale.
Output: camera-centre (tx ty tz) per frame in traj.txt.
"""
import cv2
import numpy as np
import os


def load_intrinsics(path):
    with open(path, 'r') as f:
        vals = f.read().strip().split()
    fx, fy, cx, cy, ds = [float(v) for v in vals]
    return fx, fy, cx, cy, ds


def backproject(pts_uv, depth_img, fx, fy, cx, cy, ds,
                min_z=0.1, max_z=10.0, sr=2):
    """
    Back-project 2-D pixel positions to 3-D using aligned depth image.
    pts_uv : (N, 2) float
    Returns pts3d (N, 3) float64 and valid mask (N,) bool.
    """
    h, w = depth_img.shape
    n = len(pts_uv)
    pts3d = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for k in range(n):
        u = float(pts_uv[k, 0])
        v = float(pts_uv[k, 1])
        ui = int(round(u))
        vi = int(round(v))
        d = 0
        if 0 <= vi < h and 0 <= ui < w:
            d = int(depth_img[vi, ui])
        if d == 0:
            # Search small neighbourhood (rows = vi, cols = ui)
            r0 = max(0, vi - sr); r1 = min(h, vi + sr + 1)
            c0 = max(0, ui - sr); c1 = min(w, ui + sr + 1)
            patch = depth_img[r0:r1, c0:c1]
            nz = patch[patch > 0]
            d = int(np.median(nz)) if len(nz) else 0
        if d == 0:
            continue
        z = d / ds
        if not (min_z <= z <= max_z):
            continue
        pts3d[k] = [(u - cx) * z / fx, (v - cy) * z / fy, z]
        valid[k] = True

    return pts3d, valid


def pnp(pts3d, pts2d, K, reproj=2.0, min_ni=8):
    """EPNP + iterative refinement. Returns (R, t, n_inliers) or (None,None,0)."""
    if len(pts3d) < min_ni:
        return None, None, 0
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        iterationsCount=500, reprojectionError=reproj,
        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_ni:
        return None, None, 0
    # Iterative refinement on inliers
    i3, i2 = pts3d[inl.ravel()], pts2d[inl.ravel()]
    _, rvec, tvec = cv2.solvePnP(
        i3, i2, K, None, rvec, tvec,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel(), len(inl)


def sift_corr(kp1, des1, kp2, des2, bf, ratio=0.75):
    """SIFT correspondences via Lowe ratio test. Returns (pts1, pts2) float32."""
    empty = np.empty((0, 2), np.float32)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return empty, empty
    knn = bf.knnMatch(des1, des2, k=2)
    p1, p2 = [], []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                p1.append(kp1[m.queryIdx].pt)
                p2.append(kp2[m.trainIdx].pt)
    if not p1:
        return empty, empty
    return np.array(p1, np.float32), np.array(p2, np.float32)


def klt_corr(img1, img2, max_pts=2000, win=21, lv=3, back_thr=1.0):
    """Shi-Tomasi corners in img1 tracked to img2 with backward check."""
    empty = np.empty((0, 2), np.float32)
    pts1 = cv2.goodFeaturesToTrack(
        img1, maxCorners=max_pts, qualityLevel=0.01, minDistance=7, blockSize=5)
    if pts1 is None or len(pts1) == 0:
        return empty, empty
    pts1 = pts1.reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    pts2, st1, _ = cv2.calcOpticalFlowPyrLK(
        img1, img2, pts1, None, winSize=(win, win), maxLevel=lv, criteria=crit)
    if pts2 is None:
        return empty, empty
    pts1b, st2, _ = cv2.calcOpticalFlowPyrLK(
        img2, img1, pts2, None, winSize=(win, win), maxLevel=lv, criteria=crit)
    diff = np.abs(pts1 - pts1b).reshape(-1, 2).max(axis=1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (diff < back_thr)
    return pts1.reshape(-1, 2)[good], pts2.reshape(-1, 2)[good]


def try_pose(pts_ref, pts_cur, dep_ref, fx, fy, cx, cy, ds,
             K, R_ref, t_ref, max_mot=0.8, reproj=2.0):
    """
    Back-project ref points → 3D (ref camera frame), solve PnP.
    R_ref/t_ref: world-to-ref-camera transform.
    Returns (R_world_to_curr, t_world_to_curr, n_inliers) or (None,None,0).
    """
    pts3d, valid = backproject(pts_ref, dep_ref, fx, fy, cx, cy, ds)
    pts3d_v = pts3d[valid]
    pts2d_v = pts_cur[valid].astype(np.float64)
    if len(pts3d_v) < 8:
        return None, None, 0
    R_rel, t_rel, ni = pnp(pts3d_v, pts2d_v, K, reproj)
    if R_rel is None or np.linalg.norm(t_rel) > max_mot:
        return None, None, 0
    return R_rel @ R_ref, R_rel @ t_ref + t_rel, ni


def main():
    data_dir = os.environ.get('LAB_DATA', '/data')
    art_dir = os.environ.get('LAB_ARTIFACTS', '/artifacts')

    fx, fy, cx, cy, ds = load_intrinsics(os.path.join(data_dir, 'intrinsics.txt'))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f'Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} ds={ds:.0f}')

    # Count frames
    n = 0
    while os.path.exists(os.path.join(data_dir, f'frame_{n:04d}.png')):
        n += 1
    print(f'Frames: {n}')

    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.03)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def load(idx):
        img = cv2.imread(os.path.join(data_dir, f'frame_{idx:04d}.png'), cv2.IMREAD_GRAYSCALE)
        dep = cv2.imread(os.path.join(data_dir, f'depth_{idx:04d}.png'), cv2.IMREAD_ANYDEPTH)
        return img, dep

    # Pose state: world = camera-0 frame
    R_wc = np.eye(3, dtype=np.float64)
    t_wc = np.zeros(3, dtype=np.float64)
    def cpos(): return -R_wc.T @ t_wc

    traj = [cpos().copy()]

    prev_img, prev_dep = load(0)
    prev_kp, prev_des = sift.detectAndCompute(prev_img, None)

    # Rolling keyframes — each stores current frame's data with current pose
    KF_MAX = 5
    kf_list = [{'dep': prev_dep, 'kp': prev_kp, 'des': prev_des,
                 'R': R_wc.copy(), 't': t_wc.copy(), 'idx': 0}]

    MAX_MOT = 0.8   # max motion per frame in metres (sanity cap)
    REPROJ = 2.0    # RANSAC reprojection threshold in pixels
    KLT_THR = 50    # use KLT only when SIFT gives fewer inliers

    for i in range(1, n):
        curr_img, curr_dep = load(i)
        if curr_img is None or curr_dep is None:
            traj.append(traj[-1].copy())
            continue

        curr_kp, curr_des = sift.detectAndCompute(curr_img, None)

        R_new = t_new = None
        ni_best = 0

        # --- Strategy 1: SIFT consecutive ---
        p1, p2 = sift_corr(prev_kp, prev_des, curr_kp, curr_des, bf)
        if len(p1) >= 8:
            R_s, t_s, ni_s = try_pose(
                p1, p2, prev_dep, fx, fy, cx, cy, ds,
                K, R_wc, t_wc, MAX_MOT, REPROJ)
            if R_s is not None and ni_s > ni_best:
                R_new, t_new, ni_best = R_s, t_s, ni_s

        # --- Strategy 2: KLT consecutive (only when SIFT is weak) ---
        if ni_best < KLT_THR:
            p1k, p2k = klt_corr(prev_img, curr_img)
            if len(p1k) >= 8:
                R_k, t_k, ni_k = try_pose(
                    p1k, p2k, prev_dep, fx, fy, cx, cy, ds,
                    K, R_wc, t_wc, MAX_MOT, REPROJ)
                if R_k is not None and ni_k > ni_best:
                    R_new, t_new, ni_best = R_k, t_k, ni_k

        # --- Strategy 3: SIFT against keyframes ---
        if R_new is None:
            for kf in kf_list:
                p1f, p2f = sift_corr(kf['kp'], kf['des'], curr_kp, curr_des, bf, 0.80)
                if len(p1f) < 8:
                    continue
                R_f, t_f, ni_f = try_pose(
                    p1f, p2f, kf['dep'], fx, fy, cx, cy, ds,
                    K, kf['R'], kf['t'], MAX_MOT, REPROJ)
                if R_f is not None and ni_f > ni_best:
                    R_new, t_new, ni_best = R_f, t_f, ni_f
                    break

        if R_new is not None:
            R_wc, t_wc = R_new, t_new
        else:
            print(f'Frame {i:04d}: ALL strategies failed')

        pos = cpos()
        traj.append(pos.copy())

        if i % 20 == 0:
            kf_pos = -kf_list[0]['R'].T @ kf_list[0]['t']
            print(f'Frame {i:04d}: ni={ni_best:4d}, '
                  f'({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}), '
                  f'd_kf={np.linalg.norm(pos - kf_pos):.3f}m')

        # --- Keyframe update: store current frame with current pose ---
        kf_pos = -kf_list[0]['R'].T @ kf_list[0]['t']
        d_kf = np.linalg.norm(pos - kf_pos)
        if d_kf > 0.12 or (i - kf_list[0]['idx']) >= 20:
            kf_list.insert(0, {
                'dep': curr_dep, 'kp': curr_kp, 'des': curr_des,
                'R': R_wc.copy(), 't': t_wc.copy(), 'idx': i
            })
            if len(kf_list) > KF_MAX:
                kf_list.pop()

        prev_img, prev_dep = curr_img, curr_dep
        prev_kp, prev_des = curr_kp, curr_des

    # --- Write output ---
    os.makedirs(art_dir, exist_ok=True)
    out = os.path.join(art_dir, 'traj.txt')
    with open(out, 'w') as f:
        for p in traj:
            f.write(f'{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n')

    arr = np.array(traj)
    dists = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    print(f'\nWrote {len(traj)} poses → {out}')
    print(f'Total dist: {dists.sum():.4f} m  max step: {dists.max()*1000:.1f} mm')
    print(f'Start: {traj[0]}  End: {traj[-1]}')


if __name__ == '__main__':
    main()
