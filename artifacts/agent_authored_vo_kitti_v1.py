"""
Stereo Visual Odometry for KITTI-style outdoor driving.

Pipeline per frame pair (i-1 → i):
  1. SGBM disparity → metric depth map of frame i-1  (Z = fx*b/d)
  2. ORB features detected on both left frames, matched with BF+crossCheck
     (handles 100+ px inter-frame motions robustly; no accumulated drift)
  3. Back-project matched points in frame i-1 to 3-D via depth map
  4. PnP+RANSAC → metric R, t
  5. Iterative refinement on inliers; constant-velocity fallback if PnP fails.

Output: $LAB_ARTIFACTS/traj.txt  — one "tx ty tz" camera-centre per frame.
"""

import cv2
import numpy as np
import os
import glob


# ──────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────

def load_intrinsics(path):
    with open(path) as f:
        vals = [float(x) for x in f.read().split()]
    fx, fy, cx, cy, baseline = vals
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, fx, fy, cx, cy, baseline


def load_pair(data_dir, idx):
    l = cv2.imread(os.path.join(data_dir, f'left_{idx:06d}.png'),
                   cv2.IMREAD_GRAYSCALE)
    r = cv2.imread(os.path.join(data_dir, f'right_{idx:06d}.png'),
                   cv2.IMREAD_GRAYSCALE)
    assert l is not None and r is not None, f"Missing frame {idx}"
    return l, r


# ──────────────────────────────────────────────────────────────
# Stereo depth  (SGBM)
# ──────────────────────────────────────────────────────────────

def _make_sgbm():
    b = 5
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=b,
        P1=8 * b * b, P2=32 * b * b,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


_SGBM = _make_sgbm()


def compute_depth(l, r, fx, baseline):
    disp = _SGBM.compute(l, r).astype(np.float32) / 16.0
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disp > 1.0, fx * baseline / disp, 0.0)
    return depth.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Feature matching  (ORB + BFMatcher crossCheck)
# ──────────────────────────────────────────────────────────────

_ORB = cv2.ORB_create(nfeatures=3000)
_BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def match_frames(img0, img1):
    """
    Detect ORB features on img0 and img1, return matched 2-D point pairs.
    Returns (pts0, pts1) both as float64 (N×2).  May return empty arrays.
    """
    kp0, des0 = _ORB.detectAndCompute(img0, None)
    kp1, des1 = _ORB.detectAndCompute(img1, None)
    if des0 is None or des1 is None or len(des0) < 6 or len(des1) < 6:
        return np.empty((0, 2)), np.empty((0, 2))

    matches = list(_BF.match(des0, des1))
    matches.sort(key=lambda m: m.distance)

    pts0 = np.array([kp0[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts1 = np.array([kp1[m.trainIdx].pt for m in matches], dtype=np.float64)
    return pts0, pts1


# ──────────────────────────────────────────────────────────────
# 3-D / 2-D correspondence builder
# ──────────────────────────────────────────────────────────────

def build_correspondences(pts0, pts1, depth, fx, fy, cx, cy,
                          min_d=1.5, max_d=80.0):
    """Vectorised: back-project pts0 to 3-D using depth; pair with pts1."""
    H, W = depth.shape
    us = np.round(pts0[:, 0]).astype(np.int32)
    vs = np.round(pts0[:, 1]).astype(np.int32)
    # Bounds mask
    in_b = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    if not np.any(in_b):
        return np.empty((0, 3)), np.empty((0, 2))
    # Depth look-up
    d_vals = np.zeros(len(pts0), dtype=np.float32)
    d_vals[in_b] = depth[vs[in_b], us[in_b]]
    valid = in_b & (d_vals > min_d) & (d_vals < max_d)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 2))
    dv = d_vals[valid].astype(np.float64)
    uv = us[valid].astype(np.float64)
    vv = vs[valid].astype(np.float64)
    obj = np.column_stack([(uv - cx) * dv / fx,
                           (vv - cy) * dv / fy,
                           dv])
    return obj, pts1[valid]


# ──────────────────────────────────────────────────────────────
# Pose estimation  (PnP RANSAC + iterative refinement)
# ──────────────────────────────────────────────────────────────

def estimate_pose(obj_pts, img_pts, K, dist, reproj=2.0):
    if len(obj_pts) < 6:
        return None, None, 0

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj_pts, img_pts, K, dist,
        confidence=0.9999, reprojectionError=reproj, iterationsCount=500,
    )
    if not ok or inl is None or len(inl) < 6:
        return None, None, 0

    idx = inl.flatten()
    # Iterative refinement on inliers
    ok2, rv2, tv2 = cv2.solvePnP(
        obj_pts[idx], img_pts[idx], K, dist,
        rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok2:
        rvec, tvec = rv2, tv2

    R_rel, _ = cv2.Rodrigues(rvec)
    return R_rel, tvec, len(idx)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    data_dir      = os.environ.get('LAB_DATA',      '/data')
    artifacts_dir = os.environ.get('LAB_ARTIFACTS', '/artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)

    K, fx, fy, cx, cy, baseline = load_intrinsics(
        os.path.join(data_dir, 'intrinsics.txt'))
    dist = np.zeros(4, dtype=np.float64)

    left_imgs = sorted(glob.glob(os.path.join(data_dir, 'left_*.png')))
    n = len(left_imgs)
    print(f"Frames={n}  fx={fx:.2f}  baseline={baseline:.4f}m", flush=True)

    # Accumulated pose:  P_cam = R @ P_world + t
    R = np.eye(3, dtype=np.float64)
    t = np.zeros((3, 1), dtype=np.float64)

    def cam_pos():
        return (-R.T @ t).flatten().copy()

    trajectory = []

    # Constant-velocity fallback buffer
    VEL_BUF = 5
    vel_buf = []   # list of (R_rel, t_rel)

    def median_vel():
        if not vel_buf:
            return np.eye(3), np.zeros((3, 1))
        ts = np.array([v[1].flatten() for v in vel_buf])
        return vel_buf[-1][0].copy(), np.median(ts, axis=0, keepdims=True).T

    # Load frame 0
    l0, r0 = load_pair(data_dir, 0)
    depth0  = compute_depth(l0, r0, fx, baseline)
    trajectory.append(cam_pos())

    n_failed = 0

    for i in range(1, n):
        l1, r1 = load_pair(data_dir, i)

        # ── Match ────────────────────────────────────────────
        pts0, pts1 = match_frames(l0, l1)

        # ── 3-D / 2-D ────────────────────────────────────────
        obj_pts, img_pts = build_correspondences(
            pts0, pts1, depth0, fx, fy, cx, cy)

        # ── Pose ─────────────────────────────────────────────
        R_rel, t_rel, n_inl = estimate_pose(obj_pts, img_pts, K, dist)

        pose_ok = False
        if R_rel is not None:
            ang   = float(np.arccos(
                np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
            trans = float(np.linalg.norm(t_rel))
            if ang < 0.7 and trans < 10.0:
                t = R_rel @ t + t_rel
                R = R_rel @ R
                U, _, Vt = np.linalg.svd(R)
                R = U @ Vt
                vel_buf.append((R_rel.copy(), t_rel.copy()))
                if len(vel_buf) > VEL_BUF:
                    vel_buf.pop(0)
                pose_ok = True

        if not pose_ok:
            R_fb, t_fb = median_vel()
            t = R_fb @ t + t_fb
            R = R_fb @ R
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            n_failed += 1

        trajectory.append(cam_pos())

        if i % 50 == 0:
            pos = trajectory[-1]
            print(f"  [{i:4d}/{n}] pos=({pos[0]:+8.2f},{pos[1]:+7.2f},"
                  f"{pos[2]:+8.2f})  3D={len(obj_pts)}  inl={n_inl}  fail={n_failed}",
                  flush=True)

        # ── Next frame ───────────────────────────────────────
        l0     = l1
        depth0 = compute_depth(l1, r1, fx, baseline)

    # ── Write ─────────────────────────────────────────────────
    traj_path = os.path.join(artifacts_dir, 'traj.txt')
    with open(traj_path, 'w') as f:
        for pos in trajectory:
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    print(f"\nWrote {len(trajectory)} poses → {traj_path}")
    print(f"Start: {trajectory[0]}")
    print(f"End:   {trajectory[-1]}")


if __name__ == '__main__':
    main()
