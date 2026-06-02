#!/usr/bin/env python3
"""
Monocular Visual Odometry — robust keyframe-based pipeline

Pipeline:
  1. Use goodFeaturesToTrack + optical flow with skip=2 for reliable E estimation
  2. Fall back to wider baselines or SIFT when needed
  3. Properly interpolate intermediate frames between keyframes
  4. Output one "tx ty tz" per frame to $LAB_ARTIFACTS/traj.txt
"""

import os
import glob
import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = os.environ.get("LAB_DATA",      "/data")
ARTIFACTS_DIR = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Intrinsics ─────────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "intrinsics.txt")) as f:
    vals = [float(l.strip()) for l in f if l.strip()]
fx, fy, cx, cy = vals
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)
print(f"K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

# ── Load all frames ────────────────────────────────────────────────────────────
frame_paths = sorted(glob.glob(os.path.join(DATA_DIR, "frame_*.png")))
N = len(frame_paths)
print(f"Loading {N} frames …")
imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
print("Done loading.")

# ── Feature detector / SIFT backup ────────────────────────────────────────────
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
    {"checks": 100}
)

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def optical_flow_pose(img_src, img_dst, threshold=1.0):
    """Estimate relative pose via goodFeaturesToTrack + optical flow + E."""
    corners = cv2.goodFeaturesToTrack(
        img_src, maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7
    )
    if corners is None or len(corners) < 20:
        return None, None, 0

    p1, st, _ = cv2.calcOpticalFlowPyrLK(img_src, img_dst, corners, None, **LK_PARAMS)
    good = (st.ravel() == 1)
    if good.sum() < 20:
        return None, None, 0

    pts0 = corners[good].reshape(-1, 2)
    pts1 = p1[good].reshape(-1, 2)

    E, mask_E = cv2.findEssentialMat(
        pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=threshold
    )
    if E is None or mask_E is None or int(mask_E.sum()) < 8:
        return None, None, 0

    n, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask_E)
    return R, t, n


def sift_pose(img_src, img_dst):
    """Estimate relative pose via SIFT + FLANN + E."""
    kp1, des1 = sift.detectAndCompute(img_src, None)
    kp2, des2 = sift.detectAndCompute(img_dst, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, None, 0
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None, None, 0
    good = [m for pair in matches if len(pair) == 2
            and pair[0].distance < 0.75 * pair[1].distance
            for m in [pair[0]]]
    if len(good) < 8:
        return None, None, 0
    pts0 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp2[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask is None or int(mask.sum()) < 8:
        return None, None, 0
    n, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
    return R, t, n


def best_pose(i, j):
    """Try multiple methods to get the best pose estimate from frame i to j."""
    # 1. Optical flow (fast)
    R, t, n = optical_flow_pose(imgs[i], imgs[j])
    if n >= 20:
        return R, t, n
    # 2. Relaxed optical flow threshold
    R2, t2, n2 = optical_flow_pose(imgs[i], imgs[j], threshold=2.0)
    if n2 > n:
        R, t, n = R2, t2, n2
    if n >= 10:
        return R, t, n
    # 3. SIFT
    Rs, ts, ns = sift_pose(imgs[i], imgs[j])
    if ns > n:
        return Rs, ts, ns
    return R, t, n


# ── Build keyframe trajectory ───────────────────────────────────────────────────
# Strategy: process keyframe pairs (0→2, 2→4, …) for the main trajectory.
# For failed pairs, try wider baselines.
# After computing keyframe positions, interpolate all N frames.

STEP = 2           # primary skip
MIN_OK = 10        # accept pose if inliers >= MIN_OK

# Keyframes list: [(frame_idx, R_wc, C), ...]
kf_indices = [0]
kf_C       = [np.zeros(3)]

R_wc = np.eye(3, dtype=np.float64)
C    = np.zeros(3, dtype=np.float64)

i = 0
while i + 1 < N:
    next_i = i + STEP
    if next_i >= N:
        next_i = N - 1

    R_rel, t_rel, n = best_pose(i, next_i)

    # If still poor, try wider baselines
    if n < MIN_OK:
        for wider in [3, 4, 5, 6]:
            if i + wider >= N:
                break
            Rw, tw, nw = best_pose(i, i + wider)
            if nw > n:
                R_rel, t_rel, n = Rw, tw, nw
                next_i = i + wider
                if nw >= MIN_OK:
                    break

    if n >= MIN_OK and R_rel is not None:
        # X_dst = R_rel @ X_src + t_rel
        # → R_wc_new = R_wc_old @ R_rel^T
        # → C_new    = C_old − R_wc_new @ t_rel
        R_wc = R_wc @ R_rel.T
        C    = C - R_wc @ t_rel.ravel()

    kf_indices.append(next_i)
    kf_C.append(C.copy())

    i = next_i

# ── Interpolate all frames from keyframes ──────────────────────────────────────
kf_arr = np.array(kf_indices, dtype=float)
kf_C_arr = np.array(kf_C)          # shape (K, 3)

trajectory = []
for fi in range(N):
    # Find surrounding keyframes
    # np.searchsorted to locate position
    idx = np.searchsorted(kf_arr, fi)
    if idx == 0:
        pos = kf_C_arr[0]
    elif idx >= len(kf_arr):
        pos = kf_C_arr[-1]
    else:
        k0, k1 = int(kf_arr[idx-1]), int(kf_arr[idx])
        if k0 == k1:
            pos = kf_C_arr[idx]
        else:
            alpha = (fi - k0) / (k1 - k0)
            pos = (1.0 - alpha) * kf_C_arr[idx-1] + alpha * kf_C_arr[idx]
    trajectory.append(pos)

# ── Write trajectory ───────────────────────────────────────────────────────────
out_path = os.path.join(ARTIFACTS_DIR, "traj.txt")
with open(out_path, "w") as f:
    for pos in trajectory:
        f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

print(f"\nWrote {len(trajectory)} poses to {out_path}")
print("Sample keyframes and interpolated poses:")
for fi in [0, 1, 2, 10, 20, 50, 100, 150, 198, 199]:
    p = trajectory[fi]
    kf_mark = "*" if fi in kf_indices else " "
    print(f"  {kf_mark} Frame {fi:3d}: {p}")
