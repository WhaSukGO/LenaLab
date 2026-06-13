"""
Stereo Visual Odometry — metric scale via disparity depth
=========================================================
Pipeline (per frame):
  1. Load rectified left/right grayscale pair
  2. Compute dense disparity with StereoSGBM
  3. Detect / refresh FAST features on left image
  4. Track features left_prev → left_curr with KLT
  5. For each tracked feature lift its 3-D point from prev disparity
     Z = fx * baseline / disparity
  6. PnP-RANSAC (3D prev → 2D curr) → relative [R|t]
  7. Accumulate world pose:  T_world = T_world @ T_rel
Output:
  $LAB_ARTIFACTS/traj.txt   — one "tx ty tz" per frame (camera centre)
  $LAB_ARTIFACTS/poses.txt  — one 3×4 [R|t] cam→world matrix (12 numbers)
"""

import os
import glob
import numpy as np
import cv2

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR      = os.environ.get("LAB_DATA",      "/data")
ARTIFACTS_DIR = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── load intrinsics ────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "intrinsics.txt")) as f:
    vals = [float(x) for x in f.read().split()]
fx, fy, cx, cy, baseline = vals
print(f"Intrinsics: fx={fx} fy={fy} cx={cx} cy={cy} baseline={baseline}")

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# ── collect frame indices ──────────────────────────────────────────────────
left_files  = sorted(glob.glob(os.path.join(DATA_DIR, "left_*.png")))
right_files = sorted(glob.glob(os.path.join(DATA_DIR, "right_*.png")))
assert len(left_files) == len(right_files), "Mismatched left/right counts"
N = len(left_files)
print(f"Found {N} stereo frame pairs")

# ── StereoSGBM parameters (tuned for KITTI-like data) ─────────────────────
WIN_SIZE    = 5
MIN_DISP    = 0
NUM_DISP    = 128   # must be divisible by 16
SGBM = cv2.StereoSGBM_create(
    minDisparity      = MIN_DISP,
    numDisparities    = NUM_DISP,
    blockSize         = WIN_SIZE,
    P1                = 8  * 3 * WIN_SIZE**2,
    P2                = 32 * 3 * WIN_SIZE**2,
    disp12MaxDiff     = 1,
    uniquenessRatio   = 10,
    speckleWindowSize = 100,
    speckleRange      = 32,
    preFilterCap      = 63,
    mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

def compute_disparity(left_img, right_img):
    """Return float32 disparity map (invalid pixels → 0)."""
    disp = SGBM.compute(left_img, right_img).astype(np.float32) / 16.0
    disp[disp < 1.0] = 0          # mask invalid / zero disparity
    return disp

def disparity_to_depth(disp):
    """Z = fx * baseline / disparity, invalid where disp<=0."""
    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = fx * baseline / disp[valid]
    return depth

def lift_to_3d(pts2d, depth_map):
    """
    pts2d : (N,2) float32 pixel coords (x,y) in left image
    Returns pts3d (N,3) and boolean valid mask of length N.
    """
    H, W = depth_map.shape
    xs = np.round(pts2d[:, 0]).astype(int)
    ys = np.round(pts2d[:, 1]).astype(int)

    in_bounds = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    # clamp for safe indexing
    xc = np.clip(xs, 0, W-1)
    yc = np.clip(ys, 0, H-1)

    Z = depth_map[yc, xc]
    Z[~in_bounds] = 0.0

    valid = (Z > 0.5) & (Z < 80.0)

    X = (pts2d[:, 0] - cx) * Z / fx
    Y = (pts2d[:, 1] - cy) * Z / fy
    pts3d = np.column_stack([X, Y, Z])
    return pts3d, valid

# ── KLT parameters ─────────────────────────────────────────────────────────
LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FAST_DETECTOR = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

MIN_FEATURES  = 200
MAX_FEATURES  = 500

def detect_features(img):
    kps = FAST_DETECTOR.detect(img, None)
    # Sort by response, keep top MAX_FEATURES
    kps = sorted(kps, key=lambda k: -k.response)[:MAX_FEATURES]
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    return pts

def track_features(prev_img, curr_img, prev_pts):
    """KLT forward-backward tracking with outlier rejection."""
    if len(prev_pts) == 0:
        return np.array([]).reshape(0,2), np.array([]).reshape(0,2)

    pts0 = prev_pts.reshape(-1, 1, 2).astype(np.float32)

    # Forward
    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, pts0, None, **LK_PARAMS)
    # Backward
    pts0b, st0, _ = cv2.calcOpticalFlowPyrLK(curr_img, prev_img, pts1, None, **LK_PARAMS)

    # Accept if forward-backward error < 1 px
    fb_err = np.abs(pts0 - pts0b).reshape(-1, 2).max(axis=1)
    ok = (st1.ravel() == 1) & (st0.ravel() == 1) & (fb_err < 1.0)

    p0 = pts0.reshape(-1, 2)[ok]
    p1 = pts1.reshape(-1, 2)[ok]
    return p0, p1

# ── pose accumulation ──────────────────────────────────────────────────────
# T_wc: 4×4 homogeneous cam → world transform
# Starts at identity (camera 0 at world origin)
T_wc = np.eye(4)

traj_lines  = []
poses_lines = []

def record_pose(T):
    """Append current pose to output lists."""
    t = T[:3, 3]
    traj_lines.append(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}")
    row = " ".join(f"{v:.8f}" for v in T[:3, :].ravel())
    poses_lines.append(row)

# ── main loop ──────────────────────────────────────────────────────────────
prev_img   = None
prev_depth = None
prev_pts   = None

for idx in range(N):
    left_img  = cv2.imread(left_files[idx],  cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_files[idx], cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print(f"WARNING: could not read frame {idx}")
        record_pose(T_wc)
        continue

    # Compute disparity & depth for this frame
    disp  = compute_disparity(left_img, right_img)
    depth = disparity_to_depth(disp)

    # --- Frame 0: initialise -----------------------------------------------
    if prev_img is None:
        record_pose(T_wc)
        prev_img   = left_img
        prev_depth = depth
        prev_pts   = detect_features(left_img)
        print(f"Frame {idx:04d}: initialised with {len(prev_pts)} features")
        continue

    # --- Track features prev → curr ----------------------------------------
    p0, p1 = track_features(prev_img, left_img, prev_pts)

    motion_estimated = False
    if len(p0) >= 6:
        pts3d, valid3d = lift_to_3d(p0, prev_depth)
        good3d = pts3d[valid3d].astype(np.float64)
        good2d = p1[valid3d].astype(np.float64)

        if len(good3d) >= 6:
            try:
                dist = np.zeros(4)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    good3d, good2d, K, dist,
                    iterationsCount  = 200,
                    reprojectionError= 2.0,
                    confidence       = 0.999,
                    flags            = cv2.SOLVEPNP_ITERATIVE,
                )
                if success and inliers is not None and len(inliers) >= 6:
                    # Refine with inliers only
                    in_idx = inliers.ravel()
                    _, rvec, tvec = cv2.solvePnP(
                        good3d[in_idx], good2d[in_idx], K, dist,
                        rvec, tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    R_rel, _ = cv2.Rodrigues(rvec)
                    t_rel    = tvec.ravel()

                    step      = np.linalg.norm(t_rel)
                    rot_angle = np.linalg.norm(rvec)

                    # Reject implausible motion
                    if step < 5.0 and rot_angle < np.radians(30):
                        # solvePnP gives X_cam = R * X_world + t
                        # i.e. T_curr_prev maps prev-cam 3D → curr-cam frame
                        # T_wc[i] = T_wc[i-1] @ inv(T_curr_prev)
                        T_curr_prev = np.eye(4)
                        T_curr_prev[:3, :3] = R_rel
                        T_curr_prev[:3,  3] = t_rel

                        T_wc = T_wc @ np.linalg.inv(T_curr_prev)
                        motion_estimated = True
                        print(f"Frame {idx:04d}: {len(inliers):3d}/{len(good3d)} inliers "
                              f"step={step:.3f}m rot={np.degrees(rot_angle):.2f}°")
                    else:
                        print(f"Frame {idx:04d}: implausible motion "
                              f"step={step:.2f}m rot={np.degrees(rot_angle):.1f}° — skipped")
            except Exception as e:
                print(f"Frame {idx:04d}: PnP failed: {e}")

    if not motion_estimated:
        print(f"Frame {idx:04d}: motion NOT estimated "
              f"(tracked={len(p0)}, valid3d={int(valid3d.sum()) if len(p0)>=6 else 0})")

    record_pose(T_wc)

    # Refresh features when too few or motion failed
    if len(p1) < MIN_FEATURES or not motion_estimated:
        prev_pts = detect_features(left_img)
    else:
        prev_pts = p1

    prev_img   = left_img
    prev_depth = depth

print(f"\nDone. Recorded {len(traj_lines)} poses.")

# ── write outputs ──────────────────────────────────────────────────────────
traj_path  = os.path.join(ARTIFACTS_DIR, "traj.txt")
poses_path = os.path.join(ARTIFACTS_DIR, "poses.txt")

with open(traj_path, "w") as f:
    f.write("\n".join(traj_lines) + "\n")

with open(poses_path, "w") as f:
    f.write("\n".join(poses_lines) + "\n")

print(f"Wrote {traj_path}")
print(f"Wrote {poses_path}")
print(f"First pose: {traj_lines[0]}")
print(f"Last  pose: {traj_lines[-1]}")
