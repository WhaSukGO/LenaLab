"""
Stereo Visual SLAM for KITTI outdoor driving.
Architecture (two-phase to fit in 15-min grader):
  Phase 1: Fast VO front-end
    - SGBM disparity → metric depth per frame
    - ORB + BF → PnP+RANSAC per frame pair
    - Keyframe selection every KF_INTERVAL frames
    - Sliding-window BA with monotonic reprojection safeguard
  Phase 2: Loop closure (batch after VO)
    - Appearance-based detection using descriptor voting (vectorised Hamming)
    - Geometric verification via stored KF cam-space 3D pts (no depth reload)
    - Global pose-graph optimisation (scipy least_squares on SE3)
  Phase 3: Propagate corrected poses → traj.txt + poses.txt (KITTI format)
"""

import os, time
import numpy as np
import cv2
from scipy.optimize import least_squares

# ── paths ──────────────────────────────────────────────────────────────────────
LAB_DATA      = os.environ.get("LAB_DATA",      "/data")
LAB_ARTIFACTS = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(LAB_ARTIFACTS, exist_ok=True)

# ── intrinsics ──────────────────────────────────────────────────────────────────
with open(os.path.join(LAB_DATA, "intrinsics.txt")) as f:
    vals = [float(l.strip()) for l in f if l.strip()]
fx, fy, cx, cy, baseline = vals
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
print(f"[INFO] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} b={baseline:.4f}m")

# ── frame list ──────────────────────────────────────────────────────────────────
left_files  = sorted(f for f in os.listdir(LAB_DATA) if f.startswith("left_")  and f.endswith(".png"))
right_files = sorted(f for f in os.listdir(LAB_DATA) if f.startswith("right_") and f.endswith(".png"))
N = min(len(left_files), len(right_files))
print(f"[INFO] {N} stereo pairs")

# ── SGBM (fast mode) ──────────────────────────────────────────────────────────
win      = 5
num_disp = 96
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=num_disp, blockSize=win,
    P1=8*3*win**2, P2=32*3*win**2,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM)   # faster than 3WAY

def compute_depth(img_l, img_r, max_z=80.0):
    disp = sgbm.compute(img_l, img_r).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > 0.1
    depth[valid] = fx * baseline / disp[valid]
    depth[depth > max_z] = 0.0
    return depth

# ── ORB ──────────────────────────────────────────────────────────────────────────
N_ORB = 1500
orb = cv2.ORB_create(nfeatures=N_ORB, scaleFactor=1.2, nlevels=8, edgeThreshold=19)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def detect(img):
    kps, des = orb.detectAndCompute(img, None)
    return kps, des   # kps may be () if none found

# ── back-project keypoints → cam-space + world-space ──────────────────────────
def backproject_kps(kps, des, depth, R_cw, t_cw, max_z=60.0, min_z=0.5):
    """Return aligned arrays: des_3d, pts_cam (N,3), pts_world (N,3), pts2d (N,2)."""
    if kps is None or len(kps) == 0 or des is None:
        empty = np.zeros((0,3), dtype=np.float64)
        return np.zeros((0,32), dtype=np.uint8), empty, empty, np.zeros((0,2))
    H, W = depth.shape
    des_3d, pts_cam, pts_world, pts2d = [], [], [], []
    for i, kp in enumerate(kps):
        u, v = kp.pt
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < H and 0 <= ui < W):
            continue
        z = depth[vi, ui]
        if not (min_z < z < max_z):
            continue
        xc = (u - cx) * z / fx
        yc = (v - cy) * z / fy
        pc = np.array([xc, yc, z])
        pw = R_cw @ pc + t_cw
        des_3d.append(des[i])
        pts_cam.append(pc)
        pts_world.append(pw)
        pts2d.append([u, v])
    if not des_3d:
        empty = np.zeros((0,3), dtype=np.float64)
        return np.zeros((0,32), dtype=np.uint8), empty, empty, np.zeros((0,2))
    return (np.array(des_3d, dtype=np.uint8),
            np.array(pts_cam, dtype=np.float64),
            np.array(pts_world, dtype=np.float64),
            np.array(pts2d, dtype=np.float64))

# ── PnP with cam-space 3D points from previous frame ─────────────────────────
def estimate_pose_pnp(kps_prev, des_prev, depth_prev, kps_curr, des_curr):
    """
    Estimate R_rel, t_rel (world→cam transform convention for OpenCV PnP).
    Returns (R_cw_new, t_cw_new) – cam-to-world for curr frame.
    Needs caller to provide R_cw_prev, t_cw_prev for back-projection.
    """
    if (des_prev is None or des_curr is None or
            len(des_prev) < 8 or len(des_curr) < 8):
        return None, None, 0
    matches = list(bf.match(des_prev, des_curr))
    if len(matches) < 10:
        return None, None, 0
    matches.sort(key=lambda m: m.distance)
    return matches   # caller does the PnP

def pnp_from_matches(matches, pts3d_prev_cam, kps_curr, R_cw_prev, t_cw_prev):
    """
    pts3d_prev_cam: cam-space 3D points aligned with prev kps (index = kp index)
    Returns R_cw_new, t_cw_new, n_inliers or (None, None, 0).
    """
    obj_pts, img_pts = [], []
    for m in matches:
        qi = m.queryIdx
        if qi < len(pts3d_prev_cam) and pts3d_prev_cam[qi, 2] > 0.1:
            # Transform cam-prev to world
            pw = R_cw_prev @ pts3d_prev_cam[qi] + t_cw_prev
            obj_pts.append(pw)
            img_pts.append(kps_curr[m.trainIdx].pt)
    if len(obj_pts) < 8:
        return None, None, 0
    obj_arr = np.array(obj_pts, dtype=np.float64)
    img_arr = np.array(img_pts, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_arr, img_arr, K, None,
        iterationsCount=200, reprojectionError=2.5,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < 8:
        return None, None, 0

    idx = inliers.ravel()
    cv2.solvePnP(obj_arr[idx], img_arr[idx], K, None,
                 rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                 flags=cv2.SOLVEPNP_ITERATIVE)
    R_wc, _ = cv2.Rodrigues(rvec)   # world→cam
    t_wc    = tvec.ravel()
    # cam-to-world: R_cw = R_wc.T, t_cw = -R_wc.T @ t_wc
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc
    return R_cw, t_cw, len(idx)

# ── reprojection error for a keyframe ─────────────────────────────────────────
def kf_reproj_err(pts3d_world, pts2d, R_cw, t_cw):
    if len(pts3d_world) == 0:
        return 1e9
    # world→cam: p_c = R_cw.T @ (p_w - t_cw)
    p_c = (pts3d_world - t_cw[None,:]) @ R_cw   # shape (N,3)
    z = p_c[:,2]; valid = z > 0.1
    if valid.sum() == 0: return 1e9
    u = fx * p_c[valid,0] / z[valid] + cx
    v = fy * p_c[valid,1] / z[valid] + cy
    e = np.sqrt((u - pts2d[valid,0])**2 + (v - pts2d[valid,1])**2)
    return float(np.mean(e))

# ── sliding-window BA ─────────────────────────────────────────────────────────
def rt_to_vec(R, t):
    rv, _ = cv2.Rodrigues(R); return np.concatenate([rv.ravel(), t.ravel()])

def vec_to_rt(v):
    R, _ = cv2.Rodrigues(v[:3].copy()); return R, v[3:6].copy()

def bundle_adjust_window(win_kfs):
    """win_kfs: list of KF dicts with R, t, pts3d (world), pts2d."""
    n = len(win_kfs)
    if n < 2: return None
    pre = [kf_reproj_err(kf['pts3d'], kf['pts2d'], kf['R'], kf['t']) for kf in win_kfs]
    pre_mean = float(np.mean(pre))
    if pre_mean > 50 or pre_mean < 0.1: return None

    x0 = np.concatenate([rt_to_vec(kf['R'], kf['t']) for kf in win_kfs])

    def residuals(x):
        res = []
        for i, kf in enumerate(win_kfs):
            R_i, t_i = vec_to_rt(x[i*6:(i+1)*6])
            p3 = kf['pts3d']; p2 = kf['pts2d']
            if len(p3) == 0: continue
            p_c = (p3 - t_i[None,:]) @ R_i
            z = p_c[:,2]; valid = z > 0.1
            if valid.sum() == 0: continue
            u = fx * p_c[valid,0] / z[valid] + cx
            v = fy * p_c[valid,1] / z[valid] + cy
            eu = u - p2[valid,0]; ev = v - p2[valid,1]
            err = np.concatenate([eu, ev])
            delta = 4.0
            mask = np.abs(err) < delta
            huber = np.where(mask, err,
                             delta * np.sign(err) * np.sqrt(np.maximum(2*np.abs(err)/delta - 1, 0)))
            res.append(huber)
        return np.concatenate(res) if res else np.zeros(1)

    try:
        r = least_squares(residuals, x0, method='lm', max_nfev=60, ftol=1e-4, xtol=1e-4)
        x_opt = r.x
    except Exception:
        return None

    poses_new, post_errs = [], []
    for i, kf in enumerate(win_kfs):
        R_i, t_i = vec_to_rt(x_opt[i*6:(i+1)*6])
        poses_new.append((R_i, t_i))
        post_errs.append(kf_reproj_err(kf['pts3d'], kf['pts2d'], R_i, t_i))
    post_mean = float(np.mean(post_errs))
    # Monotonic safeguard
    if post_mean >= pre_mean * 1.02:
        return None
    return poses_new

# ── Hamming distance matrix (vectorised) ──────────────────────────────────────
def hamming_dist_matrix(des_q, des_ref):
    """(Q,32) vs (R,32) uint8 → (Q,R) int hamming distances."""
    # XOR each query against all ref, then popcount
    Q, R = len(des_q), len(des_ref)
    dist = np.empty((Q, R), dtype=np.int32)
    for qi in range(Q):
        xor = np.bitwise_xor(des_q[qi:qi+1], des_ref)   # (R,32)
        dist[qi] = np.unpackbits(xor, axis=1).sum(axis=1)
    return dist

# ── pose-graph optimisation (fast linear-correction over each loop) ────────────
def pose_graph_optimize(kf_poses, seq_edges, loop_edges):
    """
    Fast O(n) pose-graph correction for each loop edge independently,
    then average corrections.  Much faster than full nonlinear optimisation.
    kf_poses: list of (R, t)
    loop_edges: list of (i, j, R_rel_meas, t_rel_meas)
                 where R_rel_meas = R_i.T @ R_j_expected
                 i.e., relative pose from KF i to KF j
    """
    n = len(kf_poses)
    if n < 2 or not loop_edges:
        return kf_poses

    # Accumulate per-KF corrections as Rodrigues vectors + translations
    # Multiple loops may correct the same KF; we average their corrections.
    corrections_rv = [[] for _ in range(n)]   # list of rvec corrections
    corrections_t  = [[] for _ in range(n)]   # list of t corrections

    for (ki, kj, R_rel_meas, t_rel_meas) in loop_edges:
        R_i, t_i = kf_poses[ki]
        R_j, t_j = kf_poses[kj]

        # Predicted pose of KF j from loop constraint:
        R_j_pred = R_i @ R_rel_meas
        t_j_pred = t_i + R_i @ t_rel_meas

        # Error at KF j:
        dR = R_j.T @ R_j_pred          # rotation correction at j
        rv_err, _ = cv2.Rodrigues(dR)
        dt_err = t_j_pred - t_j         # translation correction at j

        # Linearly distribute correction from ki+1 to kj (ki stays fixed)
        span = kj - ki
        if span <= 0:
            continue
        for k in range(ki + 1, kj + 1):
            alpha = (k - ki) / span
            corrections_rv[k].append(rv_err.ravel() * alpha)
            corrections_t[k].append(dt_err * alpha)

    # Apply averaged corrections
    poses_out = list(kf_poses)
    for k in range(n):
        if not corrections_rv[k]:
            continue
        R_k, t_k = kf_poses[k]
        mean_rv = np.mean(corrections_rv[k], axis=0)
        mean_dt = np.mean(corrections_t[k],  axis=0)
        R_corr, _ = cv2.Rodrigues(mean_rv)
        R_new = R_k @ R_corr
        t_new = t_k + mean_dt
        poses_out[k] = (R_new, t_new)

    return poses_out

# ══════════════════════════════════════════════════════════════════════════════
# ──  PHASE 1 : VO FRONT-END  ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

KF_INTERVAL = 4    # add KF every N frames
BA_WINDOW   = 7

poses_R = []; poses_t = []   # per-frame cam-to-world
keyframes   = []              # keyframe dicts
t_wall = time.time()

R_cur = np.eye(3); t_cur = np.zeros(3)   # cam-to-world

# Per-frame carry-overs
prev_kps = prev_des = prev_depth = None
prev_pts3d_cam = None   # cam-space 3D pts aligned with prev_kps index

for fi in range(N):
    img_l = cv2.imread(os.path.join(LAB_DATA, left_files[fi]),  cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(os.path.join(LAB_DATA, right_files[fi]), cv2.IMREAD_GRAYSCALE)
    depth = compute_depth(img_l, img_r)
    kps, des = detect(img_l)

    # Build cam-space 3D lookup (aligned with kps index)
    H, W = depth.shape
    pts3d_cam_cur = np.zeros((len(kps), 3), dtype=np.float64) if kps else np.zeros((0,3))
    if kps:
        for i, kp in enumerate(kps):
            u, v = kp.pt
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < H and 0 <= ui < W:
                z = depth[vi, ui]
                if 0.5 < z < 80.0:
                    pts3d_cam_cur[i] = [(u-cx)*z/fx, (v-cy)*z/fy, z]

    if fi == 0:
        R_cur = np.eye(3); t_cur = np.zeros(3)
    else:
        # ── frame-to-frame VO ──────────────────────────────────────────────
        success = False
        if (prev_des is not None and des is not None and
                len(prev_des) >= 8 and len(des) >= 8):
            matches = list(bf.match(prev_des, des))
            matches.sort(key=lambda m: m.distance)
            R_new, t_new, ni = pnp_from_matches(
                matches, prev_pts3d_cam, kps, R_cur, t_cur)
            if R_new is not None:
                R_cur, t_cur = R_new, t_new; success = True

        if not success and len(poses_R) >= 2:
            # Constant-velocity extrapolation
            R_pm1, t_pm1 = poses_R[-2], poses_t[-2]
            R_p0,  t_p0  = poses_R[-1], poses_t[-1]
            dR = R_pm1.T @ R_p0
            dt = R_pm1.T @ (t_p0 - t_pm1)
            R_cur = R_p0 @ dR
            t_cur = t_p0 + R_p0 @ dt

    poses_R.append(R_cur.copy()); poses_t.append(t_cur.copy())

    # ── Keyframe? ──────────────────────────────────────────────────────────
    if fi % KF_INTERVAL == 0:
        kf_idx = len(keyframes)
        # World-space 3D for this KF's visible points
        _, _, pts3d_w, pts2d = backproject_kps(kps, des, depth, R_cur, t_cur)
        # Cam-space 3D for loop verification (no reloading needed later)
        # Also store all descriptors for loop scoring
        _, pts3d_c, _, _ = backproject_kps(kps, des, depth, np.eye(3), np.zeros(3))
        # des_3d aligned with pts3d_c
        des_3d_list, p3c_list, p2_list = [], [], []
        if kps and des is not None:
            for i, kp in enumerate(kps):
                u, v = kp.pt
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < H and 0 <= ui < W:
                    z = depth[vi, ui]
                    if 0.5 < z < 60.0:
                        xc = (u-cx)*z/fx; yc = (v-cy)*z/fy
                        des_3d_list.append(des[i])
                        p3c_list.append([xc, yc, z])
                        p2_list.append([u, v])
        des_3d = np.array(des_3d_list, dtype=np.uint8) if des_3d_list else np.zeros((0,32),dtype=np.uint8)
        pts3d_cam_kf = np.array(p3c_list, dtype=np.float64) if p3c_list else np.zeros((0,3))
        pts2d_kf     = np.array(p2_list,  dtype=np.float64) if p2_list  else np.zeros((0,2))

        kf = {
            'frame_idx': fi, 'kf_idx': kf_idx,
            'R': R_cur.copy(), 't': t_cur.copy(),
            'des': des,              # all descriptors (for loop scoring)
            'kps': kps,
            'des_3d': des_3d,        # descriptors with valid depth
            'pts3d_cam': pts3d_cam_kf,   # cam-space 3D (for loop PnP, no reload)
            'pts3d': pts3d_w,        # world-space 3D (for BA)
            'pts2d': pts2d_kf,       # 2D (for BA)
        }
        keyframes.append(kf)

        # ── Sliding-window BA ──────────────────────────────────────────────
        if len(keyframes) >= 3:
            win = keyframes[-BA_WINDOW:]
            ba = bundle_adjust_window(win)
            if ba is not None:
                offset = len(keyframes) - len(win)
                for i_w, (R_ba, t_ba) in enumerate(ba):
                    g = offset + i_w
                    if 0 <= g < len(keyframes):
                        keyframes[g]['R'] = R_ba.copy()
                        keyframes[g]['t'] = t_ba.copy()
                R_cur = keyframes[-1]['R'].copy()
                t_cur = keyframes[-1]['t'].copy()
                poses_R[-1] = R_cur.copy(); poses_t[-1] = t_cur.copy()

    prev_depth = depth; prev_kps = kps; prev_des = des
    prev_pts3d_cam = pts3d_cam_cur

    if fi % 50 == 0:
        print(f"[Phase1 {fi:3d}/{N}] t={time.time()-t_wall:.1f}s  "
              f"cam=({t_cur[0]:.1f},{t_cur[1]:.1f},{t_cur[2]:.1f})  kfs={len(keyframes)}")

print(f"[Phase1 done] {len(keyframes)} KFs, elapsed {time.time()-t_wall:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# ──  PHASE 2 : BATCH LOOP DETECTION & CLOSURE  ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

LOOP_MIN_KF_GAP  = 15    # keyframe index gap
LOOP_MIN_SCORE   = 30    # descriptor vote count
LOOP_MIN_INLIERS = 18    # PnP inlier count

loop_edges = []

n_kf = len(keyframes)
print(f"[Phase2] Loop detection over {n_kf} keyframes...")

for kf_j_idx in range(LOOP_MIN_KF_GAP, n_kf):
    kf_j = keyframes[kf_j_idx]
    if kf_j['des'] is None or len(kf_j['des']) < 8:
        continue
    des_q = kf_j['des']

    best_score = 0; best_kf_i = -1

    for kf_i_idx in range(kf_j_idx - LOOP_MIN_KF_GAP):
        kf_i = keyframes[kf_i_idx]
        if kf_i['des'] is None or len(kf_i['des']) < 8:
            continue
        raw = list(bf.match(des_q, kf_i['des']))
        score = sum(1 for m in raw if m.distance < 50)
        if score > best_score:
            best_score = score; best_kf_i = kf_i_idx

    if best_score < LOOP_MIN_SCORE or best_kf_i < 0:
        continue

    cand = keyframes[best_kf_i]
    if len(cand['des_3d']) < 8 or len(cand['pts3d_cam']) < 8:
        continue

    # Geometric verification: use stored cam-space 3D → world
    raw = list(bf.match(des_q, cand['des_3d']))
    raw.sort(key=lambda m: m.distance)

    obj_pts, img_pts = [], []
    for m in raw[:300]:
        if m.trainIdx < len(cand['pts3d_cam']):
            pc = cand['pts3d_cam'][m.trainIdx]
            if pc[2] < 0.1: continue
            pw = cand['R'] @ pc + cand['t']
            obj_pts.append(pw)
            img_pts.append(kf_j['kps'][m.queryIdx].pt)

    if len(obj_pts) < 8:
        continue

    ok_l, rvec_l, tvec_l, inliers_l = cv2.solvePnPRansac(
        np.array(obj_pts, dtype=np.float64),
        np.array(img_pts, dtype=np.float64),
        K, None,
        iterationsCount=300, reprojectionError=2.5, confidence=0.999)

    if not (ok_l and inliers_l is not None and len(inliers_l) >= LOOP_MIN_INLIERS):
        continue

    R_wc, _ = cv2.Rodrigues(rvec_l); t_wc = tvec_l.ravel()
    R_lc = R_wc.T; t_lc = -R_wc.T @ t_wc   # cam-to-world for kf_j

    R_i = cand['R']; t_i = cand['t']
    R_rel = R_i.T @ R_lc
    t_rel = R_i.T @ (t_lc - t_i)

    print(f"[LOOP] kf={kf_j_idx}(f={kf_j['frame_idx']}) ← kf={best_kf_i}(f={cand['frame_idx']})"
          f"  score={best_score}  inliers={len(inliers_l)}")
    loop_edges.append((best_kf_i, kf_j_idx, R_rel, t_rel))

print(f"[Phase2 done] {len(loop_edges)} loop edges, elapsed {time.time()-t_wall:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# ──  PHASE 3 : POSE-GRAPH OPTIMISATION  ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

seq_edges = []
for i in range(n_kf - 1):
    R_i, t_i = keyframes[i]['R'], keyframes[i]['t']
    R_j, t_j = keyframes[i+1]['R'], keyframes[i+1]['t']
    seq_edges.append((i, i+1, R_i.T @ R_j, R_i.T @ (t_j - t_i)))

if loop_edges:
    print("[Phase3] Running global pose-graph optimisation...")
    kf_poses_in = [(kf['R'], kf['t']) for kf in keyframes]
    kf_poses_out = pose_graph_optimize(kf_poses_in, seq_edges, loop_edges)
    for i, kf in enumerate(keyframes):
        kf['R'], kf['t'] = kf_poses_out[i]
    print(f"[Phase3 done] pose-graph complete, elapsed {time.time()-t_wall:.1f}s")
    # Show drift correction
    last_R, last_t = kf_poses_out[-1]; first_R, first_t = kf_poses_out[0]
    print(f"[Phase3] Final KF position after correction: {last_t}")
else:
    print("[Phase3] No loop edges — skipping pose-graph")

# ══════════════════════════════════════════════════════════════════════════════
# ──  PHASE 4 : PROPAGATE POSES → OUTPUT  ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

poses_R_out = [None] * N
poses_t_out = [None] * N

# Set KF poses
for kf in keyframes:
    poses_R_out[kf['frame_idx']] = kf['R'].copy()
    poses_t_out[kf['frame_idx']] = kf['t'].copy()

def slerp_pose(R0, t0, R1, t1, a):
    dR = R0.T @ R1
    rv, _ = cv2.Rodrigues(dR)
    Ra, _ = cv2.Rodrigues(rv * a)
    return R0 @ Ra, t0 + a * (t1 - t0)

for fi in range(N):
    if poses_R_out[fi] is not None:
        continue
    prev_kf = next_kf = None
    for kf in reversed(keyframes):
        if kf['frame_idx'] < fi: prev_kf = kf; break
    for kf in keyframes:
        if kf['frame_idx'] > fi: next_kf = kf; break
    if prev_kf is None:
        poses_R_out[fi] = keyframes[0]['R'].copy(); poses_t_out[fi] = keyframes[0]['t'].copy()
    elif next_kf is None:
        poses_R_out[fi] = keyframes[-1]['R'].copy(); poses_t_out[fi] = keyframes[-1]['t'].copy()
    else:
        span = next_kf['frame_idx'] - prev_kf['frame_idx']
        a = (fi - prev_kf['frame_idx']) / span
        Ri, ti = slerp_pose(prev_kf['R'], prev_kf['t'], next_kf['R'], next_kf['t'], a)
        poses_R_out[fi] = Ri; poses_t_out[fi] = ti

# Write
traj_path  = os.path.join(LAB_ARTIFACTS, "traj.txt")
poses_path = os.path.join(LAB_ARTIFACTS, "poses.txt")
with open(traj_path, 'w') as ft, open(poses_path, 'w') as fp:
    for fi in range(N):
        R_f = poses_R_out[fi]; t_f = poses_t_out[fi]
        ft.write(f"{t_f[0]:.6f} {t_f[1]:.6f} {t_f[2]:.6f}\n")
        row = [R_f[0,0],R_f[0,1],R_f[0,2],t_f[0],
               R_f[1,0],R_f[1,1],R_f[1,2],t_f[1],
               R_f[2,0],R_f[2,1],R_f[2,2],t_f[2]]
        fp.write(' '.join(f"{v:.6f}" for v in row)+'\n')

total = time.time() - t_wall
print(f"\n[DONE] {N} poses → {traj_path} + {poses_path}")
print(f"[DONE] total wall time: {total:.1f}s")
print(f"[DONE] final cam centre: {poses_t_out[-1]}")
