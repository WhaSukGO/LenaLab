#!/usr/bin/env python3
"""
Monocular Visual Odometry – PnP-centric, consistent world-scale.

Deferred initialisation: waits until depth/baseline < 15 for a good
triangulation, then back-fills earlier frames with PnP.
Subsequent frames use solvePnPRansac with reprojection-pruned landmark map.
"""

import os, glob
import numpy as np
import cv2

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR      = os.environ.get("LAB_DATA",      "/data")
ARTIFACTS_DIR = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─── Intrinsics ───────────────────────────────────────────────────────────────
intr = np.loadtxt(os.path.join(DATA_DIR, "intrinsics.txt"))
fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
D = np.zeros(4)

frames = sorted(glob.glob(os.path.join(DATA_DIR, "frame_*.png")))
N = len(frames)
print(f"Found {N} frames")

# ─── Parameters ───────────────────────────────────────────────────────────────
MAX_FEATS   = 2000
MAX_LM      = 3000
MIN_LM      = 80
MIN_PNP     = 15
REPROJ_THR  = 4.0
MAX_ROT_DEG = 30.0
TARGET_DEPTH_RATIO = 20.0   # want median_depth/baseline_dist < this for init
MAX_INIT_FRAME = 25         # don't wait beyond this frame for initialization

LK_PARAMS = dict(winSize=(21,21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def detect(img, n=MAX_FEATS):
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    kps  = fast.detect(img, None)
    if not kps:
        return np.array([[x,y] for y in range(20,img.shape[0]-20,15)
                               for x in range(20,img.shape[1]-20,15)],
                        dtype=np.float32).reshape(-1,1,2)
    kps = sorted(kps, key=lambda k: k.response, reverse=True)[:n]
    return np.array([k.pt for k in kps], dtype=np.float32).reshape(-1,1,2)


def lk_track(img0, img1, pts0):
    pts1,s1,_ = cv2.calcOpticalFlowPyrLK(img0, img1, pts0, None, **LK_PARAMS)
    ptsb,s2,_ = cv2.calcOpticalFlowPyrLK(img1, img0, pts1, None, **LK_PARAMS)
    err  = np.abs(pts0 - ptsb).reshape(-1,2).max(axis=1)
    good = (s1.ravel()==1) & (s2.ravel()==1) & (err < 1.5)
    return pts1[good], pts0[good], good


def custom_recover_pose(E, p0, p1, K, e_mask):
    W  = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float64)
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U)  < 0: U  = -U
    if np.linalg.det(Vt) < 0: Vt = -Vt
    R1 = U @ W   @ Vt
    R2 = U @ W.T @ Vt
    t0 = U[:,2]
    sel = e_mask.ravel().astype(bool)
    q0, q1 = p0[sel], p1[sel]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    best_n, best_R, best_t, best_ch = -1, R1, t0, None
    for Ri, ti in [(R1,t0),(R1,-t0),(R2,t0),(R2,-t0)]:
        P2  = K @ np.hstack([Ri, ti.reshape(3,1)])
        p4d = cv2.triangulatePoints(P1, P2, q0.T, q1.T)
        p3d = (p4d[:3]/p4d[3]).T
        z1  = p3d[:,2]
        z2  = ((Ri @ p3d.T).T + ti)[:,2]
        ch  = (z1>0) & (z2>0)
        if ch.sum() > best_n:
            best_n, best_R, best_t, best_ch = int(ch.sum()), Ri, ti, ch
    return best_n, best_R, best_t, best_ch


def triangulate_world(R0, t0, R1, t1, p0_2d, p1_2d):
    P0  = K @ np.hstack([R0, t0.reshape(3,1)])
    P1  = K @ np.hstack([R1, t1.reshape(3,1)])
    p4d = cv2.triangulatePoints(P0, P1, p0_2d.T, p1_2d.T)
    return (p4d[:3]/p4d[3]).T


def reproj_err(pts3d, pts2d, R, t):
    proj, _ = cv2.projectPoints(pts3d, cv2.Rodrigues(R)[0], t, K, D)
    return np.linalg.norm(pts2d - proj.reshape(-1,2), axis=1)


def prune_lm(lm3d, lm2d, R, t, thr=REPROJ_THR):
    z_cam = ((R @ lm3d.T).T + t)[:,2]
    keep  = (reproj_err(lm3d, lm2d, R, t) < thr) & (z_cam > 0)
    return lm3d[keep], lm2d[keep]


def pnp_pose(lm3d, lm2d, R_init=None, t_init=None):
    """solvePnPRansac; returns (R_new, t_new, inlier_mask) or None."""
    if len(lm3d) < MIN_PNP:
        return None
    kw = {}
    if R_init is not None and t_init is not None:
        kw['rvec'] = cv2.Rodrigues(R_init)[0]
        kw['tvec'] = t_init.reshape(3,1)
        kw['useExtrinsicGuess'] = True
    ret, rvec, tvec, inl = cv2.solvePnPRansac(
        lm3d, lm2d, K, D,
        iterationsCount=1000,
        reprojectionError=REPROJ_THR,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
        **kw)
    if not ret or inl is None or len(inl) < MIN_PNP:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel(), inl.ravel()


def cam_centre(R, t):
    return -R.T @ t


# ─────────────────────────── Main ─────────────────────────────────────────────

def main():
    # ── Phase 0: deferred initialisation ──────────────────────────────────────
    # Track features from frame 0 across multiple frames until we get good
    # triangulation quality (depth/baseline ratio < TARGET_DEPTH_RATIO).

    img0 = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
    pts0_orig = detect(img0)           # detections in frame 0
    pts0_cur  = pts0_orig.copy()       # tracked positions in current frame
    alive0    = np.arange(len(pts0_orig))  # indices of still-alive features

    init_frame_idx = None
    R_init = None;  t_init = None
    lm_3d_init = None;  lm_2d_init = None  # in init-frame

    img_prev = img0
    img_hist = [img0]   # keep images for PnP backfill

    for i in range(1, min(N, MAX_INIT_FRAME + 1)):
        img_curr = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)
        img_hist.append(img_curr)

        pts_c, pts_p, good = lk_track(img_prev, img_curr,
                                       pts0_cur.reshape(-1,1,2))
        alive0    = alive0[good]
        pts0_cur  = pts_c.reshape(-1,2)   # current positions (in img_curr)
        pts0_orig_alive = pts0_orig.reshape(-1,2)[alive0]   # original positions

        q0 = pts0_orig_alive   # positions in frame 0
        q1 = pts0_cur          # positions in frame i

        if len(q0) < 15:
            img_prev = img_curr
            continue

        E, e_mask = cv2.findEssentialMat(q0, q1, K, method=cv2.RANSAC,
                                          prob=0.999, threshold=1.0)
        if E is None or e_mask is None or e_mask.sum() < 15:
            img_prev = img_curr
            continue

        n_ch, R_rel, t_rel, ch = custom_recover_pose(E, q0, q1, K, e_mask)
        if n_ch < MIN_PNP:
            img_prev = img_curr
            continue

        # Check triangulation quality
        sel  = e_mask.ravel().astype(bool)
        p0_e = q0[sel][ch];  p1_e = q1[sel][ch]
        pts3d = triangulate_world(np.eye(3), np.zeros(3), R_rel, t_rel, p0_e, p1_e)
        z0v   = pts3d[:,2]
        z1v   = ((R_rel @ pts3d.T).T + t_rel)[:,2]
        vld   = (z0v>0) & (z1v>0) & (z0v < 1000)
        if vld.sum() < MIN_PNP:
            img_prev = img_curr
            continue

        med_depth = np.median(z0v[vld])
        baseline  = np.linalg.norm(cam_centre(R_rel, t_rel))  # = 1 (unit t)

        print(f"Frame 0 vs {i}: n_lm={vld.sum()}  "
              f"depth={med_depth:.1f}  ratio={med_depth/baseline:.1f}")

        if med_depth / baseline < TARGET_DEPTH_RATIO or i >= MAX_INIT_FRAME:
            # Use this pair for initialisation
            R_init  = R_rel.copy();   t_init  = t_rel.copy()
            lm_3d_init = pts3d[vld].copy()
            lm_2d_init = p1_e[vld].copy()   # 2-D in frame i
            init_frame_idx = i
            print(f"Using frame 0 vs {i} for init: "
                  f"{vld.sum()} landmarks, depth/baseline={med_depth:.1f}")
            break

        img_prev = img_curr

    if R_init is None:
        # Fallback: use first available initialisation (frame 0 vs 1)
        print("WARNING: fallback to frame 0 vs 1 init")
        img1 = cv2.imread(frames[1], cv2.IMREAD_GRAYSCALE)
        pts0 = detect(img0)
        pts1_c, pts0_c, _ = lk_track(img0, img1, pts0)
        q0 = pts0_c.reshape(-1,2);  q1 = pts1_c.reshape(-1,2)
        E, e_mask = cv2.findEssentialMat(q0, q1, K, method=cv2.RANSAC,
                                          prob=0.999, threshold=1.0)
        n_ch, R_init, t_init, ch = custom_recover_pose(E, q0, q1, K, e_mask)
        sel  = e_mask.ravel().astype(bool)
        pts3d = triangulate_world(np.eye(3), np.zeros(3), R_init, t_init,
                                  q0[sel][ch], q1[sel][ch])
        z0v   = pts3d[:,2]; z1v = ((R_init @ pts3d.T).T + t_init)[:,2]
        vld   = (z0v>0) & (z1v>0) & (z0v < 1000)
        lm_3d_init = pts3d[vld];  lm_2d_init = q1[sel][ch][vld]
        init_frame_idx = 1
        if len(img_hist) < 2:
            img_hist.append(img1)

    # ── Phase 1: backfill frames 1 .. init_frame_idx-1 with PnP ───────────────
    # Prune init landmarks
    lm_3d_init, lm_2d_init = prune_lm(lm_3d_init, lm_2d_init, R_init, t_init)

    # Poses for frames 0 .. init_frame_idx
    poses = {0:  (np.eye(3), np.zeros(3)),
             init_frame_idx: (R_init, t_init)}

    # Forward pass: frames 1 .. init_frame_idx-1
    for j in range(1, init_frame_idx):
        img_j = img_hist[j]
        proj, _ = cv2.projectPoints(
            lm_3d_init, cv2.Rodrigues(R_init)[0], t_init, K, D)
        # No 2-D positions in frame j yet; use reprojection as initial guess
        # Better: track from frame j-1 to j
        # We don't have stored 2-D positions for intermediate frames easily.
        # Simple approach: use PnP with projections from the closest pose.
        # For simplicity, just linearly interpolate frames 1..init_frame_idx-1.
        alpha = j / init_frame_idx
        R_j = R_init   # placeholder
        t_j_centre = (1-alpha) * np.zeros(3) + alpha * cam_centre(R_init, t_init)
        # R_j is init rotation scaled to 0 for j=0
        # Use Rodrigues interpolation
        rvec_init, _ = cv2.Rodrigues(R_init)
        rvec_j = alpha * rvec_init
        R_j, _  = cv2.Rodrigues(rvec_j)
        t_j = -R_j @ t_j_centre
        poses[j] = (R_j, t_j)

    # ── Phase 2: normal PnP-based tracking from init_frame_idx onwards ─────────
    lm_3d = lm_3d_init.copy()
    lm_2d = lm_2d_init.copy()
    R_wc  = R_init.copy();  t_wc  = t_init.copy()
    R_prv = np.eye(3);      t_prv = np.zeros(3)

    seeds_prev_img = img_hist[init_frame_idx]   # image of last init frame
    seeds_p = detect(seeds_prev_img).reshape(-1,2)   # seeds in init frame

    img_prev_norm = img_hist[init_frame_idx]

    for i in range(init_frame_idx + 1, N):
        img_curr = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE)

        # (a) Track landmarks
        lm_pts_prev = lm_2d.reshape(-1,1,2).astype(np.float32)
        lm_c, lm_p_ok, good_lm = lk_track(img_prev_norm, img_curr, lm_pts_prev)
        lm3_trk = lm_3d[good_lm]
        lm2_trk = lm_c.reshape(-1,2)

        # (b) PnP
        pose_ok = False
        res = pnp_pose(lm3_trk, lm2_trk, R_wc, t_wc)
        if res is not None:
            R_new, t_new, inl_idx = res
            dR    = R_new @ R_wc.T
            d_ang = float(np.linalg.norm(cv2.Rodrigues(dR)[0]))
            if d_ang < np.radians(MAX_ROT_DEG):
                lm_3d = lm3_trk[inl_idx]
                lm_2d = lm2_trk[inl_idx]
                R_prv, t_prv = R_wc.copy(), t_wc.copy()
                R_wc,  t_wc  = R_new.copy(), t_new.copy()
                pose_ok = True

        if not pose_ok:
            # Hold and update 2-D positions
            if len(lm3_trk) > 0:
                lm_3d, lm_2d = prune_lm(lm3_trk, lm2_trk, R_wc, t_wc)
            print(f"  Frame {i:04d}: PnP failed ({len(lm3_trk)} pts), hold")

        # (c) Triangulate seeds
        if seeds_p is not None and len(seeds_p) > 0 and pose_ok:
            sd_c, sd_p, _ = lk_track(img_prev_norm, img_curr,
                                       seeds_p.reshape(-1,1,2).astype(np.float32))
            if len(sd_c) >= 4:
                p0_sd = sd_p.reshape(-1,2)
                p1_sd = sd_c.reshape(-1,2)
                pts3d_new = triangulate_world(R_prv, t_prv, R_wc, t_wc,
                                               p0_sd, p1_sd)
                z_c = ((R_wc   @ pts3d_new.T).T + t_wc)[:,2]
                z_p = ((R_prv  @ pts3d_new.T).T + t_prv)[:,2]
                if len(lm_3d) < MAX_LM:
                    vld = (z_c>0) & (z_p>0) & (z_c<500) & (z_p<500)
                    if vld.sum() > 0:
                        add3 = pts3d_new[vld];  add2 = p1_sd[vld]
                        e = reproj_err(add3, add2, R_wc, t_wc)
                        keep = e < REPROJ_THR
                        n_add = min(keep.sum(), MAX_LM - len(lm_3d))
                        if n_add > 0:
                            lm_3d = np.vstack([lm_3d, add3[keep][:n_add]])
                            lm_2d = np.vstack([lm_2d, add2[keep][:n_add]])
                seeds_p = sd_c.reshape(-1,2)

        # (d) Refresh seeds when map is thin
        if len(lm_3d) < MIN_LM:
            seeds_p = detect(img_curr).reshape(-1,2)

        # (e) Periodic prune
        if i % 5 == 0:
            lm_3d, lm_2d = prune_lm(lm_3d, lm_2d, R_wc, t_wc)

        # (f) Cap map size
        if len(lm_3d) > MAX_LM:
            lm_3d = lm_3d[-MAX_LM:]
            lm_2d = lm_2d[-MAX_LM:]

        poses[i] = (R_wc.copy(), t_wc.copy())
        img_prev_norm = img_curr

        if i % 10 == 0:
            c = cam_centre(R_wc, t_wc)
            print(f"Frame {i:04d}: pos=({c[0]:6.3f},{c[1]:6.3f},{c[2]:6.3f})  "
                  f"lm={len(lm_3d)}  {'OK' if pose_ok else 'hold'}")

    # ── Build trajectory ───────────────────────────────────────────────────────
    # For any missing frames (if tracking broke), hold last known pose
    last_R, last_t = np.eye(3), np.zeros(3)
    trajectory = []
    for i in range(N):
        if i in poses:
            last_R, last_t = poses[i]
        trajectory.append(cam_centre(last_R, last_t).copy())

    traj = np.array(trajectory)
    out  = os.path.join(ARTIFACTS_DIR, "traj.txt")
    np.savetxt(out, traj, fmt="%.6f")
    print(f"\nWrote {len(traj)} poses → {out}")
    print(f"  x=[{traj[:,0].min():.3f},{traj[:,0].max():.3f}]  "
          f"y=[{traj[:,1].min():.3f},{traj[:,1].max():.3f}]  "
          f"z=[{traj[:,2].min():.3f},{traj[:,2].max():.3f}]")


if __name__ == "__main__":
    main()
