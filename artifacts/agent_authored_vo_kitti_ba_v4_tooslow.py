#!/usr/bin/env python3
"""
Stereo Visual Odometry for KITTI with Windowed Bundle Adjustment.

Architecture:
  1. SGBM disparity → metric 3D (Z = fx * baseline / disparity)
  2. ORB features + BF matching → frame-to-frame PnP (metric pose)
  3. Sliding window BA (N keyframes) with monotonic reprojection safeguard
     - Fix anchor (first KF in window) to prevent gauge freedom
     - Jointly optimise poses 1..N-1 and landmark 3D positions
     - Huber loss, tight tolerance, many iterations
     - Only accept BA result if window reprojection error strictly decreases
     - Also reject if any pose moved more than MAX_BA_POSE_DELTA metres
  4. KEY DESIGN: BA updates the OUTPUT trajectory at KF positions only.
     self.pose_c2w (PnP state) is NEVER modified by BA, preventing
     "teleportation" that corrupts subsequent PnP 3D→world transforms.
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import os

# ─────────────── CONFIG ───────────────
WINDOW_SIZE       = 18      # keyframes in sliding window
MIN_TRACK_LEN     = 2       # min KF observations for a landmark to enter BA
MAX_LM_BA         = 400     # max landmarks per BA call
BA_FTOL           = 1e-7
BA_XTOL           = 1e-7
BA_GTOL           = 1e-9
BA_MAX_NFE        = 6000    # max function evaluations
HUBER_DELTA       = 2.0     # pixels
MAX_BA_POSE_DELTA = 4.0     # metres: reject BA if any KF moved more than this

MIN_DEPTH         = 1.0     # metres
MAX_DEPTH         = 60.0    # tighter → more reliable SGBM depths
MAX_FRAME_DISP    = 8.0     # metres: reject PnP if > this (sanity check)

# Keyframe selection
KF_EVERY_N        = 3       # frames between KF candidates
KF_MIN_DIST       = 0.2     # metres min between KFs

N_ORB             = 3000
# ──────────────────────────────────────

DATA = os.environ.get('LAB_DATA',      '/data')
ART  = os.environ.get('LAB_ARTIFACTS', '/artifacts')


# ════════════════════════════════════════
#  Geometry helpers
# ════════════════════════════════════════

def Tinv(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3,3]  = -(R.T @ t)
    return Ti

def to_mat(rv, tv):
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64))
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3]  = np.asarray(tv, dtype=np.float64).flatten()
    return T

def from_mat(T):
    rv, _ = cv2.Rodrigues(T[:3,:3])
    return rv.flatten(), T[:3,3].copy()


# ════════════════════════════════════════
#  Stereo depth
# ════════════════════════════════════════

def make_sgbm():
    bs = 11
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=bs,
        P1=8*3*bs*bs, P2=32*3*bs*bs,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

def get_disp(sgbm, left, right):
    d = sgbm.compute(left, right).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d

def backproject(pts2d, disp, fx, fy, cx, cy, baseline,
                min_depth=MIN_DEPTH, max_depth=MAX_DEPTH):
    """2-D pixel list → metric 3-D via disparity. Returns (Nx3, valid_idx)."""
    pts3d, valid = [], []
    for i, (u, v) in enumerate(pts2d):
        ui, vi = int(u + .5), int(v + .5)
        if not (0 <= vi < disp.shape[0] and 0 <= ui < disp.shape[1]):
            continue
        d = disp[vi, ui]
        if not (np.isfinite(d) and d > 1.0):
            r = 2
            patch = disp[max(0,vi-r):min(disp.shape[0],vi+r+1),
                         max(0,ui-r):min(disp.shape[1],ui+r+1)]
            good = patch[np.isfinite(patch) & (patch > 1.0)]
            if len(good) == 0:
                continue
            d = np.median(good)
        Z = fx * baseline / d
        if not (min_depth <= Z <= max_depth):
            continue
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pts3d.append([X, Y, Z])
        valid.append(i)
    if pts3d:
        return np.array(pts3d, dtype=np.float64), valid
    return np.empty((0,3), dtype=np.float64), []


# ════════════════════════════════════════
#  Bundle Adjustment
# ════════════════════════════════════════

class WindowedBA:
    """Windowed BA with monotonic reprojection safeguard + pose-delta check."""

    def __init__(self, K):
        self.K  = K
        self.fx = K[0,0]; self.fy = K[1,1]
        self.cx = K[0,2]; self.cy = K[1,2]
        self.n_accepted = 0
        self.n_rejected = 0

    def _project(self, rv, tv, pt3d):
        """Project a single 3D point. Returns (u,v) or None."""
        R, _ = cv2.Rodrigues(rv)
        p = R @ pt3d + tv
        if p[2] < 1e-4:
            return None
        u = self.fx * p[0] / p[2] + self.cx
        v = self.fy * p[1] / p[2] + self.cy
        return np.array([u, v])

    def residuals(self, x, n_free, n_lm, obs_list, rv0, tv0):
        lm_off = n_free * 6
        lms = x[lm_off:].reshape(n_lm, 3)
        r = np.empty(len(obs_list) * 2, dtype=np.float64)
        for k, (i, j, obs) in enumerate(obs_list):
            if i == 0:
                rv, tv = rv0, tv0
            else:
                b = (i-1)*6
                rv, tv = x[b:b+3], x[b+3:b+6]
            proj = self._project(rv, tv, lms[j])
            if proj is None:
                r[2*k] = r[2*k+1] = 50.0
            else:
                r[2*k]   = proj[0] - obs[0]
                r[2*k+1] = proj[1] - obs[1]
        return r

    def mse(self, x, n_free, n_lm, obs_list, rv0, tv0):
        r = self.residuals(x, n_free, n_lm, obs_list, rv0, tv0)
        return float(np.mean(r * r))

    def run(self, kfs, lms_dict):
        """
        Returns (accepted: bool, updated_poses: list[4×4], updated_lm: dict).
        updated_poses[i] is the refined cam-to-world pose of kfs[i].
        """
        n = len(kfs)
        if n < 4:
            return False, None, None

        kfid2i = {kf['id']: i for i, kf in enumerate(kfs)}

        # ── collect active landmarks ──────────────────────────────
        good = []
        for lm_id, lm in lms_dict.items():
            obs_in = [(kfid2i[k], np.asarray(v))
                      for k, v in lm['obs'].items() if k in kfid2i]
            if len(obs_in) >= MIN_TRACK_LEN:
                good.append((lm_id, obs_in))
        if len(good) < 20:
            return False, None, None

        good.sort(key=lambda x: -len(x[1]))
        if len(good) > MAX_LM_BA:
            good = good[:MAX_LM_BA]
        n_lm = len(good)

        obs_list = []
        for j, (lm_id, obs_in) in enumerate(good):
            for i, pt2d in obs_in:
                obs_list.append((i, j, pt2d))
        if len(obs_list) < 50:
            return False, None, None

        # ── build x0 ─────────────────────────────────────────────
        n_free   = n - 1
        lm_off   = n_free * 6
        n_params = lm_off + n_lm * 3
        x0 = np.zeros(n_params)

        # anchor = kfs[0], fixed
        T_anc = Tinv(kfs[0]['pose'])
        rv0, tv0 = from_mat(T_anc)

        for i in range(1, n):
            T_w2c = Tinv(kfs[i]['pose'])
            rv, tv = from_mat(T_w2c)
            b = (i-1)*6
            x0[b:b+3]   = rv
            x0[b+3:b+6] = tv

        for j, (lm_id, _) in enumerate(good):
            x0[lm_off+j*3 : lm_off+j*3+3] = lms_dict[lm_id]['pt']

        err_before = self.mse(x0, n_free, n_lm, obs_list, rv0, tv0)

        # ── sparsity ──────────────────────────────────────────────
        n_res = len(obs_list) * 2
        S = lil_matrix((n_res, n_params), dtype=np.int8)
        for k, (i, j, _) in enumerate(obs_list):
            if i > 0:
                b = (i-1)*6
                S[2*k,   b:b+6] = 1
                S[2*k+1, b:b+6] = 1
            li = lm_off + j*3
            S[2*k,   li:li+3] = 1
            S[2*k+1, li:li+3] = 1

        # ── optimise ─────────────────────────────────────────────
        try:
            res = least_squares(
                self.residuals, x0,
                args=(n_free, n_lm, obs_list, rv0, tv0),
                method='trf', jac_sparsity=S,
                ftol=BA_FTOL, xtol=BA_XTOL, gtol=BA_GTOL,
                max_nfev=BA_MAX_NFE,
                loss='huber', f_scale=HUBER_DELTA,
                verbose=0
            )
            x_opt = res.x
        except Exception:
            self.n_rejected += 1
            return False, None, None

        err_after = self.mse(x_opt, n_free, n_lm, obs_list, rv0, tv0)

        # ── guard 1: reprojection must improve ────────────────────
        if err_after >= err_before:
            self.n_rejected += 1
            return False, None, None

        # ── decode poses and guard 2: pose delta bound ────────────
        updated_poses = [kfs[0]['pose'].copy()]
        for i in range(1, n):
            b = (i-1)*6
            rv = x_opt[b:b+3]; tv = x_opt[b+3:b+6]
            new_pose = Tinv(to_mat(rv, tv))
            delta = np.linalg.norm(new_pose[:3,3] - kfs[i]['pose'][:3,3])
            if delta > MAX_BA_POSE_DELTA:
                self.n_rejected += 1
                return False, None, None
            updated_poses.append(new_pose)

        # ── decode landmarks ──────────────────────────────────────
        updated_lm = {}
        for j, (lm_id, _) in enumerate(good):
            updated_lm[lm_id] = x_opt[lm_off+j*3 : lm_off+j*3+3].copy()

        self.n_accepted += 1
        return True, updated_poses, updated_lm


# ════════════════════════════════════════
#  Main VO class
# ════════════════════════════════════════

class StereoVO:
    def __init__(self):
        vals = [float(l.strip()) for l in open(f'{DATA}/intrinsics.txt')]
        fx, fy, cx, cy, baseline = vals
        self.K        = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
        self.fx       = fx; self.fy = fy
        self.cx       = cx; self.cy = cy
        self.baseline = baseline

        self.sgbm = make_sgbm()
        self.orb  = cv2.ORB_create(nfeatures=N_ORB, scaleFactor=1.2, nlevels=8,
                                    edgeThreshold=15, patchSize=31)
        self.bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ba   = WindowedBA(self.K)

        # PnP state — NEVER modified by BA
        self.pose_c2w = np.eye(4, dtype=np.float64)

        # previous frame data
        self.prev = None

        # keyframe window
        self.kfs     = []
        self.lms     = {}   # lm_id → {'pt': array(3), 'obs': {kf_id: array(2)}}
        self.lm_cnt  = 0
        self.kf_cnt  = 0
        self.frames_since_kf = 999

        # output lists (indexed by frame number)
        self.traj  = []  # array(3) per frame
        self.poses = []  # 4×4 per frame

    # ─── frame processing ──────────────────────────────────────────

    def process(self, idx, left, right):
        disp = get_disp(self.sgbm, left, right)
        kps, descs = self.orb.detectAndCompute(left, None)

        if idx == 0 or self.prev is None:
            self.prev = dict(kps=kps, descs=descs, disp=disp)
            self._record()
            self._try_add_kf(idx, kps, descs, disp)
            return

        if descs is None or len(kps) < 15 or self.prev['descs'] is None:
            self._record()
            self.prev = dict(kps=kps, descs=descs, disp=disp)
            self.frames_since_kf += 1
            return

        # ── feature matching ──────────────────────────────────────
        matches = list(self.bf.match(self.prev['descs'], descs))
        matches.sort(key=lambda m: m.distance)

        prev_pts = np.array([self.prev['kps'][m.queryIdx].pt for m in matches], np.float32)
        curr_pts = np.array([kps[m.trainIdx].pt              for m in matches], np.float32)

        # ── back-project previous frame → world 3D ────────────────
        pts3d_cam, vi = backproject(
            prev_pts, self.prev['disp'],
            self.fx, self.fy, self.cx, self.cy, self.baseline
        )
        if len(pts3d_cam) < 15:
            self._record()
            self.prev = dict(kps=kps, descs=descs, disp=disp)
            self.frames_since_kf += 1
            return

        pts3d_w   = (self.pose_c2w[:3,:3] @ pts3d_cam.T).T + self.pose_c2w[:3,3]
        curr_valid = curr_pts[vi]

        # ── PnP ──────────────────────────────────────────────────
        try:
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                pts3d_w.reshape(-1,1,3), curr_valid.reshape(-1,1,2),
                self.K, None,
                iterationsCount=300, reprojectionError=3.0,
                confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok and inliers is not None and len(inliers) >= 10:
                inl = inliers.flatten()
                rv, tv = cv2.solvePnPRefineLM(
                    pts3d_w[inl].reshape(-1,1,3),
                    curr_valid[inl].reshape(-1,1,2),
                    self.K, None, rv, tv,
                    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
                )
                new_c2w = Tinv(to_mat(rv, tv))
                dt = np.linalg.norm(new_c2w[:3,3] - self.pose_c2w[:3,3])
                if dt < MAX_FRAME_DISP:
                    self.pose_c2w = new_c2w
        except Exception:
            pass

        self._record()
        self._try_add_kf(idx, kps, descs, disp)
        self.prev = dict(kps=kps, descs=descs, disp=disp)
        self.frames_since_kf += 1

    # ─── keyframe management ───────────────────────────────────────

    def _try_add_kf(self, idx, kps, descs, disp):
        if self.frames_since_kf < KF_EVERY_N:
            return
        if len(self.kfs) > 0:
            dist = np.linalg.norm(self.pose_c2w[:3,3] - self.kfs[-1]['pose'][:3,3])
            if dist < KF_MIN_DIST:
                return

        kf_id  = self.kf_cnt; self.kf_cnt += 1
        traj_i = len(self.traj) - 1   # index in self.traj/poses for this KF
        kf = dict(id=kf_id, frame=idx, pose=self.pose_c2w.copy(),
                  traj_idx=traj_i,
                  kps=kps, descs=descs, disp=disp, kp2lm={})

        for prev_kf in self.kfs[-4:]:
            self._link(prev_kf, kf)

        self.kfs.append(kf)

        if len(self.kfs) >= 5:
            self._run_ba()

        while len(self.kfs) > WINDOW_SIZE:
            old = self.kfs.pop(0)
            self._prune(old['id'])

        self.frames_since_kf = 0

    def _link(self, kf_a, kf_b):
        if kf_a['descs'] is None or kf_b['descs'] is None:
            return
        matches = [m for m in self.bf.match(kf_a['descs'], kf_b['descs'])
                   if m.distance < 64]
        for m in matches:
            ai, bi = m.queryIdx, m.trainIdx
            pt_a = np.array(kf_a['kps'][ai].pt, dtype=np.float64)
            pt_b = np.array(kf_b['kps'][bi].pt, dtype=np.float64)

            if ai in kf_a['kp2lm']:
                lm_id = kf_a['kp2lm'][ai]
                if kf_b['id'] not in self.lms[lm_id]['obs']:
                    self.lms[lm_id]['obs'][kf_b['id']] = pt_b
                    kf_b['kp2lm'][bi] = lm_id
            else:
                pt3d, ok = backproject(
                    pt_a.reshape(1,2), kf_a['disp'],
                    self.fx, self.fy, self.cx, self.cy, self.baseline
                )
                if len(pt3d) == 0:
                    continue
                pt3d_w = kf_a['pose'][:3,:3] @ pt3d[0] + kf_a['pose'][:3,3]

                lm_id = self.lm_cnt; self.lm_cnt += 1
                self.lms[lm_id] = {'pt': pt3d_w,
                                   'obs': {kf_a['id']: pt_a, kf_b['id']: pt_b}}
                kf_a['kp2lm'][ai] = lm_id
                kf_b['kp2lm'][bi] = lm_id

    def _prune(self, removed_id):
        to_del = []
        for lm_id, lm in self.lms.items():
            lm['obs'].pop(removed_id, None)
            if len(lm['obs']) < 2:
                to_del.append(lm_id)
        for lm_id in to_del:
            del self.lms[lm_id]

    def _run_ba(self):
        accepted, updated_poses, updated_lm = self.ba.run(self.kfs, self.lms)
        if not accepted:
            return

        # ── update keyframe poses ─────────────────────────────────
        for i, kf in enumerate(self.kfs):
            kf['pose'] = updated_poses[i]

        # ── update stored OUTPUT for each KF frame ────────────────
        # (does NOT modify self.pose_c2w — PnP frontend is unaffected)
        for kf in self.kfs:
            ti = kf['traj_idx']
            if 0 <= ti < len(self.traj):
                self.traj[ti]  = kf['pose'][:3,3].copy()
                self.poses[ti] = kf['pose'].copy()

        # ── update landmarks ──────────────────────────────────────
        for lm_id, pt in updated_lm.items():
            if lm_id in self.lms:
                self.lms[lm_id]['pt'] = pt

    # ─── output ────────────────────────────────────────────────────

    def _record(self):
        self.traj.append(self.pose_c2w[:3,3].copy())
        self.poses.append(self.pose_c2w.copy())

    def save(self):
        os.makedirs(ART, exist_ok=True)
        with open(f'{ART}/traj.txt', 'w') as f:
            for p in self.traj:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        with open(f'{ART}/poses.txt', 'w') as f:
            for T in self.poses:
                row = T[:3].flatten()
                f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
        print(f'Saved {len(self.traj)} poses → {ART}')
        print(f'BA: accepted={self.ba.n_accepted}  rejected={self.ba.n_rejected}')


# ════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════

def main():
    vo = StereoVO()
    i = 0
    while True:
        lp = f'{DATA}/left_{i:06d}.png'
        rp = f'{DATA}/right_{i:06d}.png'
        if not os.path.exists(lp):
            break
        L = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            break
        vo.process(i, L, R)
        if i % 50 == 0:
            p  = vo.pose_c2w[:3,3]
            print(f'Frame {i:4d}  pos=({p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f})'
                  f'  kfs={len(vo.kfs)}  lms={len(vo.lms)}')
        i += 1

    vo.save()
    print(f'Done. {i} frames processed.')

if __name__ == '__main__':
    main()
