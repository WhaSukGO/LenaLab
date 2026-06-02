"""Reference RGB-D SLAM with loop closure — baseline for the SLAM experiment.

VO alone drifts on long sequences; SLAM adds GLOBAL consistency:
  1. front-end: frame-to-frame RGB-D PnP (metric), selecting KEYFRAMES
  2. loop closure: match each keyframe's features against temporally-distant keyframes;
     a geometrically-verified match (PnP + enough inliers) adds a loop constraint
  3. pose-graph optimization: optimize keyframe poses so sequential + loop constraints are
     globally consistent (self-contained SE(3) least-squares — no g2o/gtsam dependency)
  4. propagate the keyframe corrections back to a per-frame trajectory

Reads $LAB_DATA (frame_/depth_ + intrinsics fx fy cx cy depth_scale); writes
$LAB_ARTIFACTS/traj.txt (camera centre per frame). VO_DEGENERATE=1 -> static (control)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

KF_STRIDE = 8          # keyframe every N frames
MIN_LOOP_GAP = 15      # only consider loops to keyframes this many keyframes back
MIN_LOOP_MATCHES = 35
MIN_LOOP_INLIERS = 18


def _rvec(R):
    v, _ = cv2.Rodrigues(R); return v.ravel()


def _R(rv):
    R, _ = cv2.Rodrigues(np.asarray(rv, float)); return R


def _T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M


def main() -> int:
    data = Path(os.environ["LAB_DATA"]); art = Path(os.environ["LAB_ARTIFACTS"])
    art.mkdir(parents=True, exist_ok=True)
    frames = sorted(data.glob("frame_*.png")); n = len(frames)
    if n == 0:
        print("ERROR: no frames", file=sys.stderr); return 2
    if os.environ.get("VO_DEGENERATE") == "1":
        np.savetxt(art / "traj.txt", np.zeros((n, 3)), fmt="%.6f"); return 0

    fx, fy, cx, cy, ds = np.loadtxt(data / "intrinsics.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def load(i):
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(str(data / f"depth_{i:04d}.png"), cv2.IMREAD_UNCHANGED).astype(float) / ds
        return g, d

    def feats(g):
        return orb.detectAndCompute(g, None)

    def backproj(kps, depth):
        p3, idx = [], []
        for j, kp in enumerate(kps):
            u, v = kp.pt; ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
                z = depth[vi, ui]
                if 0.1 < z < 8.0:
                    p3.append([(u - cx) * z / fx, (v - cy) * z / fy, z]); idx.append(j)
        return np.array(p3, float), idx

    def rel_pnp(kpA, desA, depthA, kpB, desB):
        """Relative pose cam_A -> cam_B (X_B = R X_A + t) via 3D(A)-2D(B) PnP, + inlier count."""
        if desA is None or desB is None or len(kpA) < 6 or len(kpB) < 6:
            return None, 0
        ms = bf.match(desA, desB)
        if len(ms) < 6:
            return None, 0
        p3, idx = backproj(kpA, depthA); keep = {q: r for r, q in enumerate(idx)}
        obj, img = [], []
        for m in ms:
            if m.queryIdx in keep:
                obj.append(p3[keep[m.queryIdx]]); img.append(kpB[m.trainIdx].pt)
        if len(obj) < 6:
            return None, 0
        ok, rv, tv, inl = cv2.solvePnPRansac(np.array(obj), np.array(img), K, None,
                                             reprojectionError=3.0, iterationsCount=100)
        if not ok or inl is None:
            return None, 0
        return _T(_R(rv), tv), len(inl)

    # --- front-end VO + keyframes -------------------------------------------------------
    Twc = np.eye(4); vo = [Twc.copy()]
    g0, d0 = load(0); kp0, de0 = feats(g0)
    kf_idx = [0]; kf = {0: (kp0, de0, d0)}
    for i in range(1, n):
        g1, d1 = load(i); kp1, de1 = feats(g1)
        M, ninl = rel_pnp(kp0, de0, d0, kp1, de1)
        if M is not None and ninl >= 10:
            Twc = Twc @ np.linalg.inv(M)
        vo.append(Twc.copy())
        if i % KF_STRIDE == 0:
            kf_idx.append(i); kf[i] = (kp1, de1, d1)
        g0, d0, kp0, de0 = g1, d1, kp1, de1
    vo = np.array(vo)

    # --- loop detection (geometrically verified) ---------------------------------------
    edges = []  # (src_kf_pos, dst_kf_pos, M)  with X_dst = M X_src
    for a in range(1, len(kf_idx)):                     # sequential edges (from VO)
        ia, ib = kf_idx[a - 1], kf_idx[a]
        edges.append((a - 1, a, np.linalg.inv(vo[ib]) @ vo[ia]))
    n_loops = 0
    for a in range(MIN_LOOP_GAP, len(kf_idx)):          # loop edges
        ia = kf_idx[a]; kpA, deA, dA = kf[ia]
        for b in range(0, a - MIN_LOOP_GAP):
            ib = kf_idx[b]; kpB, deB, _dB = kf[ib]
            if deA is None or deB is None:
                continue
            if len(bf.match(deA, deB)) < MIN_LOOP_MATCHES:
                continue
            M, ninl = rel_pnp(kpA, deA, dA, kpB, deB)   # cam_a -> cam_b
            if M is not None and ninl >= MIN_LOOP_INLIERS:
                edges.append((a, b, M)); n_loops += 1
                break                                    # one loop edge per keyframe is enough
    print(f"keyframes={len(kf_idx)}  loop edges detected={n_loops}")

    # --- pose-graph optimization (fix keyframe 0) ---------------------------------------
    init = np.concatenate([np.concatenate([_rvec(vo[kf_idx[k]][:3, :3]), vo[kf_idx[k]][:3, 3]])
                           for k in range(1, len(kf_idx))]) if len(kf_idx) > 1 else np.zeros(0)
    T0 = vo[kf_idx[0]].copy()

    def poses(p):
        Ts = [T0]
        for k in range(len(kf_idx) - 1):
            seg = p[6 * k:6 * k + 6]; Ts.append(_T(_R(seg[:3]), seg[3:]))
        return Ts

    def resid(p):
        Ts = poses(p); r = []
        for src, dst, M in edges:
            pred = np.linalg.inv(Ts[dst]) @ Ts[src]      # predicted cam_src -> cam_dst
            err = M @ np.linalg.inv(pred)
            r.extend(_rvec(err[:3, :3])); r.extend(err[:3, 3])
        return np.array(r)

    if len(init) and n_loops > 0:
        sol = least_squares(resid, init, method="trf", loss="soft_l1", max_nfev=200)
        opt = poses(sol.x)
    else:
        opt = poses(init)                                # no loops -> nothing to correct

    # --- propagate keyframe corrections to a per-frame trajectory -----------------------
    corr = {}                                            # frame_idx -> correction T_opt @ inv(T_vo)
    for k, fidx in enumerate(kf_idx):
        corr[fidx] = opt[k] @ np.linalg.inv(vo[fidx])
    kf_arr = np.array(kf_idx)
    traj = []
    for f in range(n):
        kpos = kf_arr[np.searchsorted(kf_arr, f, side="right") - 1]
        Twc_corr = corr[kpos] @ vo[f]
        traj.append(Twc_corr[:3, 3])
    np.savetxt(art / "traj.txt", np.array(traj), fmt="%.6f")
    print(f"SLAM trajectory written: {n} frames, {n_loops} loop closures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
