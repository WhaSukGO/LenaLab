"""LOCKED stereo-VO front-end for the M2 SCAFFOLD experiment.

This is the PROVEN reference stereo VO (same SGBM->ORB->PnP pipeline that scores ~2.81% t_err on
the held-out), refactored to be IMPORTABLE. The scaffold agent imports `run_frontend` and builds
ONLY a loop-closure + pose-graph layer on top of it — it must NOT reimplement or modify this file.

run_frontend(data_dir) returns the VO trajectory PLUS, per frame, everything a loop-closure layer
needs: ORB keypoints + descriptors (for appearance-based place recognition) and metric 3-D points
in the camera frame (for geometric verification / relative-pose estimation at a detected loop).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def run_frontend(data_dir, artifacts_dir=None):
    """Run the proven stereo VO. Returns a dict:
        'n'      : number of frames
        'K'      : 3x3 intrinsics
        'traj'   : (n,3) camera centres (world)
        'poses'  : list of n  4x4 cam->world matrices (the raw VO estimate)
        'frames' : list of n dicts, each:
                     'idx'   : frame index
                     'kps'   : (m,2) ORB keypoint pixel coords
                     'des'   : (m,32) uint8 ORB descriptors  (None if none found)
                     'pts3d' : (k,3) metric 3-D points in THIS frame's camera coords
                     'pidx'  : (k,) indices into kps that have a valid 3-D point
                     'Twc'   : 4x4 cam->world VO pose for this frame
    If artifacts_dir is given, also writes the raw VO traj.txt + poses.txt there (the un-closed
    baseline the loop-closure layer will improve on).
    """
    data = Path(data_dir)
    left = sorted(data.glob("left_*.png"))
    n = len(left)
    if n == 0:
        raise RuntimeError("no left_*.png frames in data_dir")

    fx, fy, cx, cy, baseline = np.loadtxt(data / "intrinsics.txt")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
                                 P1=8 * 7 * 7, P2=32 * 7 * 7, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2)

    def depth_of(i):
        l = cv2.imread(str(data / f"left_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(str(data / f"right_{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        disp = sgbm.compute(l, r).astype(np.float64) / 16.0
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(disp > 0.5, fx * baseline / disp, 0.0)
        return l, z

    def backproject(kps, depth):
        pts3d, idx = [], []
        for j, kp in enumerate(kps):
            u, v = kp.pt
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
                z = depth[vi, ui]
                if 1.0 < z < 60.0:
                    pts3d.append([(u - cx) * z / fx, (v - cy) * z / fy, z]); idx.append(j)
        return np.array(pts3d, np.float64).reshape(-1, 3), idx

    def frame_record(i, kps, depth, Twc):
        pts3d, pidx = backproject(kps, depth)
        return {"idx": i,
                "kps": np.array([kp.pt for kp in kps], np.float64).reshape(-1, 2),
                "des": None, "pts3d": pts3d, "pidx": np.array(pidx, int), "Twc": Twc.copy()}

    Twc = np.eye(4)
    traj = [Twc[:3, 3].copy()]
    poses = [Twc[:3, :4].reshape(-1).copy()]
    full_poses = [Twc.copy()]
    g0, z0 = depth_of(0)
    kp0, des0 = orb.detectAndCompute(g0, None)
    frames = [frame_record(0, kp0, z0, Twc)]; frames[0]["des"] = des0

    for i in range(1, n):
        g1, z1 = depth_of(i)
        kp1, des1 = orb.detectAndCompute(g1, None)
        ok = False
        if des0 is not None and des1 is not None and len(kp0) >= 6 and len(kp1) >= 6:
            matches = bf.match(des0, des1)
            if len(matches) >= 6:
                p3d_all, idx = backproject(kp0, z0)
                keep = {kp0_i: row for row, kp0_i in enumerate(idx)}
                obj, img = [], []
                for m in matches:
                    if m.queryIdx in keep:
                        obj.append(p3d_all[keep[m.queryIdx]]); img.append(kp1[m.trainIdx].pt)
                if len(obj) >= 6:
                    obj = np.array(obj, np.float64); img = np.array(img, np.float64)
                    okp, rvec, tvec, inl = cv2.solvePnPRansac(
                        obj, img, K, None, reprojectionError=2.0, iterationsCount=150)
                    if okp and inl is not None and len(inl) >= 6:
                        R, _ = cv2.Rodrigues(rvec)
                        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = tvec.ravel()
                        Twc = Twc @ np.linalg.inv(M)
                        ok = True
        traj.append(Twc[:3, 3].copy())
        poses.append(Twc[:3, :4].reshape(-1).copy())
        full_poses.append(Twc.copy())
        fr = frame_record(i, kp1, z1, Twc); fr["des"] = des1
        frames.append(fr)
        g0, z0, kp0, des0 = g1, z1, kp1, des1

    if artifacts_dir is not None:
        a = Path(artifacts_dir); a.mkdir(parents=True, exist_ok=True)
        np.savetxt(a / "traj.txt", np.array(traj), fmt="%.6f")
        np.savetxt(a / "poses.txt", np.array(poses), fmt="%.8e")

    return {"n": n, "K": K, "traj": np.array(traj), "poses": full_poses, "frames": frames}
