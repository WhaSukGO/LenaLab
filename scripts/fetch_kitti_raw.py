"""Fetch + convert KITTI RAW drives into our provider contract — REAL, environment-labeled driving
data with REAL OXTS IMU/GPS and GT poses (replaces the synthetic domain for the SLAM-verification work).

Per drive it downloads <date>_drive_<id>_sync.zip (images + oxts) + <date>_calib.zip, then writes:
  data/kitti_raw/seq_<env>_<date>_<id>/input/{left_%06d.png, right_%06d.png, intrinsics.txt, imu.txt}
                                       /{gt.txt (centres), gt_poses.txt (3x4 cam~imu->world)}
intrinsics: fx fy cx cy baseline (from P_rect_00 / P_rect_01).  imu.txt: `wf wl wu af al au` per frame
(OXTS vehicle-frame gyro rad/s + accel m/s^2).  GT poses: OXTS lat/lon/alt + r/p/y via Mercator,
first-frame origin (IMU frame ~ camera frame to a small constant SE3 offset, removed by the grader's
global alignment).

usage: python scripts/fetch_kitti_raw.py <date> <drive_id> <env>
   e.g. python scripts/fetch_kitti_raw.py 2011_09_26 0001 city
"""
import sys, os, zipfile, shutil, urllib.request
from pathlib import Path
import numpy as np

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
CACHE = Path.home() / ".cache/vo_lab/kitti_raw"
OUT = Path("/home/ws/devel/whasuk/LenaLab/data/kitti_raw")


def _dl(url, dst):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  cached {dst.name}"); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, dst)
    print(f"  -> {dst.name} ({dst.stat().st_size/1e6:.0f} MB)")


def _read_calib(p):
    rows = {}
    for ln in Path(p).read_text().splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            try: rows[k.strip()] = np.fromstring(v, sep=" ")
            except Exception: pass
    P0 = rows["P_rect_00"].reshape(3, 4); P1 = rows["P_rect_01"].reshape(3, 4)
    fx, fy, cx, cy = P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2]
    baseline = -P1[0, 3] / fx
    return fx, fy, cx, cy, baseline


def _load_RT(path):                              # KITTI calib "R:/T:" -> 4x4
    rows = {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("R:") or ln.startswith("T:"):
            rows[ln[0]] = np.fromstring(ln.split(":", 1)[1], sep=" ")
    T = np.eye(4); T[:3, :3] = rows["R"].reshape(3, 3); T[:3, 3] = rows["T"]
    return T


def _T_cam_imu(ex, date):                        # rectified left-cam (cam0) <- imu
    T_velo_imu = _load_RT(ex / date / "calib_imu_to_velo.txt")
    T_cam_velo = _load_RT(ex / date / "calib_velo_to_cam.txt")
    R_rect = None
    for ln in (ex / date / "calib_cam_to_cam.txt").read_text().splitlines():
        if ln.startswith("R_rect_00:"):
            R_rect = np.fromstring(ln.split(":", 1)[1], sep=" ").reshape(3, 3)
    Rr = np.eye(4); Rr[:3, :3] = R_rect
    return Rr @ T_cam_velo @ T_velo_imu


def _oxts_to_poses(oxts):                       # oxts: (n,30) array -> IMU poses in world
    er = 6378137.0
    lat0 = oxts[0, 0]; scale = np.cos(lat0 * np.pi / 180.0)
    poses = []; T0inv = None
    for o in oxts:
        lat, lon, alt, roll, pitch, yaw = o[0], o[1], o[2], o[3], o[4], o[5]
        mx = scale * lon * np.pi * er / 180.0
        my = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = [mx, my, alt]
        if T0inv is None: T0inv = np.linalg.inv(T)
        poses.append(T0inv @ T)
    return poses


def main(date, drive, env):
    drive = drive.zfill(4)
    sync = f"{date}_drive_{drive}_sync"
    _dl(f"{BASE}/{date}_drive_{drive}/{sync}.zip", CACHE / f"{sync}.zip")
    _dl(f"{BASE}/{date}_calib.zip", CACHE / f"{date}_calib.zip")
    ex = CACHE / "extract"
    for z in [CACHE / f"{sync}.zip", CACHE / f"{date}_calib.zip"]:
        with zipfile.ZipFile(z) as zf: zf.extractall(ex)
    root = ex / date / sync
    fx, fy, cx, cy, baseline = _read_calib(ex / date / "calib_cam_to_cam.txt")
    oxts_files = sorted((root / "oxts" / "data").glob("*.txt"))
    oxts = np.array([np.fromstring(f.read_text(), sep=" ")[:30] for f in oxts_files])
    imu_poses = _oxts_to_poses(oxts)            # IMU-frame poses in world
    # transform to CAMERA frame (so GT matches camera-frame VO): T_w_cam = T_w_imu @ inv(T_cam_imu),
    # then re-origin to the first camera pose.
    Tci = _T_cam_imu(ex, date); Tic = np.linalg.inv(Tci)
    cam = [T @ Tic for T in imu_poses]
    C0inv = np.linalg.inv(cam[0]); poses = [C0inv @ T for T in cam]
    n = len(oxts_files)

    out = OUT / f"seq_{env}_{date.replace('_','')}_{drive}"
    inp = out / "input"; inp.mkdir(parents=True, exist_ok=True)
    np.savetxt(inp / "intrinsics.txt", np.array([fx, fy, cx, cy, baseline]), fmt="%.6f")
    left = sorted((root / "image_00" / "data").glob("*.png"))
    right = sorted((root / "image_01" / "data").glob("*.png"))
    for i in range(n):
        shutil.copy(left[i], inp / f"left_{i:06d}.png")
        shutil.copy(right[i], inp / f"right_{i:06d}.png")
    # imu: vehicle-frame gyro (wf,wl,wu = idx 20,21,22) + accel (af,al,au = idx 14,15,16)
    imu = np.column_stack([oxts[:, 20], oxts[:, 21], oxts[:, 22], oxts[:, 14], oxts[:, 15], oxts[:, 16]])
    np.savetxt(inp / "imu.txt", imu, fmt="%.8e")
    centres = np.array([T[:3, 3] for T in poses])
    np.savetxt(out / "gt.txt", centres, fmt="%.6f")
    np.savetxt(out / "gt_poses.txt", np.array([T[:3, :4].reshape(-1) for T in poses]), fmt="%.8e")
    print(f"WROTE {out}  ({n} frames, env={env}, fx={fx:.1f} baseline={baseline:.3f})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: fetch_kitti_raw.py <date> <drive_id> <env>"); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
