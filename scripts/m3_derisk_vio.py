"""M3 DE-RISK (offline, non-billed): does fusing the IMU actually rescue the trajectory through
vision blackouts? Generates a synthetic VIO sequence (stereo + IMU + blackouts), then compares:
  (1) stereo VO ALONE (the proven front-end) — should fail badly on the blackouts;
  (2) a reference loosely-coupled VIO that bridges blackouts with the IMU — should recover.
If (2) << (1), fusion has real headroom and the billed agent VIO run is worth it.
"""
import os, sys, json, tempfile, subprocess, shutil
from pathlib import Path
import numpy as np
import cv2
ROOT = Path("/home/ws/devel/whasuk/LenaLab"); sys.path.insert(0, str(ROOT))
from vo_lab.plugins.vo_ref.synthetic_vio import generate_vio_sequence, DT
from vo_lab.plugins.vo_ref.synthetic_imu import G_WORLD
from vo_lab.plugins.vo_ref.frontend_kitti import run_frontend

EVAL = ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py"


def grade(poses_by_seq, ho):
    art = Path(tempfile.mkdtemp())
    for s, P in poses_by_seq.items():
        np.savetxt(art / f"poses_{s}.txt", np.array([T[:3, :].reshape(-1) for T in P]), fmt="%.8e")
        np.savetxt(art / f"traj_{s}.txt", np.array([T[:3, 3] for T in P]), fmt="%.6f")
    ev = art / "eval"
    subprocess.run([sys.executable, str(EVAL)],
                   env=dict(os.environ, LAB_DATA=str(ho), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                   check=True, capture_output=True)
    return json.load(open(ev / "heldout.json"))


def expSO3(w):
    R, _ = cv2.Rodrigues(np.asarray(w, float).reshape(3, 1)); return R


def reference_vio(fe, imu, blackouts, dt):
    """Loosely-coupled: VO relative pose where vision works; integrate IMU through blackouts using
    the velocity carried from the last good VO step. (Uses known blackout ranges = the oracle the
    agent must instead DETECT; this only proves headroom exists.)"""
    Pvo = fe["poses"]; n = len(Pvo)
    gyro, accel = imu["gyro"], imu["accel"]
    bo = lambda i: any(s <= i < s + L for s, L in blackouts)
    pose = Pvo[0].copy(); vel = (Pvo[1][:3, 3] - Pvo[0][:3, 3]) / dt
    out = [pose.copy()]
    for i in range(1, n):
        if not bo(i):
            rel = np.linalg.inv(Pvo[i - 1]) @ Pvo[i]
            new = pose @ rel
            vel = (new[:3, 3] - pose[:3, 3]) / dt
            pose = new
        else:
            R = pose[:3, :3]; p = pose[:3, 3]
            a_world = R @ accel[i - 1] + G_WORLD
            p = p + vel * dt + 0.5 * a_world * dt * dt
            vel = vel + a_world * dt
            R = R @ expSO3(gyro[i - 1] * dt)
            pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = p
        out.append(pose.copy())
    return out


def main():
    work = Path(tempfile.mkdtemp(prefix="m3derisk_")); ho = work / "ho"
    specs = [("vioA", "A", 280, 101), ("vioB", "B", 300, 202)]
    vo_poses, vio_poses, info = {}, {}, {}
    for name, kind, n, seed in specs:
        sq = ho / f"seq_{name}"
        meta = generate_vio_sequence(sq / "input", gt_dir=sq, kind=kind, n=n, seed=seed)
        info[name] = meta
        fe = run_frontend(sq / "input")                       # stereo VO ALONE
        imu = np.loadtxt(sq / "input" / "imu.txt").reshape(-1, 6)
        imud = {"gyro": imu[:, :3], "accel": imu[:, 3:], "dt": DT}
        vo_poses[name] = fe["poses"]
        vio_poses[name] = reference_vio(fe, imud, meta["blackouts"], DT)
        print(f"  seq_{name}: {n} frames, blackouts {meta['blackouts']}", flush=True)
    vo = grade(vo_poses, ho); vio = grade(vio_poses, ho)
    res = {
        "VO_alone_t_err": round(vo["t_err_pct"], 3),
        "VO_alone_per_seq": {k: round(v["t_err_pct"], 2) for k, v in vo["per_seq"].items()},
        "reference_VIO_t_err": round(vio["t_err_pct"], 3),
        "reference_VIO_per_seq": {k: round(v["t_err_pct"], 2) for k, v in vio["per_seq"].items()},
    }
    res["headroom"] = vio["t_err_pct"] < 0.6 * vo["t_err_pct"]
    Path("/tmp/m3_derisk.txt").write_text(json.dumps(res, indent=2))
    print("\n=== M3 VIO DE-RISK ===")
    print(json.dumps(res, indent=2))
    print(f"-> fusion {'HAS real headroom -> billed VIO worth it' if res['headroom'] else 'gives little gain -> reconsider'}")


if __name__ == "__main__":
    main()
