"""
Visual-Inertial Odometry (VIO) — loosely-coupled fusion of stereo VO + IMU.

Key insight: after a long blackout the VO "recovery" pose is WRONG in absolute
terms because it accumulates motion relative to the stale held pre-blackout pose.
We fix this with an offset-tracking scheme:
  • During good VO (no recent blackout): corrected_pos = vo_pos + vo_offset
  • During blackout: IMU integration (gyro + gravity-subtracted accel,
      velocity carried from last good VO segment)
  • At recovery (first good frame after blackout): IMU step gives the NEW
      absolute position; set vo_offset = imu_pos - vo_pos_at_recovery so
      subsequent VO frames are correctly anchored.
"""

import numpy as np
import os
import sys
from pathlib import Path

_LAB_CODE = os.environ.get('LAB_CODE', '/code')
if _LAB_CODE not in sys.path:
    sys.path.insert(0, _LAB_CODE)

from frontend import run_frontend


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def skew(v):
    return np.array([
        [ 0.0,  -v[2],  v[1]],
        [ v[2],  0.0,  -v[0]],
        [-v[1],  v[0],  0.0 ]
    ])


def rodrigues_rot(w, dt):
    """Rotation matrix from angular velocity w (rad/s) and time dt (s)."""
    angle = np.linalg.norm(w) * dt
    if angle < 1e-9:
        return np.eye(3) + skew(w) * dt
    axis = w / np.linalg.norm(w)
    K = skew(axis)
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# ---------------------------------------------------------------------------
# Core VIO processor
# ---------------------------------------------------------------------------

def process_sequence(data_dir, artifacts_dir,
                     traj_fname='traj.txt', poses_fname='poses.txt'):
    print(f"[VIO] Processing: {data_dir}")

    # ---- 1. Stereo VO -------------------------------------------------------
    fe = run_frontend(data_dir)
    n        = fe['n']
    vo_poses = fe['poses']   # list of n 4×4 cam->world matrices
    frames   = fe['frames']

    # ---- 2. IMU -------------------------------------------------------------
    imu_path = os.path.join(data_dir, 'imu.txt')
    imu = np.loadtxt(imu_path)  # (n, 6): wx wy wz ax ay az
    assert imu.ndim == 2 and imu.shape[1] == 6

    dt = 0.1  # seconds

    # At rest (R=I): ay≈-9.82  →  a_world = R@a - g_world ≈ 0
    # So g_world = (0, -9.81, 0)
    g_world = np.array([0.0, -9.81, 0.0])

    # ---- 3. Vision-failure detection ----------------------------------------
    MIN_PTS = 15

    is_good = np.array([len(frames[i]['pts3d']) >= MIN_PTS for i in range(n)])
    # Also mark held VO poses as failures
    for i in range(1, n):
        if np.linalg.norm(vo_poses[i][:3, 3] - vo_poses[i-1][:3, 3]) < 1e-7:
            is_good[i] = False

    n_good = int(is_good.sum())
    print(f"[VIO]   frames={n}  good={n_good}  failed={n-n_good} ({100*(n-n_good)/n:.1f}%)")

    # ---- 4. Build fused trajectory with offset tracking --------------------
    fused_poses = [None] * n
    fused_poses[0] = vo_poses[0].copy()

    cur_R = vo_poses[0][:3, :3].copy()
    cur_p = vo_poses[0][:3, 3].copy()
    cur_v = np.zeros(3)

    # Offset: fused_position = vo_absolute_position + vo_offset
    # Updated at each blackout recovery to keep absolute positions consistent
    vo_offset = np.zeros(3)
    prev_was_blackout = False

    # Velocity history (only good frames, using corrected positions)
    VEL_WINDOW = 5
    vel_history = [(0, cur_p.copy())]

    for i in range(1, n):
        imu_meas = imu[i - 1]   # measurement for interval [i-1, i]
        w = imu_meas[:3]
        a = imu_meas[3:]

        if is_good[i]:
            if not prev_was_blackout:
                # ── Normal VO frame ──────────────────────────────────────────
                # Apply accumulated offset to VO absolute position
                corrected_p = vo_poses[i][:3, 3] + vo_offset
                cur_R = vo_poses[i][:3, :3].copy()
                cur_p = corrected_p

                pose = vo_poses[i].copy()
                pose[:3, 3] = corrected_p
                fused_poses[i] = pose

            else:
                # ── Recovery from blackout ───────────────────────────────────
                # Do one more IMU step from the last blackout frame
                R_delta  = rodrigues_rot(w, dt)
                new_R    = cur_R @ R_delta
                a_world  = cur_R @ a - g_world
                new_p    = cur_p + cur_v * dt + 0.5 * a_world * (dt**2)
                new_v    = cur_v + a_world * dt

                # Update the VO offset: difference between our IMU absolute
                # position and the (wrongly-anchored) VO absolute position
                vo_offset = new_p - vo_poses[i][:3, 3]

                # Use VO rotation (more precise than accumulated IMU rotation)
                cur_R = vo_poses[i][:3, :3].copy()
                cur_p = new_p
                cur_v = new_v

                pose = np.eye(4)
                pose[:3, :3] = cur_R
                pose[:3, 3]  = cur_p
                fused_poses[i] = pose

            # Update velocity estimate from sliding window of good frames
            vel_history.append((i, cur_p.copy()))
            if len(vel_history) > VEL_WINDOW:
                vel_history.pop(0)
            if len(vel_history) >= 2:
                dt_total = (vel_history[-1][0] - vel_history[0][0]) * dt
                if dt_total > 1e-9:
                    cur_v = (vel_history[-1][1] - vel_history[0][1]) / dt_total

            prev_was_blackout = False

        else:
            # ── IMU integration ──────────────────────────────────────────────
            R_delta = rodrigues_rot(w, dt)
            new_R   = cur_R @ R_delta
            a_world = cur_R @ a - g_world
            new_p   = cur_p + cur_v * dt + 0.5 * a_world * (dt**2)
            new_v   = cur_v + a_world * dt

            pose = np.eye(4)
            pose[:3, :3] = new_R
            pose[:3, 3]  = new_p
            fused_poses[i] = pose

            cur_R = new_R
            cur_p = new_p
            cur_v = new_v
            prev_was_blackout = True

    # Fallback
    for i in range(n):
        if fused_poses[i] is None:
            fused_poses[i] = vo_poses[i].copy()

    # ---- 5. Write artifacts -------------------------------------------------
    art = Path(artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)

    traj_arr  = np.array([p[:3, 3]          for p in fused_poses])
    poses_arr = np.array([p[:3, :4].ravel()  for p in fused_poses])

    np.savetxt(art / traj_fname,  traj_arr,  fmt='%.6f')
    np.savetxt(art / poses_fname, poses_arr, fmt='%.8e')

    path_len = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1)))
    print(f"[VIO]   path={path_len:.2f} m  start={traj_arr[0]}  end={traj_arr[-1]}")
    print(f"[VIO]   offset after all blackouts: {vo_offset}")
    print(f"[VIO]   Written: {traj_fname}, {poses_fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    data_dir      = os.environ['LAB_DATA']
    artifacts_dir = os.environ['LAB_ARTIFACTS']
    data_path     = Path(data_dir)

    seqs = sorted(data_path.glob("seq_*"))
    if seqs:
        print(f"[VIO] Found {len(seqs)} sequence(s) under {data_dir}")
        for seq_path in seqs:
            seq_name = seq_path.name.replace("seq_", "")
            process_sequence(
                str(seq_path), artifacts_dir,
                traj_fname  = f"traj_{seq_name}.txt",
                poses_fname = f"poses_{seq_name}.txt",
            )
    else:
        process_sequence(data_dir, artifacts_dir)


if __name__ == "__main__":
    main()
