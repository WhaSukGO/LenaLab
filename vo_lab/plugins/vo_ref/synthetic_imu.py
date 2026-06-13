"""Synthetic IMU generator for the M3 (visual-inertial odometry) experiment.

Given a camera trajectory Twc[i] (cam->world) at frame rate dt, emit a body-frame IMU stream
(gyroscope angular velocity + accelerometer specific force) with a REALISTIC noise + bias model,
so that:
  • integrating the CLEAN IMU reconstructs the trajectory (the model is physically correct), but
  • integrating the NOISY IMU DRIFTS badly (bias random walk) — so the IMU is a real sensor to be
    FUSED with vision, NOT a free answer. This is what makes the VIO task honest.

Frame conventions: world y is DOWN (matches synthetic_stereo); gravity g_world = (0, +9.81, 0).
Accelerometer measures specific force f = a_world - g_world, expressed in the body (camera) frame.
"""
from __future__ import annotations
import numpy as np
import cv2

G_WORLD = np.array([0.0, 9.81, 0.0])    # gravity acceleration (y-down)


def _logSO3(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def _expSO3(w):
    R, _ = cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))
    return R


def generate_imu(Twc, dt=0.1, *, seed=0, gyro_noise=0.004, accel_noise=0.04,
                 gyro_bias_rw=3e-4, accel_bias_rw=1.5e-3, gyro_bias0=0.01, accel_bias0=0.05):
    """Return a dict with per-frame IMU readings (frame-rate; one sample per frame interval):
        'gyro'  : (n,3) body-frame angular velocity (rad/s)  [+ bias + white noise]
        'accel' : (n,3) body-frame specific force (m/s^2)     [+ bias + white noise]
        'dt'    : dt
    Sample i is the measurement over interval [i, i+1] (gyro/accel constant on the interval).
    Noise model: per-axis white noise + a slowly-drifting bias (random walk) starting at bias0.
    """
    rng = np.random.default_rng(seed)
    n = len(Twc)
    R = [T[:3, :3] for T in Twc]
    p = np.array([T[:3, 3] for T in Twc])
    # world acceleration via finite differences, boundaries replicated (no end spikes)
    v = np.diff(p, axis=0) / dt                            # (n-1,3) velocity over [k,k+1]
    a = np.zeros((n, 3))
    a[1:n - 1] = (v[1:] - v[:-1]) / dt                     # a[i]=(p[i+1]-2p[i]+p[i-1])/dt^2
    if n >= 3:
        a[0], a[n - 1] = a[1], a[n - 2]
    gyro, accel = [], []
    gb = rng.normal(0, gyro_bias0, 3)
    ab = rng.normal(0, accel_bias0, 3)
    for i in range(n):
        w = _logSO3(R[i].T @ R[min(i + 1, n - 1)]) / dt if i < n - 1 else np.zeros(3)
        f_body = R[i].T @ (a[i] - G_WORLD)                # specific force in body frame
        gb = gb + rng.normal(0, gyro_bias_rw, 3)          # bias random walk
        ab = ab + rng.normal(0, accel_bias_rw, 3)
        gyro.append(w + gb + rng.normal(0, gyro_noise, 3))
        accel.append(f_body + ab + rng.normal(0, accel_noise, 3))
    if n >= 2:
        gyro[n - 1] = gyro[n - 2]
    return {"gyro": np.array(gyro), "accel": np.array(accel), "dt": dt}


def dead_reckon(imu, T0, dt=None, v0=None):
    """Integrate an IMU stream from initial pose T0 (4x4 cam->world) and initial velocity v0 (world,
    default zero). Returns (n,3) positions. Used only to VERIFY the generator (clean+true v0 ->
    matches GT, noisy -> drifts); a real VIO must fuse this with vision because alone it drifts."""
    dt = dt or imu["dt"]
    gyro, accel = imu["gyro"], imu["accel"]
    n = len(gyro)
    R = T0[:3, :3].copy()
    pos = T0[:3, 3].copy().astype(float)
    vel = np.zeros(3) if v0 is None else np.asarray(v0, float).copy()
    out = [pos.copy()]
    for i in range(n - 1):
        a_world = R @ accel[i] + G_WORLD                  # rotate specific force back, add gravity
        pos = pos + vel * dt + 0.5 * a_world * dt * dt
        vel = vel + a_world * dt
        R = R @ _expSO3(gyro[i] * dt)
        out.append(pos.copy())
    return np.array(out)
