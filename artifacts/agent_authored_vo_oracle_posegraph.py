"""
Pose-graph optimisation for stereo SLAM.
- Imports the LOCKED front-end (frontend.py) for VO poses.
- Reads oracle loop closures from $LAB_DATA/loops.txt.
- Builds a pose graph (sequential odometry edges + loop edges).
- Optimises on SE(3) via scipy least_squares with a sparse Jacobian.
- Writes $LAB_ARTIFACTS/traj.txt  and  $LAB_ARTIFACTS/poses.txt.
"""

import os, sys
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

# ── SE(3) helpers ────────────────────────────────────────────────────────────

def skew(v):
    return np.array([[ 0.,   -v[2],  v[1]],
                     [ v[2],  0.,   -v[0]],
                     [-v[1],  v[0],  0.  ]])

def so3_exp(phi):
    """Rodrigues: rotation-vector → rotation matrix."""
    th = float(np.linalg.norm(phi))
    if th < 1e-10:
        return np.eye(3) + skew(phi)
    K = skew(phi / th)
    return np.eye(3) + np.sin(th)*K + (1.0 - np.cos(th))*(K @ K)

def so3_log(R):
    """Rotation matrix → rotation vector (Rodrigues)."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th    = np.arccos(cos_a)
    if th < 1e-10:
        return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    if abs(th - np.pi) < 1e-5:
        # near π: recover axis from diagonal
        d  = np.array([R[0,0], R[1,1], R[2,2]])
        ax = np.sqrt(np.maximum((d + 1.0) / 2.0, 0.0))
        # fix signs
        ax[0] *= (1.0 if (R[2,1] - R[1,2]) >= 0.0 else -1.0)
        ax[1] *= (1.0 if (R[0,2] - R[2,0]) >= 0.0 else -1.0)
        ax[2] *= (1.0 if (R[1,0] - R[0,1]) >= 0.0 else -1.0)
        nrm = np.linalg.norm(ax)
        return th * ax / (nrm + 1e-20)
    f = th / (2.0 * np.sin(th))
    return f * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def pose_from_xi(xi):
    """
    xi = [dt_x, dt_y, dt_z, dr_x, dr_y, dr_z]
    Returns 4×4 matrix  [[R, dt], [0, 1]].
    (Direct-product parameterisation; equals identity at xi=0.)
    """
    T = np.eye(4)
    T[:3, :3] = so3_exp(xi[3:])
    T[:3,  3] = xi[:3]
    return T

def se3_residual(T_err):
    """6D error from a 4×4 SE(3) error matrix: [t_err | r_err]."""
    return np.concatenate([T_err[:3, 3], so3_log(T_err[:3, :3])])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.environ['LAB_DATA']
    art_dir  = os.environ['LAB_ARTIFACTS']
    os.makedirs(art_dir, exist_ok=True)

    lab_code = os.environ.get('LAB_CODE', '/code')
    sys.path.insert(0, lab_code)
    from frontend import run_frontend

    # ── Front-end VO ─────────────────────────────────────────────────────────
    print("Running front-end VO …")
    fe   = run_frontend(data_dir)
    n    = fe['n']
    P    = fe['poses']           # list[n] of 4×4 float64  cam→world (Twc)
    print(f"  {n} frames")

    # ── Oracle loop closures ──────────────────────────────────────────────────
    # Format per line: i j  r11 r12 r13 tx  r21 r22 r23 ty  r31 r32 r33 tz
    # Constraint: Twc_j ≈ Twc_i @ T_ij
    loops = []
    with open(os.path.join(data_dir, 'loops.txt')) as fh:
        for line in fh:
            v = line.split()
            if len(v) < 14:
                continue
            i, j = int(v[0]), int(v[1])
            rv = list(map(float, v[2:]))
            T_ij = np.array([[rv[0], rv[1], rv[2],  rv[3]],
                             [rv[4], rv[5], rv[6],  rv[7]],
                             [rv[8], rv[9], rv[10], rv[11]],
                             [0.,    0.,    0.,     1.   ]], np.float64)
            loops.append((i, j, T_ij))
    print(f"  {len(loops)} loop closure(s)")

    # ── Build edge list ───────────────────────────────────────────────────────
    # Sequential odometry:  T_meas = P[k]^{-1} @ P[k+1]
    odom_edges = [(k, k+1, np.linalg.inv(P[k]) @ P[k+1]) for k in range(n-1)]
    loop_edges = [(i, j, T) for i, j, T in loops]
    all_edges  = odom_edges + loop_edges
    n_edges    = len(all_edges)

    # Weights: odometry = 1, oracle loops = very high (drift-free constraint)
    W = np.ones(n_edges)
    W[len(odom_edges):] = 500.0   # oracle loops are essentially perfect

    # ── State vector ──────────────────────────────────────────────────────────
    # Free poses:  k = 1 … n-1   (pose 0 fixed at P[0] = identity)
    # x[6*(k-1) : 6*k] = xi_k ∈ R^6  (right-perturbation increment)
    # T_k = P[k] @ pose_from_xi(xi_k)
    n_free = n - 1
    x0     = np.zeros(6 * n_free)

    def get_T(x, k):
        if k == 0:
            return P[0]
        xi = x[(k-1)*6 : k*6]
        return P[k] @ pose_from_xi(xi)

    def residuals(x):
        r = np.empty(6 * n_edges)
        for e, (i, j, T_meas) in enumerate(all_edges):
            Ti   = get_T(x, i)
            Tj   = get_T(x, j)
            T_err = np.linalg.inv(T_meas) @ np.linalg.inv(Ti) @ Tj
            r[e*6:(e+1)*6] = W[e] * se3_residual(T_err)
        return r

    # ── Sparse Jacobian pattern ───────────────────────────────────────────────
    # Residual block e (rows e*6..(e+1)*6) depends on poses i and j of that edge.
    rr, cc = [], []
    for e, (i, j, _) in enumerate(all_edges):
        for row in range(e*6, (e+1)*6):
            if i > 0:
                for col in range((i-1)*6, i*6):
                    rr.append(row); cc.append(col)
            if j > 0:
                for col in range((j-1)*6, j*6):
                    rr.append(row); cc.append(col)
    sp = csr_matrix((np.ones(len(rr), dtype=np.float64), (rr, cc)),
                    shape=(6*n_edges, 6*n_free))

    # ── Optimise ──────────────────────────────────────────────────────────────
    print("Optimising pose graph (scipy TRF + sparse Jacobian) …")
    result = least_squares(
        residuals, x0,
        jac_sparsity=sp,
        method='trf',
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=10000,
        verbose=1,
    )
    print(f"  cost = {result.cost:.6f}   [{result.message}]")

    # ── Verify loop closure residuals ─────────────────────────────────────────
    x_opt = result.x
    for li, (i, j, T_meas) in enumerate(loops):
        Ti  = get_T(x_opt, i)
        Tj  = get_T(x_opt, j)
        Ter = np.linalg.inv(T_meas) @ np.linalg.inv(Ti) @ Tj
        t_e = np.linalg.norm(Ter[:3, 3])
        r_e = np.degrees(np.linalg.norm(so3_log(Ter[:3, :3])))
        print(f"  loop ({i},{j}): t_err={t_e:.4f} m  r_err={r_e:.4f} deg")

    # ── Write artifacts ───────────────────────────────────────────────────────
    opt_P = [get_T(x_opt, k) for k in range(n)]

    traj = np.array([T[:3, 3] for T in opt_P])
    np.savetxt(os.path.join(art_dir, 'traj.txt'), traj, fmt='%.6f')

    poses_mat = np.array([T[:3, :4].ravel() for T in opt_P])
    np.savetxt(os.path.join(art_dir, 'poses.txt'), poses_mat, fmt='%.8e')

    print(f"Wrote traj.txt + poses.txt  ({n} poses)  →  {art_dir}")


if __name__ == '__main__':
    main()
