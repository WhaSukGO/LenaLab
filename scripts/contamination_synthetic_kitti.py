"""Option B contamination probe: PROCEDURALLY GENERATE a synthetic stereo VO sequence with
EXACT known ground truth (camera trajectory + intrinsics) — data no LLM has ever seen — then
prove it is a valid VO problem by running the proven classical reference stereo solver on it
(the POSITIVE CONTROL). All local / non-billed.

Geometry (rectified stereo, KITTI-like):
  - Intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]; stereo baseline b (m).
  - Right camera = left translated by +b along camera x-axis (standard rectified stereo).
    A 3D camera-frame point (X,Y,Z) -> left pixel (fx*X/Z+cx, fy*Y/Z+cy),
    right pixel (fx*(X-b)/Z+cx, fy*Y/Z+cy). Same row v; disparity = fx*b/Z. SGBM-recoverable.
  - Static 3D world = a set of textured PLANES (ground, ceiling, two side walls, a far back wall).
    Rendering is per-pixel inverse ray-casting: each image pixel's camera ray is intersected with
    every plane; the nearest hit in front of the camera gives exact metric depth Z, and a
    deterministic 3D procedural texture sampled at the world hit-point gives the pixel intensity.
    The RIGHT camera is the SAME rig translated +b in camera x, so it ray-casts the same planes and
    samples the same world texture -> dense continuous texture with EXACT disparity = fx*b/Z at
    every pixel (the rigorous stereo-consistent construction SGBM block-matching needs).
  - Smooth cam->world trajectory Twc[i] (forward motion + gentle yaw), a few hundred frames.

Writes held-out format under _contamination_B_run/seq_<name>/:
  input/{left_%06d.png,right_%06d.png,intrinsics.txt}, gt.txt (centres), gt_poses.txt (3x4 c2w).
Then runs run_kitti_stereo.py per seq, copies traj/poses -> traj_<s>/poses_<s>, grades eval_kitti.py.
"""
from __future__ import annotations
import os, sys, json, shutil, subprocess, time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
OUT = ROOT / "_contamination_B_run"
REF = ROOT / "vo_lab/plugins/vo_ref/run_kitti_stereo.py"
EVAL = ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py"

# ---- camera / image config (KITTI-ish) ----
W, H = 1226, 370
FX = FY = 707.0
CX, CY = 612.0, 185.0
BASELINE = 0.54
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


# ---- procedural 3D value-noise texture (deterministic; identical across frames and L/R views) ----
# Two perpendicular surface axes per plane are passed in so texture varies over the surface.
def _hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647).astype(np.uint64)
    h ^= (h >> np.uint64(13)); h = (h * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
    return (h.astype(np.float64) / 4294967295.0)


def _vnoise(P, scale):
    """Trilinear value-noise sampled at world points P (Nx3) at given cell `scale` (m)."""
    q = P / scale
    f = np.floor(q).astype(np.int64)
    t = q - f
    t = t * t * (3 - 2 * t)                       # smoothstep
    c = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (np.where(dx, t[:, 0], 1 - t[:, 0]) *
                     np.where(dy, t[:, 1], 1 - t[:, 1]) *
                     np.where(dz, t[:, 2], 1 - t[:, 2]))
                c = c + w * _hash3(f[:, 0] + dx, f[:, 1] + dy, f[:, 2] + dz)
    return c


def texture_at(P):
    """Multi-octave grayscale (0..1) procedural texture at world points P -> high local contrast
    so ORB finds corners AND SGBM block-matches (dense, non-repetitive)."""
    v = (0.62 * _vnoise(P, 0.45) + 0.38 * _vnoise(P, 1.8))
    return np.clip(v, 0, 1)


def make_world(rng, zmax):
    """World = textured infinite planes (corridor): ground, ceiling, two side walls, far back wall.
    Each plane: (normal n, offset D) with n·X = D, plus depth-validity handled at render time.
    `zmax` sets the back wall so there is always near structure (3..45 m) ahead of every frame."""
    planes = [
        # ground (road) at y=+5  (y is DOWN)            normal +y
        {"n": np.array([0.0, 1.0, 0.0]), "D": 5.0},
        # ceiling at y=-8                               normal -y
        {"n": np.array([0.0, -1.0, 0.0]), "D": 8.0},
        # left wall at x=-30                            normal +x
        {"n": np.array([1.0, 0.0, 0.0]), "D": -30.0},
        # right wall at x=+30                           normal -x
        {"n": np.array([-1.0, 0.0, 0.0]), "D": -30.0},
    ]
    # moving back wall: keep it ~70 m ahead of the furthest reach so a frontal surface always exists
    planes.append({"n": np.array([0.0, 0.0, -1.0]), "D": -(zmax + 70.0)})
    return planes


def make_trajectory(kind, n):
    """Return list of 4x4 cam->world Twc. Camera looks down +z (world). Forward = +z, gentle yaw."""
    Twc = []
    pos = np.zeros(3)
    yaw = 0.0
    speed = 0.85   # m/frame ~ KITTI-ish at ~10-12 km/h
    for i in range(n):
        if kind == "A":
            dyaw = 0.0030 * np.sin(i * 0.020)         # gentle S-curve (stays in wide corridor)
        else:
            dyaw = 0.0040 * np.sin(i * 0.012)         # broader sweeping turn
        yaw += dyaw
        # heading in world xz-plane; +z forward, yaw rotates about world y (down) axis
        fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        pos = pos + speed * fwd
        # small vertical bob so motion is not perfectly planar
        py = 0.05 * np.sin(i * 0.06)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)  # rot about y
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [pos[0], py, pos[2]]
        Twc.append(T)
    return Twc


# precompute camera-frame ray directions for every pixel (left rig). Right rig shares the same
# ray directions but its centre is offset +b in camera x (rectified stereo) -> standard model.
_uu, _vv = np.meshgrid(np.arange(W), np.arange(H))
_RAYS_CAM = np.stack([(_uu - CX) / FX, (_vv - CY) / FY, np.ones_like(_uu, float)], -1)  # H,W,3


def _render_view(planes, Twc, cam_offset_x, rng):
    """Inverse ray-cast one grayscale view. cam_offset_x shifts the centre +b for the right cam."""
    C = Twc[:3, 3] + Twc[:3, :3] @ np.array([cam_offset_x, 0.0, 0.0])   # camera centre in world
    dirs = _RAYS_CAM.reshape(-1, 3) @ Twc[:3, :3].T                     # ray dirs in world (N,3)
    N = dirs.shape[0]
    best_t = np.full(N, np.inf)
    hit = np.zeros((N, 3))
    for pl in planes:
        n, D = pl["n"], pl["D"]
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (D - C @ n) / denom
        ok = (denom != 0) & (t > 1.0) & (t < best_t)
        if ok.any():
            best_t[ok] = t[ok]
            hit[ok] = C + t[ok, None] * dirs[ok]
    valid = np.isfinite(best_t)
    tex = np.full(N, 0.06)
    if valid.any():
        tex[valid] = 0.10 + 0.85 * texture_at(hit[valid])
    img = (np.clip(tex, 0, 1) * 255).reshape(H, W).astype(np.uint8)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    return img


def render_pair(planes, Twc, rng):
    """Render a rectified stereo pair. Both cams ray-cast the SAME planes + SAME world texture;
    the right cam centre is +b along camera x, giving exact disparity = fx*b/Z per pixel."""
    left = _render_view(planes, Twc, 0.0, rng)
    right = _render_view(planes, Twc, BASELINE, rng)
    return left, right


def gen_sequence(name, kind, n, seed):
    rng = np.random.default_rng(seed)
    sq = OUT / f"seq_{name}"
    inp = sq / "input"
    inp.mkdir(parents=True, exist_ok=True)
    Twc = make_trajectory(kind, n)
    # back wall sits past the furthest reach so there is always frontal structure.
    zreach = max(float(T[2, 3]) for T in Twc)
    planes = make_world(rng, zmax=zreach)

    np.savetxt(inp / "intrinsics.txt", np.array([FX, FY, CX, CY, BASELINE]), fmt="%.6f")
    centres, poses = [], []
    t0 = time.time()
    for i, T in enumerate(Twc):
        left, right = render_pair(planes, T, rng)
        cv2.imwrite(str(inp / f"left_{i:06d}.png"), left)
        cv2.imwrite(str(inp / f"right_{i:06d}.png"), right)
        centres.append(T[:3, 3].copy())
        poses.append(T[:3, :4].reshape(-1).copy())
        if (i + 1) % 50 == 0:
            print(f"    seq_{name}: rendered {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    np.savetxt(sq / "gt.txt", np.array(centres), fmt="%.6f")
    np.savetxt(sq / "gt_poses.txt", np.array(poses), fmt="%.8e")
    print(f"  seq_{name}: {n} frames rendered in {time.time()-t0:.0f}s", flush=True)
    return name


def run_ref(name, art):
    sq = OUT / f"seq_{name}"
    env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(art))
    t0 = time.time()
    subprocess.run([sys.executable, str(REF)], env=env, check=True)
    print(f"  ref VO seq_{name} ran in {time.time()-t0:.0f}s", flush=True)
    shutil.copy(art / "traj.txt", art / f"traj_{name}.txt")
    shutil.copy(art / "poses.txt", art / f"poses_{name}.txt")
    # diagnostic: how many poses were "held"
    poses = np.loadtxt(art / f"poses_{name}.txt").reshape(-1, 12)
    return poses.shape[0]


def main():
    seqs = [("synthA", "A", 280, 12345), ("synthB", "B", 260, 67890)]
    print("[1] generating synthetic stereo sequences ...", flush=True)
    for name, kind, n, seed in seqs:
        gen_sequence(name, kind, n, seed)

    art = OUT / "art"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    print("[2] running reference stereo VO (positive control) ...", flush=True)
    for name, _, _, _ in seqs:
        run_ref(name, art)

    print("[3] grading with official eval_kitti.py ...", flush=True)
    evout = OUT / "eval"
    env = dict(os.environ, LAB_DATA=str(OUT), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(evout))
    subprocess.run([sys.executable, str(EVAL)], env=env, check=True)
    d = json.load(open(evout / "heldout.json"))
    print("\n=== POSITIVE CONTROL RESULT (synthetic stereo, official KITTI metric) ===")
    print("mode:", d.get("metric_mode"), "| mean t_err%:", round(d["t_err_pct"], 3),
          "| r_err deg/m:", d.get("r_err_deg_m"), "| ATE m:", round(d["ate_rmse"], 3))
    print("per-seq:", {k: {"t_err%": round(v["t_err_pct"], 3),
                           "ate": round(v["ate_rmse"], 3),
                           "len_m": v.get("path_len_m")} for k, v in d["per_seq"].items()})


if __name__ == "__main__":
    main()
