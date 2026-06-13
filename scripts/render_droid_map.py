"""Render DROID's dense reconstruction as a 3D point-cloud map of the driving scene + camera trajectory.
Runs INSIDE vo-droid:1 (needs torch). Back-projects per-keyframe inverse-depth (disps_up, full res) to world
points using the inverted poses (DROID poses are world->cam). Output: top-down colored map + oblique 3D view.

usage (in container): python render_droid_map.py /out/recon_<seq>.pth /out/map_<seq>.png [seq]
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quat_to_R(q):                                            # q = [qx,qy,qz,qw]
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    recon, outpng = sys.argv[1], sys.argv[2]
    seq = sys.argv[3] if len(sys.argv) > 3 else ""
    d = torch.load(recon, map_location="cpu")
    poses = d["poses"].numpy()                              # [N,7] world->cam
    disps = d["disps"].numpy()                              # [N,H,W] inverse depth (full res)
    images = d["images"].numpy()                            # [N,3,H,W] uint8 (BGR)
    intr = d["intrinsics"].numpy()                          # [N,4] at /8 res
    N, H, W = disps.shape
    print(f"recon: {N} keyframes, depth {H}x{W}")

    pts, cols, cams = [], [], []
    step = 3                                                # pixel subsample
    for i in range(N):
        fx, fy, cx, cy = intr[i] * 8.0                     # disps_up is full res -> intrinsics x8
        R = quat_to_R(poses[i, 3:7]); t = poses[i, :3]     # world->cam
        C = -R.T @ t                                        # camera centre in world
        cams.append(C)
        depth = 1.0 / np.clip(disps[i], 1e-4, None)
        img = images[i].transpose(1, 2, 0)[:, :, ::-1]      # BGR->RGB
        vs, us = np.mgrid[0:H:step, 0:W:step]; vs = vs.ravel(); us = us.ravel()
        z = depth[vs, us]
        m = (z > 0.5) & (z < 60) & (disps[i][vs, us] > 0.4 * disps[i].mean())   # valid + DROID conf-ish
        vs, us, z = vs[m], us[m], z[m]
        cam = np.stack([z * (us - cx) / fx, z * (vs - cy) / fy, z], 1)          # [M,3] camera frame
        pts.append(cam @ R + C)                             # world = R^T @ cam + C  == cam @ R + C
        cols.append(img[vs, us] / 255.0)
    P = np.concatenate(pts); Cc = np.concatenate(cols); cams = np.array(cams)
    # cap points for plotting
    if len(P) > 400000:
        k = np.random.default_rng(0).choice(len(P), 400000, replace=False); P, Cc = P[k], Cc[k]
    print(f"  {len(P)} map points, {len(cams)} cameras")

    height = -P[:, 1]                                       # Y is down in cam/world -> negate for "up"
    hclip = np.clip(height, np.percentile(height, 2), np.percentile(height, 98))
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121)                              # top-down map (X-Z), colored by height
    sc = ax1.scatter(P[:, 0], P[:, 2], c=hclip, cmap="viridis", s=0.6, marker=".", linewidths=0)
    ax1.plot(cams[:, 0], cams[:, 2], "r-", lw=2, label="camera trajectory")
    ax1.scatter([cams[0, 0]], [cams[0, 2]], c="red", s=70, marker="*", zorder=5, label="start/end (loop)")
    ax1.set_title(f"DROID-SLAM dense map {seq} — top-down (reconstructed scene, colored by height)")
    ax1.axis("equal"); ax1.legend(loc="upper right"); ax1.set_xlabel("x (m)"); ax1.set_ylabel("z (m)")
    plt.colorbar(sc, ax=ax1, label="height (m)", shrink=0.6)
    ax2 = fig.add_subplot(122, projection="3d")             # oblique 3D, colored by photo
    ax2.scatter(P[:, 0], P[:, 2], height, c=Cc, s=0.5, marker=".", linewidths=0)
    ax2.plot(cams[:, 0], cams[:, 2], -cams[:, 1], "r-", lw=2)
    ax2.set_title("oblique 3D view (photo-colored)"); ax2.view_init(elev=28, azim=-65)
    try: ax2.set_box_aspect((1, 1, 0.35))
    except Exception: pass
    fig.tight_layout(); fig.savefig(outpng, dpi=130)
    print(f"  wrote {outpng}")


if __name__ == "__main__":
    main()
