"""Phase-0 de-risk part 1: rasterize vehicle 3D boxes into an ego VOXEL grid (occupancy GT),
probe where vehicles land in Z (to set the grid range empirically), and visualize (height-colored
top-down + a few Z-slices) to confirm the GT is geometrically sane. Runs in vo-bev:1.

usage: python occ_derisk_gt.py <nuscenes_root> <out_dir>
"""
import sys, os
import numpy as np, cv2
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
XB = YB = (-50.0, 50.0)
ZB = (-5.0, 5.0)                      # WIDE probe range; we'll tighten from the histogram
RES = 0.5
XG = int((XB[1] - XB[0]) / RES); YG = int((YB[1] - YB[0]) / RES); ZG = int((ZB[1] - ZB[0]) / RES)
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)

# voxel center coordinates (ego frame)
xs = XB[0] + (np.arange(XG) + 0.5) * RES
ys = YB[0] + (np.arange(YG) + 0.5) * RES
zs = ZB[0] + (np.arange(ZG) + 0.5) * RES


def voxelize(sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    trans = -np.array(ego["translation"]); rot = Quaternion(ego["rotation"]).inverse
    grid = np.zeros((XG, YG, ZG), np.uint8)
    nb = 0
    for tok in sample["anns"]:
        ann = nusc.get("sample_annotation", tok)
        if not ann["category_name"].startswith("vehicle"):
            continue
        box = nusc.get_box(ann["token"]); box.translate(trans); box.rotate(rot)   # global->ego
        c = box.center; R = box.rotation_matrix                      # box->ego
        w, l, h = box.wlh
        half = np.array([l / 2, w / 2, h / 2])                       # local x=length,y=width,z=height
        # AABB of the box in ego, clipped to grid index ranges (fill only nearby voxels)
        corners = box.corners().T                                    # (8,3) ego
        mn, mx = corners.min(0), corners.max(0)
        ix0, ix1 = np.searchsorted(xs, mn[0]) - 1, np.searchsorted(xs, mx[0]) + 1
        iy0, iy1 = np.searchsorted(ys, mn[1]) - 1, np.searchsorted(ys, mx[1]) + 1
        iz0, iz1 = np.searchsorted(zs, mn[2]) - 1, np.searchsorted(zs, mx[2]) + 1
        ix0, iy0, iz0 = max(ix0, 0), max(iy0, 0), max(iz0, 0)
        ix1, iy1, iz1 = min(ix1, XG), min(iy1, YG), min(iz1, ZG)
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue
        gx, gy, gz = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], zs[iz0:iz1], indexing="ij")
        P = np.stack([gx, gy, gz], -1).reshape(-1, 3)                # candidate voxel centers
        local = (P - c) @ R                                          # ego->box (R^T applied via P@R)
        inside = np.all(np.abs(local) <= half, axis=1).reshape(ix1 - ix0, iy1 - iy0, iz1 - iz0)
        grid[ix0:ix1, iy0:iy1, iz0:iz1] |= inside.astype(np.uint8)
        nb += 1
    return grid, nb


# pick the busiest sample by vehicle-voxel count
best = None; bn = -1
for s in nusc.sample:
    g, nb = voxelize(s)
    if g.sum() > bn:
        bn = g.sum(); best = (s, g, nb)
s, grid, nb = best
occ = np.argwhere(grid > 0)
zocc = ZB[0] + (occ[:, 2] + 0.5) * RES
print(f"busiest sample: {nb} vehicles, {int(grid.sum())} occupied voxels", flush=True)
print(f"occupied Z range: [{zocc.min():.2f}, {zocc.max():.2f}] m  (probe grid Z {ZB})", flush=True)
hist = np.bincount(occ[:, 2], minlength=ZG)
print("occupied voxels per Z-slice:", hist.tolist(), flush=True)

# viz: height-colored top-down (color = highest occupied z) + 3 Z-slices
height = np.where(grid.any(2), grid.argmax(2) * 0 + (grid * (np.arange(ZG) + 1)).max(2), 0)
hc = cv2.applyColorMap((height / max(height.max(), 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
hc[height == 0] = (30, 30, 30)
hc = cv2.flip(hc, 0)
ex = ey = XG // 2; cv2.circle(hc, (ey, ex), 3, (255, 255, 255), -1)
cv2.putText(hc, "height-colored top-down", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
slices = []
for zi in [z for z in range(ZG) if hist[z] > 0][:3] or [ZG // 2]:
    sl = np.full((XG, YG, 3), 30, np.uint8); sl[grid[:, :, zi] > 0] = (0, 220, 255)
    sl = cv2.flip(sl, 0); cv2.putText(sl, f"z={zs[zi]:.1f}m", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    slices.append(sl)
row = np.hstack([hc] + [np.full((XG, 6, 3), 255, np.uint8)] + slices)
cv2.imwrite(os.path.join(OUT, "occ_gt_check.png"), row)
print(f"wrote {OUT}/occ_gt_check.png ({row.shape})", flush=True)
