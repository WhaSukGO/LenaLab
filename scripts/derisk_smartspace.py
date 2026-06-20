"""Phase-0 de-risk for the smart-space occupancy domain (NVIDIA Physical AI Smart Spaces).
Verifies the three unknowns before we build the adapter:
  1. calibration projects GT 3D boxes onto the real camera image (cameraMatrix world->image),
  2. we can rasterize a clean 2D BEV floor-occupancy GT from the world 3D boxes,
  3. frame indexing + the world frame line up (camera frame <-> GT frame <-> map.png).
Saves a 3-panel figure. usage: python derisk_smartspace.py <scene_dir> <out_png>
"""
import sys, json
import numpy as np
import cv2

SCENE, OUT = sys.argv[1], sys.argv[2]
FRAME = 0                                              # de-risk on frame 0

calib = json.load(open(f"{SCENE}/calibration.json"))
gt = json.load(open(f"{SCENE}/ground_truth.json"))
cam = next(s for s in calib["sensors"] if s["id"] == "Camera_0000")
P = np.array(cam["cameraMatrix"], float)              # 3x4 world->image
objs = gt[str(FRAME)]
centers = np.array([o["3d location"] for o in objs], float)        # (N,3) world meters

# ---- 1. project GT centers into the camera image (try direct world frame) ----
homc = np.c_[centers, np.ones(len(centers))]
uvw = (P @ homc.T).T
infront = uvw[:, 2] > 0
px = uvw[:, :2] / uvw[:, 2:3]
cap = cv2.VideoCapture(f"{SCENE}/videos/Camera_0000.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ok, frame = cap.read()
Hh, Ww = frame.shape[:2]
on_img = 0
for (u, v), f, o in zip(px, infront, objs):
    if f and 0 <= u < Ww and 0 <= v < Hh:
        on_img += 1
        cv2.circle(frame, (int(u), int(v)), 7, (0, 0, 255), -1)
        cv2.putText(frame, o["object type"][:4], (int(u) + 6, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
print(f"[proj] {len(objs)} objs | in-front {int(infront.sum())} | landed on image {on_img}")

# ---- 2. world bounds (over a sweep of frames) + BEV floor-occupancy raster ----
allc = np.array([o["3d location"][:2]
                 for fr in list(gt.values())[::500] for o in fr], float)
lo, hi = allc.min(0) - 2, allc.max(0) + 2
RES = 0.2
XG = int(np.ceil((hi[0] - lo[0]) / RES)); YG = int(np.ceil((hi[1] - lo[1]) / RES))
print(f"[grid] world X[{lo[0]:.1f},{hi[0]:.1f}] Y[{lo[1]:.1f},{hi[1]:.1f}] -> {XG}x{YG} @ {RES}m")
bev = np.zeros((XG, YG), np.uint8)
xs = lo[0] + (np.arange(XG) + 0.5) * RES
ys = lo[1] + (np.arange(YG) + 0.5) * RES
for o in objs:
    cx, cy = o["3d location"][:2]
    w, l, _ = o["3d bounding box scale"]; yaw = o["3d bounding box rotation"][2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    half = np.array([w / 2, l / 2])
    rad = float(np.hypot(w, l) / 2 + RES)
    ix0, ix1 = max(int((cx - rad - lo[0]) / RES), 0), min(int((cx + rad - lo[0]) / RES) + 1, XG)
    iy0, iy1 = max(int((cy - rad - lo[1]) / RES), 0), min(int((cy + rad - lo[1]) / RES) + 1, YG)
    if ix0 >= ix1 or iy0 >= iy1:
        continue
    gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
    local = (np.stack([gx, gy], -1).reshape(-1, 2) - [cx, cy]) @ R
    inside = np.all(np.abs(local) <= half, 1).reshape(ix1 - ix0, iy1 - iy0)
    bev[ix0:ix1, iy0:iy1] |= inside.astype(np.uint8)
print(f"[bev] occupied cells {int(bev.sum())} / {XG*YG} ({100*bev.mean():.2f}%)")

# ---- 3. three-panel figure: camera+proj | BEV | map.png ----
bev_img = cv2.applyColorMap((bev.T[::-1] * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
mp = cv2.imread(f"{SCENE}/map.png")
def fit(im, h=540):
    return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
panel = np.hstack([fit(frame), fit(bev_img), fit(mp)])
cv2.imwrite(OUT, panel)
print(f"[fig] wrote {OUT}  (camera+projected GT | BEV occupancy | map.png)")
