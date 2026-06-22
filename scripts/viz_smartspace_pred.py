"""Animated pred-vs-GT demo for smart-space: for each held-out frame (temporal order), show a camera
view, the GT floor map, the agent's PREDICTED floor map, and a TP/FN/FP overlay — swept into a gif (2x2).
A semi-transparent green region on the maps marks **camera 0's floor footprint** (its frustum ∩ ground
plane) so you can see which floor area that camera covers. Real predictions (pred_<token>.npy). CPU-only.
usage: python viz_smartspace_pred.py <val_npz_dir> <preds_dir> <out.gif> [n] [stride]
"""
import sys, glob, os
import numpy as np
import cv2
import imageio

VAL, PREDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
IMG_W, IMG_H = 352, 128          # the resolution cam_proj projects into
S = 360                          # per-cell render size (2x2 grid)

files = sorted(glob.glob(os.path.join(VAL, "*.npz")))[::STRIDE][:N]

# --- camera 0 floor footprint (constant: static camera + fixed grid) ---
ref = np.load(files[0])
XG, YG = ref["bev"].shape
gb = ref["grid_bounds"]; RES = (gb[1] - gb[0]) / XG
P0 = ref["cam_proj"][0].astype(float)              # camera 0 world->image (scaled)
xs = gb[0] + (np.arange(XG) + 0.5) * RES
ys = gb[2] + (np.arange(YG) + 0.5) * RES
GX, GY = np.meshgrid(xs, ys, indexing="ij")        # (XG,YG) world coords of each cell, z=0
pts = np.stack([GX, GY, np.zeros_like(GX), np.ones_like(GX)], -1)   # (XG,YG,4)
uvw = pts @ P0.T                                   # (XG,YG,3)
zc = uvw[..., 2]
u = uvw[..., 0] / np.clip(zc, 1e-6, None); v = uvw[..., 1] / np.clip(zc, 1e-6, None)
seen = (zc > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)   # cells camera 0 sees
# map grid cell (ix,iy) -> rendered S×S pixel (matches mapimg: grid.T[::-1] then resize)
ij = np.argwhere(seen)
if len(ij):
    px = (ij[:, 0] * S / XG).astype(np.int32)
    py = ((YG - 1 - ij[:, 1]) * S / YG).astype(np.int32)
    hull = cv2.convexHull(np.stack([px, py], 1))
else:
    hull = None

def draw_fov(img):
    if hull is None:
        return img
    ov = img.copy(); cv2.fillConvexPoly(ov, hull, (0, 170, 0))
    img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)
    cv2.polylines(img, [hull], True, (60, 255, 60), 2)
    return img

def mapimg(grid, title, fov=True, color=cv2.COLORMAP_INFERNO):
    g = (grid.T[::-1] * 255).astype(np.uint8)
    im = cv2.applyColorMap(g, color)
    im = cv2.resize(im, (S, S), interpolation=cv2.INTER_NEAREST)
    if fov:
        im = draw_fov(im)
    cv2.putText(im, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im

frames, ious = [], []
for f in files:
    tok = os.path.splitext(os.path.basename(f))[0]
    d = np.load(f); gt = d["bev"].astype(bool)
    pp = os.path.join(PREDS, f"pred_{tok}.npy")
    if not os.path.exists(pp):
        continue
    pred = np.load(pp).astype(bool)
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    iou = inter / union if union else 1.0; ious.append(iou)
    cam = cv2.resize(cv2.cvtColor(d["imgs"][0], cv2.COLOR_RGB2BGR), (S, S))
    cv2.putText(cam, "camera 0", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 2)
    ov = np.zeros((*gt.shape, 3), np.uint8)        # TP/FN/FP: green=TP, red=FN, blue=FP
    ov[gt & pred] = (0, 255, 0); ov[gt & ~pred] = (0, 0, 255); ov[~gt & pred] = (255, 0, 0)
    ov = cv2.resize(ov.transpose(1, 0, 2)[::-1], (S, S), interpolation=cv2.INTER_NEAREST)
    ov = draw_fov(ov)
    cv2.putText(ov, f"TP/FN/FP  IoU {iou:.2f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    sep_h = np.full((S, 4, 3), 40, np.uint8)
    top = np.hstack([cam, sep_h, mapimg(gt, "GT")])
    bot = np.hstack([mapimg(pred, "agent pred"), sep_h, ov])
    grid = np.vstack([top, np.full((4, 2 * S + 4, 3), 40, np.uint8), bot])   # 2x2
    cv2.putText(grid, "green = camera 0 field of view", (8, 2 * S - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
    frames.append(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

imageio.mimsave(OUT, frames, duration=0.18, loop=0)
print(f"wrote {OUT}: {len(frames)} frames | cam0 sees {int(seen.sum())}/{XG*YG} cells | "
      f"mean IoU over shown = {np.mean(ious):.4f}")
