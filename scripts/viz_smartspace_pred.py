"""Animated pred-vs-GT demo for smart-space (2x2): camera 0 | GT floor map | agent PRED map | TP/FN/FP.
Aids for reading the map:
  - camera 0's floor footprint drawn as a semi-transparent green region + a viewing-direction ARROW;
  - every agent CAMERA 0 CAN SEE gets a unique color: a box on the video AND a matching dot on the maps,
    so you can tell which blob corresponds to which object (instances from the raw ground_truth.json).
Real predictions (pred_<token>.npy). CPU-only.
usage: python viz_smartspace_pred.py <scene_dir> <val_npz_dir> <preds_dir> <out.gif> [n] [stride] [dur]
"""
import sys, glob, os, json, colorsys
import numpy as np
import cv2
import imageio

SCENE, VAL, PREDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 36
STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 2
DUR = float(sys.argv[7]) if len(sys.argv) > 7 else 0.18       # original (faster) speed
OW, OH = 1920, 1080          # original video resolution (GT 2d boxes are in these coords)
IMG_W, IMG_H = 352, 128      # resolution cam_proj projects into
S = 360                      # per-cell render size
CAM = "Camera_0000"

gt_json = json.load(open(os.path.join(SCENE, "ground_truth.json")))
files = sorted(glob.glob(os.path.join(VAL, "*.npz")))[::STRIDE][:N]

# --- geometry: camera 0 footprint (region) + viewing arrow (constant) ---
ref = np.load(files[0]); XG, YG = ref["bev"].shape
gb = ref["grid_bounds"]; RES = (gb[1] - gb[0]) / XG
P0 = ref["cam_proj"][0].astype(float)
xs = gb[0] + (np.arange(XG) + 0.5) * RES; ys = gb[2] + (np.arange(YG) + 0.5) * RES
GX, GY = np.meshgrid(xs, ys, indexing="ij")
uvw = np.stack([GX, GY, np.zeros_like(GX), np.ones_like(GX)], -1) @ P0.T
zc = uvw[..., 2]; u = uvw[..., 0] / np.clip(zc, 1e-6, None); v = uvw[..., 1] / np.clip(zc, 1e-6, None)
seen = (zc > 0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)

def world_px(X, Y):                       # world (m) -> rendered S×S map pixel
    ix = (X - gb[0]) / RES; iy = (Y - gb[2]) / RES
    return int(ix * S / XG), int((YG - 1 - iy) * S / YG)

ij = np.argwhere(seen)
hull = cv2.convexHull(np.stack([(ij[:, 0]*S/XG).astype(np.int32),
                                ((YG-1-ij[:, 1])*S/YG).astype(np.int32)], 1)) if len(ij) else None
# camera centre = right null space of P0  -> viewing arrow toward footprint centroid
_, _, Vt = np.linalg.svd(P0); C = Vt[-1]; C = C[:3] / C[3]
cam_px = np.array(world_px(C[0], C[1]), float)
cen = np.array(world_px(xs[ij[:, 0]].mean(), ys[ij[:, 1]].mean()), float)
dvec = cen - cam_px; dvec = dvec / (np.linalg.norm(dvec) + 1e-9)
a_tail = (cen - dvec * 0.22 * S).astype(int); a_head = (cen + dvec * 0.20 * S).astype(int)

def color(oid):
    h = (int(oid) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))     # BGR

def decorate(im, draw_dots=None):
    if hull is not None:
        ov = im.copy(); cv2.fillConvexPoly(ov, hull, (0, 150, 0)); im = cv2.addWeighted(ov, 0.22, im, 0.78, 0)
        cv2.polylines(im, [hull], True, (60, 220, 60), 2)
    cv2.arrowedLine(im, tuple(a_tail), tuple(a_head), (60, 255, 60), 3, tipLength=0.3)
    if draw_dots:
        for (X, Y, col) in draw_dots:
            p = world_px(X, Y); cv2.circle(im, p, 7, col, -1); cv2.circle(im, p, 7, (255, 255, 255), 1)
    return im

def mapimg(grid, title, dots):
    g = (grid.T[::-1] * 255).astype(np.uint8)
    im = cv2.resize(cv2.applyColorMap(g, cv2.COLORMAP_BONE), (S, S), interpolation=cv2.INTER_NEAREST)
    im = decorate(im, dots)
    cv2.putText(im, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im

frames, ious = [], []
for f in files:
    tok = os.path.splitext(os.path.basename(f))[0]; fidx = str(int(tok[1:]))
    d = np.load(f); gt = d["bev"].astype(bool)
    pp = os.path.join(PREDS, f"pred_{tok}.npy")
    if not os.path.exists(pp) or fidx not in gt_json:
        continue
    pred = np.load(pp).astype(bool); union = (pred | gt).sum()
    iou = (pred & gt).sum() / union if union else 1.0; ious.append(iou)
    # instances camera 0 sees: colored box on video + matching dot on maps
    dots = []
    cam = cv2.resize(cv2.cvtColor(d["imgs"][0], cv2.COLOR_RGB2BGR), (S, S))
    for o in gt_json[fidx]:
        box = (o.get("2d bounding box visible") or {}).get(CAM)
        if not box:
            continue
        col = color(o["object id"])
        x1, y1, x2, y2 = box
        cv2.rectangle(cam, (int(x1*S/OW), int(y1*S/OH)), (int(x2*S/OW), int(y2*S/OH)), col, 2)
        X, Y = o["3d location"][:2]; dots.append((X, Y, col))
    cv2.putText(cam, f"camera 0  ({len(dots)} agents in view)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2)
    # TP/FN/FP as per-object circles: green = correctly detected, red = missed (FN) or false (FP)
    ov = decorate(np.zeros((S, S, 3), np.uint8))
    def gpx(ix, iy):
        return (int(ix * S / XG), int((YG - 1 - iy) * S / YG))
    ng, lab = cv2.connectedComponents(gt.astype(np.uint8))      # one circle per GT object (>=2 cells)
    for k in range(1, ng):
        m = lab == k; idx = np.argwhere(m)
        if len(idx) < 2:
            continue
        c = gpx(idx[:, 0].mean(), idx[:, 1].mean())
        cv2.circle(ov, c, 9, (0, 255, 0) if (pred & m).any() else (0, 0, 255), 2)
    npd, labp = cv2.connectedComponents(pred.astype(np.uint8))  # false predictions (no GT overlap)
    for k in range(1, npd):
        m = labp == k; idx = np.argwhere(m)
        if len(idx) >= 2 and not (gt & m).any():
            cv2.circle(ov, gpx(idx[:, 0].mean(), idx[:, 1].mean()), 9, (0, 0, 255), 2)
    cv2.putText(ov, f"green=hit  red=miss/false  IoU {iou:.2f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    sep = np.full((S, 4, 3), 40, np.uint8)
    top = np.hstack([cam, sep, mapimg(gt, "GT  (dots = same agents)", dots)])
    bot = np.hstack([mapimg(pred, "agent pred", dots), sep, ov])
    grid = np.vstack([top, np.full((4, 2 * S + 4, 3), 40, np.uint8), bot])
    cv2.putText(grid, "green region+arrow = camera 0 field of view & facing", (8, 2 * S - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 1)
    frames.append(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

imageio.mimsave(OUT, frames, duration=DUR, loop=0)
print(f"wrote {OUT}: {len(frames)} frames @ {DUR}s | cam0 sees {int(seen.sum())} cells | mean IoU {np.mean(ious):.4f}")
