"""Animated pred-vs-GT demo for smart-space: for each held-out frame (temporal order), show a camera
view, the GT floor map, the agent's PREDICTED floor map, and a TP/FN/FP overlay — swept into a gif.
Uses real predictions written by the agent's model (pred_<token>.npy). CPU-only (just rendering).
usage: python viz_smartspace_pred.py <val_npz_dir> <preds_dir> <out.gif> [n] [stride]
"""
import sys, glob, os
import numpy as np
import cv2
import imageio

VAL, PREDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2

files = sorted(glob.glob(os.path.join(VAL, "*.npz")))[::STRIDE][:N]
S = 360  # per-cell render size (2x2 grid)

def mapimg(grid, title, color=cv2.COLORMAP_INFERNO):
    g = (grid.T[::-1] * 255).astype(np.uint8)
    im = cv2.applyColorMap(g, color)
    im = cv2.resize(im, (S, S), interpolation=cv2.INTER_NEAREST)
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
    cv2.putText(cam, "camera 0", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)
    # TP/FN/FP overlay: green=TP, red=FN(missed), blue=FP(false)
    ov = np.zeros((*gt.shape, 3), np.uint8)
    ov[gt & pred] = (0, 255, 0); ov[gt & ~pred] = (0, 0, 255); ov[~gt & pred] = (255, 0, 0)
    ov = cv2.resize(ov.transpose(1, 0, 2)[::-1], (S, S), interpolation=cv2.INTER_NEAREST)
    cv2.putText(ov, f"TP/FN/FP  IoU {iou:.2f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    sep = np.full((4, S, 3), 40, np.uint8); vsep = np.full((2 * S + 4, 4, 3), 40, np.uint8)
    top = np.hstack([cam, np.full((S, 4, 3), 40, np.uint8), mapimg(gt, "GT")])
    bot = np.hstack([mapimg(pred, "agent pred"), np.full((S, 4, 3), 40, np.uint8), ov])
    grid = np.vstack([top, np.full((4, 2 * S + 4, 3), 40, np.uint8), bot])   # 2x2
    frames.append(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

imageio.mimsave(OUT, frames, duration=0.18, loop=0)
print(f"wrote {OUT}: {len(frames)} frames | mean IoU over shown frames = {np.mean(ious):.4f}")
