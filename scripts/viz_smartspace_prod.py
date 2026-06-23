"""Render a working-model demo for the PRODUCTION per-space floor-occupancy model (W000).
Per frame, shows: a camera view | ground-truth floor map | model probability | TP/FN/FP overlay,
with the live IoU. CPU-only (matplotlib Agg + imageio) — no GPU.

usage: python viz_smartspace_prod.py <val_npz_dir> <preds_dir> <out.gif> [thr=0.8] [dur_ms=500]
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

VAL, PREDS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
DUR = float(sys.argv[5]) if len(sys.argv) > 5 else 500.0   # milliseconds per frame


def dilate(mask, r=2):                         # display-only: grow sparse cells into visible blobs
    m = mask.copy()
    for _ in range(r):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]; g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    return m

preds = sorted(glob.glob(os.path.join(PREDS, "*.npy")))
frames = []
ious = []
for pf in preds:
    tok = os.path.splitext(os.path.basename(pf))[0]
    npz = os.path.join(VAL, tok + ".npz")
    if not os.path.exists(npz):
        continue
    d = np.load(npz)
    prob = np.load(pf)                       # (XG,YG) probability
    gt = d["bev"].astype(bool)               # (XG,YG)
    pred = prob > THR
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    iou = inter / union if union else 0.0
    ious.append(iou)
    cam0 = d["imgs"][0]                       # (H,W,3) uint8

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Production per-space model — Warehouse_000  |  frame {tok}  |  IoU = {iou:.3f}",
                 fontsize=13, fontweight="bold")
    ax[0, 0].imshow(cam0); ax[0, 0].set_title("Camera 0 (one of 19 static cams)", fontsize=10)
    gtd = dilate(gt).T
    ax[0, 1].imshow(np.ones_like(gtd, float), origin="lower", cmap="Greys", vmin=0, vmax=1)
    ax[0, 1].imshow(np.ma.masked_where(~gtd, gtd), origin="lower", cmap="Greens", vmin=0, vmax=1.5)
    ax[0, 1].set_title("Ground-truth floor occupancy", fontsize=10)
    ax[1, 0].imshow(prob.T, origin="lower", cmap="magma", vmin=0, vmax=1)
    ax[1, 0].set_title(f"Model probability (threshold {THR})", fontsize=10)
    # TP/FN/FP overlay (dilated for visibility)
    ov = np.zeros((*gt.T.shape, 3))
    tp = dilate(pred & gt).T; fn = dilate(gt & ~pred).T; fp = dilate(pred & ~gt).T
    ov[fp] = [1.0, 0.6, 0.0]      # orange = false alarm
    ov[fn] = [0.9, 0.1, 0.1]      # red    = missed
    ov[tp] = [0.1, 0.85, 0.1]     # green  = correct (drawn last)
    ax[1, 1].imshow(ov, origin="lower")
    ax[1, 1].set_title("TP (green) · FN (red) · FP (orange)", fontsize=10)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())
    plt.close(fig)

imageio.mimsave(OUT, frames, duration=DUR, loop=0)
print(f"wrote {OUT}: {len(frames)} frames @ {DUR:.0f}ms | mean IoU {np.mean(ious):.3f}")
