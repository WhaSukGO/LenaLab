"""Visualize 3D occupancy predictions vs GT on held-out scenes. Architecture-agnostic: reads saved
pred_<token>.npy (XG,YG,ZG) + cams/GT from the occ cache. Per sample, a row of:
  [6 surround cams] | [GT height-colored top-down] | [Pred height-colored] | [TP/FN/FP collapsed]
height color = highest occupied z (jet); collapse = max over z. Runs with cv2 (vo-bev:1).

usage: python viz_occ_pred.py <preds_dir> <occ_cache> <out.png> [n]
"""
import sys, os, glob
import numpy as np, cv2

PRED, CACHE, OUTP = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
preds = {os.path.basename(p)[5:-4]: p for p in glob.glob(os.path.join(PRED, "pred_*.npy"))}
rows_data = []
for tok, pp in preds.items():
    vz = os.path.join(CACHE, "val", tok + ".npz")
    if not os.path.exists(vz):
        continue
    d = np.load(vz); gt = d["occ"] > 0.5; pred = np.load(pp).astype(bool)
    inter = (gt & pred).sum(); union = (gt | pred).sum(); iou = inter / union if union else 0.0
    rows_data.append((iou, tok, d["imgs"], gt, pred))
rows_data.sort(key=lambda r: -r[0])
pick = [rows_data[0], rows_data[len(rows_data) // 2], rows_data[-1]][:N] if rows_data else []
ZG = pick[0][3].shape[2] if pick else 12


def height_top(vox, size):                       # height-colored top-down (max occupied z per column)
    XG, YG, _ = vox.shape
    h = (vox * (np.arange(ZG) + 1)).max(2)
    c = cv2.applyColorMap((h / max(h.max(), 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    c[h == 0] = (30, 30, 30)
    return cv2.resize(cv2.flip(c, 0), (size, size), interpolation=cv2.INTER_NEAREST)


def tpfnfp_top(gt, pred, size):                  # collapse z: column-wise TP/FN/FP on any-occupied
    g = gt.any(2); p = pred.any(2)
    XG, YG = g.shape
    c = np.full((XG, YG, 3), 28, np.uint8)
    c[g & p] = (90, 220, 90); c[g & ~p] = (70, 70, 220); c[~g & p] = (220, 180, 70)
    e = XG // 2; cv2.circle(c, (e, e), 3, (255, 255, 255), -1)
    return cv2.resize(cv2.flip(c, 0), (size, size), interpolation=cv2.INTER_NEAREST)


rows = []
for iou, tok, imgs, gt, pred in pick:
    top = np.hstack([imgs[0], imgs[1], imgs[2]])[:, :, ::-1]
    bot = np.hstack([imgs[3], imgs[4], imgs[5]])[:, :, ::-1]
    cams = np.vstack([top, bot]); s = cams.shape[0]; sep = np.full((s, 6, 3), 255, np.uint8)
    g = height_top(gt, s); cv2.putText(g, "GT", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    pr = height_top(pred, s); cv2.putText(pr, "Pred", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    ov = tpfnfp_top(gt, pred, s); cv2.putText(ov, f"IoU {iou:.2f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.hstack([cams, sep, g, sep, pr, sep, ov]))
legend = np.full((30, rows[0].shape[1], 3), 255, np.uint8)
cv2.putText(legend, "height-colored top-down (max-z) | TP=green FN=red FP=blue (z-collapsed)",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
canvas = legend
for r in rows:
    canvas = np.vstack([canvas, np.full((8, r.shape[1], 3), 255, np.uint8), r])
cv2.imwrite(OUTP, canvas)
print(f"wrote {OUTP} ({canvas.shape}) from {len(preds)} predictions")
