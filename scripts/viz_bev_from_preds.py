"""Visualize an AGENT's BEV output: read saved pred_<token>.npy masks + join the surround cameras
and GT from the prepped cache -> TP/FN/FP panels on held-out scenes. Architecture-agnostic (reads
only the predicted masks, so it works for whatever network the agent authored).

usage: python viz_bev_from_preds.py <preds_dir> <bev_cache> <out.png> [n_samples]
  <preds_dir>  dir of pred_<token>.npy (the agent's $LAB_ARTIFACTS)
  <bev_cache>  ~/.cache/vo_lab/bev  (has val/<token>.npz with imgs + bev)
"""
import sys, os, glob
import numpy as np, cv2

PRED, CACHE, OUTP = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 3
preds = {os.path.basename(p)[len("pred_"):-4]: p for p in glob.glob(os.path.join(PRED, "pred_*.npy"))}
# rank held-out samples by IoU achieved (show a spread: best, median, worst)
rows_data = []
for tok, pp in preds.items():
    vz = os.path.join(CACHE, "val", tok + ".npz")
    if not os.path.exists(vz):
        continue
    d = np.load(vz); gt = d["bev"] > 0.5; pred = np.load(pp).astype(bool)
    inter = (gt & pred).sum(); union = (gt | pred).sum()
    iou = inter / union if union else 0.0
    rows_data.append((iou, tok, d["imgs"], gt, pred))
rows_data.sort(key=lambda r: -r[0])
pick = []
if rows_data:
    pick = [rows_data[0], rows_data[len(rows_data) // 2], rows_data[-1]][:N]


def panel(gt, pred, size):
    XG, YG = gt.shape
    c = np.full((XG, YG, 3), 28, np.uint8)
    tp, fn, fp = gt & pred, gt & ~pred, ~gt & pred
    c[tp] = (90, 220, 90); c[fn] = (70, 70, 220); c[fp] = (220, 180, 70)
    e = XG // 2
    cv2.circle(c, (e, e), 3, (255, 255, 255), -1)
    cv2.arrowedLine(c, (e, e), (e, e - 14), (255, 255, 255), 1, tipLength=0.4)
    return cv2.resize(cv2.flip(c, 0), (size, size), interpolation=cv2.INTER_NEAREST)


rows = []
for iou, tok, imgs, gt, pred in pick:
    top = np.hstack([imgs[0], imgs[1], imgs[2]])[:, :, ::-1]
    bot = np.hstack([imgs[3], imgs[4], imgs[5]])[:, :, ::-1]
    cams = np.vstack([top, bot])
    pn = panel(gt, pred, cams.shape[0])
    cv2.putText(pn, f"IoU {iou:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.hstack([cams, np.full((cams.shape[0], 8, 3), 255, np.uint8), pn]))
legend = np.full((34, rows[0].shape[1], 3), 255, np.uint8)
for i, (col, txt) in enumerate([((90, 220, 90), "TP (correct)"), ((70, 70, 220), "FN (missed)"),
                                ((220, 180, 70), "FP (false)")]):
    x0 = 20 + i * 220
    cv2.rectangle(legend, (x0, 10), (x0 + 22, 26), col, -1)
    cv2.putText(legend, txt, (x0 + 28, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
canvas = legend
for r in rows:
    canvas = np.vstack([canvas, np.full((10, r.shape[1], 3), 255, np.uint8), r])
cv2.imwrite(OUTP, canvas)
print(f"wrote {OUTP}  ({canvas.shape}) from {len(preds)} agent predictions")
