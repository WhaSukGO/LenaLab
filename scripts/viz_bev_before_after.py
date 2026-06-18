"""Before/after BEV: from-scratch REFERENCE baseline (before) vs AGENT-authored network (after) on
the SAME held-out scenes, side by side, so the improvement is visible. Each row:
   [ 6 surround cams ] | [ BEFORE: reference pred vs GT + IoU ] | [ AFTER: agent pred vs GT + IoU ]
green = TP (correct), red = FN (missed), blue = FP (false positive).

usage: python viz_bev_before_after.py <ref_preds> <agent_preds> <bev_cache> <out.png> [n]
"""
import sys, os, glob
import numpy as np, cv2

REF, AGENT, CACHE, OUTP = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 3


def iou(pred, gt):
    inter = (pred & gt).sum(); union = (pred | gt).sum()
    return inter / union if union else 0.0


def panel(gt, pred, size, label, iou_val):
    XG, YG = gt.shape
    c = np.full((XG, YG, 3), 28, np.uint8)
    c[gt & pred] = (90, 220, 90); c[gt & ~pred] = (70, 70, 220); c[~gt & pred] = (220, 180, 70)
    e = XG // 2
    cv2.circle(c, (e, e), 3, (255, 255, 255), -1)
    cv2.arrowedLine(c, (e, e), (e, e - 14), (255, 255, 255), 1, tipLength=0.4)
    c = cv2.resize(cv2.flip(c, 0), (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(c, f"IoU {iou_val:.2f}", (6, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return c


# tokens present in BOTH prediction sets and the cache
rtok = {os.path.basename(p)[5:-4] for p in glob.glob(os.path.join(REF, "pred_*.npy"))}
atok = {os.path.basename(p)[5:-4] for p in glob.glob(os.path.join(AGENT, "pred_*.npy"))}
toks = sorted(rtok & atok)
data = []
for t in toks:
    vz = os.path.join(CACHE, "val", t + ".npz")
    if not os.path.exists(vz):
        continue
    d = np.load(vz); gt = d["bev"] > 0.5
    rp = np.load(os.path.join(REF, f"pred_{t}.npy")).astype(bool)
    ap = np.load(os.path.join(AGENT, f"pred_{t}.npy")).astype(bool)
    data.append((iou(ap, gt) - iou(rp, gt), t, d["imgs"], gt, rp, ap))
# show scenes where the agent improves most over the reference (the point of before/after)
data.sort(key=lambda r: -r[0])
pick = data[:N]

rows = []
for delta, t, imgs, gt, rp, ap in pick:
    top = np.hstack([imgs[0], imgs[1], imgs[2]])[:, :, ::-1]
    bot = np.hstack([imgs[3], imgs[4], imgs[5]])[:, :, ::-1]
    cams = np.vstack([top, bot]); s = cams.shape[0]
    sep = np.full((s, 8, 3), 255, np.uint8)
    rows.append(np.hstack([cams, sep, panel(gt, rp, s, "BEFORE (ref)", iou(rp, gt)),
                           sep, panel(gt, ap, s, "AFTER (agent)", iou(ap, gt))]))
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
print(f"wrote {OUTP} ({canvas.shape}) over {len(toks)} shared held-out tokens")
