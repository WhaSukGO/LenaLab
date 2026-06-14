"""Visualize the BEV model on HELD-OUT scenes: surround cameras (ring) + BEV panel showing
ground-truth vs prediction (green=correct/TP, red=missed/FN, blue=false-positive/FP).

usage: python viz_bev_pred.py <solver_model.py> <ckpt.pt> <data_root> <out.png> [n_samples]
"""
import sys, os, glob, json, importlib.util
import numpy as np, cv2, torch

SOLVER, CKPT, ROOT, OUTP = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 3
dev = "cuda" if torch.cuda.is_available() else "cpu"
spec = importlib.util.spec_from_file_location("solver", SOLVER)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
model = mod.build_model().to(dev).eval(); model.load_state_dict(torch.load(CKPT, map_location=dev))
meta = json.load(open(os.path.join(ROOT, "meta.json")))
XG, YG = meta["grid"]; mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
files = sorted(glob.glob(os.path.join(ROOT, "val", "*.npz")))
# pick the busiest val samples (most informative)
files = sorted(files, key=lambda f: -int(np.load(f)["bev"].sum()))[:N]


def bev_panel(gt, pred, size):
    c = np.full((XG, YG, 3), 28, np.uint8)
    tp = gt & pred; fn = gt & ~pred; fp = ~gt & pred
    c[tp] = (90, 220, 90); c[fn] = (70, 70, 220); c[fp] = (220, 180, 70)   # BGR
    ex = ey = XG // 2
    cv2.circle(c, (ey, ex), 3, (255, 255, 255), -1)
    cv2.arrowedLine(c, (ey, ex), (ey, ex - 14), (255, 255, 255), 1, tipLength=0.4)
    c = cv2.flip(c, 0)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_NEAREST)


rows = []
with torch.no_grad():
    for f in files:
        d = np.load(f)
        imgs_u8 = d["imgs"]
        x = torch.from_numpy(imgs_u8).float().permute(0, 3, 1, 2) / 255.0
        x = ((x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)).unsqueeze(0).float().to(dev)
        K = torch.from_numpy(d["intrins"]).float().unsqueeze(0).to(dev)
        c2e = torch.from_numpy(d["cam2ego"]).float().unsqueeze(0).to(dev)
        logits = model(x, K, c2e)[0].cpu().numpy()
        gt = d["bev"] > 0.5; pred = logits > 0.0
        inter = (gt & pred).sum(); union = (gt | pred).sum(); iou = inter / union if union else 0
        H, W = imgs_u8.shape[1:3]
        top = np.hstack([imgs_u8[0], imgs_u8[1], imgs_u8[2]])[:, :, ::-1]   # FL F FR -> BGR
        bot = np.hstack([imgs_u8[3], imgs_u8[4], imgs_u8[5]])[:, :, ::-1]   # BL B BR
        cams = np.vstack([top, bot])
        panel = bev_panel(gt, pred, cams.shape[0])
        cv2.putText(panel, f"IoU {iou:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.hstack([cams, np.full((cams.shape[0], 8, 3), 255, np.uint8), panel]))
sep = np.full((10, rows[0].shape[1], 3), 255, np.uint8)
canvas = rows[0]
for r in rows[1:]:
    canvas = np.vstack([canvas, sep, r])
# legend strip
legend = np.full((34, canvas.shape[1], 3), 255, np.uint8)
for i, (col, txt) in enumerate([((90, 220, 90), "TP (correct)"), ((70, 70, 220), "FN (missed)"),
                                ((220, 180, 70), "FP (false)")]):
    x0 = 20 + i * 220
    cv2.rectangle(legend, (x0, 10), (x0 + 22, 26), col, -1)
    cv2.putText(legend, txt, (x0 + 28, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite(OUTP, np.vstack([legend, canvas]))
print(f"wrote {OUTP}  ({canvas.shape}) over held-out {meta['val_scenes']}")
