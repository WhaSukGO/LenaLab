"""Temporal BEV sweep over a HELD-OUT scene: walk the scene's samples in order (via nuScenes
devkit), and for each show the 6 surround cameras + the AGENT's BEV prediction vs GT
(green=TP, red=FN, blue=FP) with a running IoU. Writes an MP4 (convert to GIF after).

Runs in vo-bev:1 (devkit + cv2). usage:
  python viz_bev_gif.py <nuscenes_root> <agent_preds> <bev_cache> <scene_name> <out.mp4>
"""
import sys, os
import numpy as np, cv2
from nuscenes.nuscenes import NuScenes

ROOT, AGENT, CACHE, SCENE, OUT = sys.argv[1:6]
nusc = NuScenes(version="v1.0-mini", dataroot=ROOT, verbose=False)
scene = [s for s in nusc.scene if s["name"] == SCENE][0]

# ordered sample tokens in the scene
order, tok = [], scene["first_sample_token"]
while tok:
    order.append(tok); tok = nusc.get("sample", tok)["next"]


def bev_panel(gt, pred, size):
    XG, YG = gt.shape
    c = np.full((XG, YG, 3), 28, np.uint8)
    c[gt & pred] = (90, 220, 90); c[gt & ~pred] = (70, 70, 220); c[~gt & pred] = (220, 180, 70)
    e = XG // 2
    cv2.circle(c, (e, e), 3, (255, 255, 255), -1)
    cv2.arrowedLine(c, (e, e), (e, e - 14), (255, 255, 255), 1, tipLength=0.4)
    return cv2.resize(cv2.flip(c, 0), (size, size), interpolation=cv2.INTER_NEAREST)


frames = []
for i, t in enumerate(order):
    vz = os.path.join(CACHE, "val", t + ".npz")
    pp = os.path.join(AGENT, f"pred_{t}.npy")
    if not (os.path.exists(vz) and os.path.exists(pp)):
        continue
    d = np.load(vz); gt = d["bev"] > 0.5; pred = np.load(pp).astype(bool)
    inter = (gt & pred).sum(); union = (gt | pred).sum(); iou = inter / union if union else 0.0
    imgs = d["imgs"]
    top = np.hstack([imgs[0], imgs[1], imgs[2]])[:, :, ::-1]
    bot = np.hstack([imgs[3], imgs[4], imgs[5]])[:, :, ::-1]
    cams = np.vstack([top, bot]); s = cams.shape[0]
    pn = bev_panel(gt, pred, s)
    cv2.putText(pn, f"IoU {iou:.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    frame = np.hstack([cams, np.full((s, 8, 3), 255, np.uint8), pn])
    cv2.putText(frame, f"{SCENE}  frame {i+1}/{len(order)}  (held-out)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    frames.append(frame)

H, W = frames[0].shape[:2]
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 4, (W, H))
for f in frames:
    vw.write(f)
vw.release()
print(f"wrote {OUT}: {len(frames)} frames over {SCENE} ({W}x{H})")
