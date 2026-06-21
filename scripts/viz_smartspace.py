"""Visualize the smart-space domain: 'a map, not a camera'. For one held-out frame, show a couple of
the raw camera views beside the top-down floor-occupancy GT (the harness-owned target the agent is
graded against). No model needed — illustrates the input->target the agent learns.
usage: python viz_smartspace.py <smartspace_occ_root> <out.png> [frame_idx]
"""
import sys, glob
import numpy as np
import cv2

ROOT, OUT = sys.argv[1], sys.argv[2]
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])

f = sorted(glob.glob(f"{ROOT}/val/*.npz"))[IDX]
d = np.load(f)
imgs = d["imgs"]                       # (N,128,352,3) uint8
bev = d["bev"]                         # (XG,YG) uint8

def cam(i):
    return cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)

# top-down occupancy: white agents on dark floor, oriented like a map (y up)
occ = (bev.T[::-1] * 255).astype(np.uint8)
occ = cv2.applyColorMap(occ, cv2.COLORMAP_BONE)
occ = cv2.resize(occ, (256, 256), interpolation=cv2.INTER_NEAREST)
cv2.putText(occ, "floor occupancy (GT)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def fit(im, h=256):
    return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))

# pick 3 spread-out cameras + the map
panel = np.hstack([fit(cam(0)), fit(cam(9)), fit(cam(15)), occ])
cv2.imwrite(OUT, panel)
print(f"wrote {OUT}: cams 0/9/15 | floor-occupancy GT ({int(bev.sum())} cells) | frame {IDX}")
