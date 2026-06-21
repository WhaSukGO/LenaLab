"""Warehouse demo: a static camera (with the dataset's agent detections drawn) beside the live
TOP-DOWN floor-occupancy map, animated across the held-out (unseen-time) window. Shows the domain's
job — turn fixed cameras into a moving 'what's where' map of the space. GT-based (illustrates the
task/target); CPU-only. usage: python demo_smartspace.py <scene_dir> <out.gif> [cam] [start] [n] [stride]
"""
import sys, json
import numpy as np
import cv2
import imageio

SCENE, OUT = sys.argv[1], sys.argv[2]
CAM = sys.argv[3] if len(sys.argv) > 3 else "Camera_0000"
START = int(sys.argv[4]) if len(sys.argv) > 4 else 6400     # held-out window (last 30% of 9000)
N = int(sys.argv[5]) if len(sys.argv) > 5 else 60
STRIDE = int(sys.argv[6]) if len(sys.argv) > 6 else 12
RES = 0.5

calib = json.load(open(f"{SCENE}/calibration.json"))
gt = json.load(open(f"{SCENE}/ground_truth.json"))
cam = next(s for s in calib["sensors"] if s["id"] == CAM)
P = np.array(cam["cameraMatrix"], float)

# world grid bounds from all GT agents
allc = np.array([o["3d location"][:2] for fr in list(gt.values())[::200] for o in fr], float)
lo, hi = allc.min(0) - 2, allc.max(0) + 2
XG = int(np.ceil((hi[0] - lo[0]) / RES)); YG = int(np.ceil((hi[1] - lo[1]) / RES))
xs = lo[0] + (np.arange(XG) + 0.5) * RES; ys = lo[1] + (np.arange(YG) + 0.5) * RES

def occupancy(objs):
    g = np.zeros((XG, YG), np.uint8)
    for o in objs:
        cx, cy = o["3d location"][:2]; w, l, _ = o["3d bounding box scale"]; yaw = o["3d bounding box rotation"][2]
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]); half = np.array([w/2, l/2])
        rad = float(np.hypot(w, l)/2 + RES)
        ix0, ix1 = max(int((cx-rad-lo[0])/RES), 0), min(int((cx+rad-lo[0])/RES)+1, XG)
        iy0, iy1 = max(int((cy-rad-lo[1])/RES), 0), min(int((cy+rad-lo[1])/RES)+1, YG)
        if ix0 >= ix1 or iy0 >= iy1: continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
        loc = (np.stack([gx, gy], -1).reshape(-1, 2) - [cx, cy]) @ R
        g[ix0:ix1, iy0:iy1] |= np.all(np.abs(loc) <= half, 1).reshape(ix1-ix0, iy1-iy0).astype(np.uint8)
    return g

cap = cv2.VideoCapture(f"{SCENE}/videos/{CAM}.mp4")
frames_out = []
for k in range(N):
    fi = START + k * STRIDE
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok or str(fi) not in gt: break
    objs = gt[str(fi)]
    # left: camera with the dataset's detections for THIS cam (boxes the cam sees)
    ndet = 0
    for o in objs:
        vb = o.get("2d bounding box visible") or {}
        box = vb.get(CAM)
        if not box: continue
        ndet += 1
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
        cv2.putText(frame, o["object type"][:6], (x1, max(y1-4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)
    cv2.putText(frame, f"{CAM}  detections: {ndet}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)
    cam_img = cv2.resize(frame, (480, 270))
    # right: live top-down floor-occupancy map (ALL agents)
    occ = (occupancy(objs).T[::-1] * 255).astype(np.uint8)
    occ = cv2.applyColorMap(occ, cv2.COLORMAP_INFERNO)
    occ = cv2.resize(occ, (270, 270), interpolation=cv2.INTER_NEAREST)
    cv2.putText(occ, f"floor map ({len(objs)} agents)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    panel = np.hstack([cam_img, np.full((270, 6, 3), 30, np.uint8), occ])
    frames_out.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

imageio.mimsave(OUT, frames_out, duration=0.12, loop=0)
print(f"wrote {OUT}: {len(frames_out)} frames | grid {XG}x{YG} | frames {START}..{START+N*STRIDE}")
