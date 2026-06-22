"""Cross-space adapter: prep ONE warehouse scene -> per-frame npz for CROSS-SCENE training (train on
several warehouses, test on an unseen one). Camera-count-agnostic: pad to MAXCAMS with a validity mask
so scenes with 11/16/19 cameras share one tensor shape. Canonical grid: fixed XG×YG centred on the
scene's agent centroid (per-scene grid_bounds), so one model works on any warehouse's world frame.

  imgs       (MAXCAMS,H,W,3) uint8   padded camera views (zeros where invalid)
  cam_proj   (MAXCAMS,3,4)  float32  scaled world->image per camera (zeros where invalid)
  cam_valid  (MAXCAMS,)     float32  1 = real camera, 0 = padding
  grid_bounds (4,)          float32  [x0,x1,y0,y1] canonical floor extent for THIS scene
  bev        (XG,YG)        uint8    floor occupancy GT

usage: python prep_smartspace_xspace.py <scene_dir> <out_dir/SceneName> [stride=30] [maxcams=19]
"""
import sys, os, json
import numpy as np
import cv2

SCENE, OUT = sys.argv[1], sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 30
MAXCAMS = int(sys.argv[4]) if len(sys.argv) > 4 else 19
H, W = 128, 352
RES = 0.5
XG = YG = 224                                  # canonical grid (112m x 112m @ 0.5m), fixed across scenes


def cam_proj(sensor):
    P = np.array(sensor["cameraMatrix"], float)
    S = np.diag([W / 1920.0, H / 1080.0, 1.0])
    return (S @ P).astype(np.float32)


def rasterize(objs, lo):
    grid = np.zeros((XG, YG), np.uint8)
    xs = lo[0] + (np.arange(XG) + 0.5) * RES; ys = lo[1] + (np.arange(YG) + 0.5) * RES
    for o in objs:
        cx, cy = o["3d location"][:2]; w, l, _ = o["3d bounding box scale"]; yaw = o["3d bounding box rotation"][2]
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]); half = np.array([w/2, l/2])
        rad = float(np.hypot(w, l)/2 + RES)
        ix0, ix1 = max(int((cx-rad-lo[0])/RES), 0), min(int((cx+rad-lo[0])/RES)+1, XG)
        iy0, iy1 = max(int((cy-rad-lo[1])/RES), 0), min(int((cy+rad-lo[1])/RES)+1, YG)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
        loc = (np.stack([gx, gy], -1).reshape(-1, 2) - [cx, cy]) @ R
        grid[ix0:ix1, iy0:iy1] |= np.all(np.abs(loc) <= half, 1).reshape(ix1-ix0, iy1-iy0).astype(np.uint8)
    return grid


def main():
    calib = json.load(open(f"{SCENE}/calibration.json")); gt = json.load(open(f"{SCENE}/ground_truth.json"))
    cams = [s for s in calib["sensors"] if s.get("type") == "camera"]
    caps, good = {}, []
    for c in cams:
        cap = cv2.VideoCapture(f"{SCENE}/videos/{c['id']}.mp4")
        if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
            caps[c["id"]] = cap; good.append(c)
    good = good[:MAXCAMS]; ncam = len(good)
    # canonical grid centred on the scene's agent centroid
    allc = np.array([o["3d location"][:2] for fr in list(gt.values())[::100] for o in fr], float)
    ctr = allc.mean(0); lo = ctr - np.array([XG, YG]) * RES / 2
    grid_bounds = np.array([lo[0], lo[0]+XG*RES, lo[1], lo[1]+YG*RES], np.float32)
    # padded calibration tensors
    K = np.zeros((MAXCAMS, 3, 3), np.float32); CP = np.zeros((MAXCAMS, 3, 4), np.float32)
    valid = np.zeros((MAXCAMS,), np.float32)
    for i, c in enumerate(good):
        CP[i] = cam_proj(c); valid[i] = 1.0
        Kk = np.array(c["intrinsicMatrix"], float); Kk[0] *= W/1920.0; Kk[1] *= H/1080.0; K[i] = Kk
    os.makedirs(OUT, exist_ok=True)
    frames = sorted(int(k) for k in gt.keys())[::STRIDE]
    print(f"[xprep {os.path.basename(OUT)}] {ncam}/{MAXCAMS} cams | grid {XG}x{YG} | {len(frames)} frames", flush=True)
    # efficient grab per camera
    per_cam = {c["id"]: {} for c in good}
    for c in good:
        cap = caps[c["id"]]; want = iter(frames); nxt = next(want, None); idx = 0
        while nxt is not None:
            if idx == nxt:
                ok, fr = cap.read()
                if not ok:
                    break
                per_cam[c["id"]][nxt] = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                nxt = next(want, None)
            else:
                if not cap.grab():
                    break
            idx += 1
    n = 0
    for fidx in frames:
        if any(fidx not in per_cam[c["id"]] for c in good):
            continue
        imgs = np.zeros((MAXCAMS, H, W, 3), np.uint8)
        for i, c in enumerate(good):
            imgs[i] = per_cam[c["id"]][fidx]
        bev = rasterize(gt[str(fidx)], lo)
        np.savez_compressed(f"{OUT}/f{fidx:06d}.npz", imgs=imgs, intrins=K, cam_proj=CP,
                            cam_valid=valid, grid_bounds=grid_bounds, bev=bev)
        n += 1
    json.dump({"scene": os.path.basename(OUT), "ncam": ncam, "maxcams": MAXCAMS, "grid": [XG, YG],
               "res": RES, "n": n}, open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"DONE {os.path.basename(OUT)} n={n} ncam={ncam}", flush=True)


if __name__ == "__main__":
    main()
