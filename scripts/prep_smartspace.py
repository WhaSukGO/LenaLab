"""Harness-owned NVIDIA Physical AI Smart Spaces -> 2D BEV floor-occupancy adapter (the 7th domain).

Per sample (one timestamp in one warehouse scene), the fixed on-disk contract the solver is graded on:
  imgs        (N, H, W, 3) uint8    N static-camera views (resized)
  intrins     (N, 3, 3)   float32   pinhole K scaled to the resized images
  cam_proj    (N, 3, 4)   float32   world->image projection (scaled cameraMatrix); IPM samples through it
  grid_bounds (4,)        float32   [x0, x1, y0, y1] world extent the BEV grid covers (meters)
  bev         (XG, YG)    uint8     HARNESS-OWNED GT: agent floor-occupancy on the world grid

GT = oriented XY footprints of every annotated agent (Person/Forklift/PalletTruck/Transporter/robots)
rasterized into the world floor grid. Per-space self-verification split: the scene's first SPLIT of the
timeline -> train, the last (1-SPLIT) -> val (held-out, UNSEEN TIME, same space). Each camera's world->cam
[R|t] is recovered from the verified cameraMatrix P as inv(K)@P (Phase-0: P projects GT into the image at
7/8 inside the dataset's own 2D boxes), so cam2world = inverse -- no ambiguity about the extrinsic convention.

usage: python prep_smartspace.py <scene_dir> <out_dir> [stride=30] [res=0.5] [H=128] [W=352] [max_cams=0]
"""
import sys, os, json
import numpy as np
import cv2

SCENE, OUT = sys.argv[1], sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 30      # sample 1 frame / STRIDE (30 = 1/sec @ 30fps)
RES = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5      # meters/cell (mirrors nuScenes BEV scale)
H = int(sys.argv[5]) if len(sys.argv) > 5 else 128
W = int(sys.argv[6]) if len(sys.argv) > 6 else 352
MAX_CAMS = int(sys.argv[7]) if len(sys.argv) > 7 else 0      # 0 = all cameras in the scene
SPLIT = 0.70                                                 # first 70% of time -> train, last 30% -> val


def cam_geometry(sensor):
    """K_resized (3,3), cam_proj (3,4) = the verified world->image cameraMatrix scaled to the resized
    image. IPM projects floor cells straight through cam_proj (Phase-0 verified: 7/8 inside GT 2D boxes)
    -- no cam2world decomposition, which is brittle for projectively-scaled cameraMatrices."""
    K = np.array(sensor["intrinsicMatrix"], float)
    P = np.array(sensor["cameraMatrix"], float)             # 3x4 world->original-image
    S = np.diag([W / 1920.0, H / 1080.0, 1.0])              # videos are 1080p; scale pixels to resized
    cam_proj = S @ P
    Ks = K.copy(); Ks[0] *= W / 1920.0; Ks[1] *= H / 1080.0
    return Ks.astype(np.float32), cam_proj.astype(np.float32)


def world_bounds(gt):
    c = np.array([o["3d location"][:2] for fr in list(gt.values())[::100] for o in fr], float)
    lo, hi = c.min(0) - 2.0, c.max(0) + 2.0
    XG = int(np.ceil((hi[0] - lo[0]) / RES)); YG = int(np.ceil((hi[1] - lo[1]) / RES))
    return lo, XG, YG


def rasterize(objs, lo, XG, YG):
    grid = np.zeros((XG, YG), np.uint8)
    xs = lo[0] + (np.arange(XG) + 0.5) * RES
    ys = lo[1] + (np.arange(YG) + 0.5) * RES
    for o in objs:
        cx, cy = o["3d location"][:2]
        w, l, _ = o["3d bounding box scale"]; yaw = o["3d bounding box rotation"][2]
        Rm = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        half = np.array([w / 2, l / 2]); rad = float(np.hypot(w, l) / 2 + RES)
        ix0, ix1 = max(int((cx - rad - lo[0]) / RES), 0), min(int((cx + rad - lo[0]) / RES) + 1, XG)
        iy0, iy1 = max(int((cy - rad - lo[1]) / RES), 0), min(int((cy + rad - lo[1]) / RES) + 1, YG)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], indexing="ij")
        local = (np.stack([gx, gy], -1).reshape(-1, 2) - [cx, cy]) @ Rm
        inside = np.all(np.abs(local) <= half, 1).reshape(ix1 - ix0, iy1 - iy0)
        grid[ix0:ix1, iy0:iy1] |= inside.astype(np.uint8)
    return grid


def main():
    calib = json.load(open(f"{SCENE}/calibration.json"))
    gt = json.load(open(f"{SCENE}/ground_truth.json"))
    cams = [s for s in calib["sensors"] if s.get("type") == "camera"]
    if MAX_CAMS:
        cams = cams[:MAX_CAMS]
    # drop cameras whose video is missing/truncated/unreadable (degrade to fewer cams, not 0 samples)
    caps, good = {}, []
    for c in cams:
        cap = cv2.VideoCapture(f"{SCENE}/videos/{c['id']}.mp4")
        if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
            caps[c["id"]] = cap; good.append(c)
        else:
            print(f"  WARN: dropping {c['id']} (unreadable/empty video)", flush=True)
    cams = good
    cam_ids = [c["id"] for c in cams]
    Ks = np.stack([cam_geometry(c)[0] for c in cams])
    cam_projs = np.stack([cam_geometry(c)[1] for c in cams])

    lo, XG, YG = world_bounds(gt)
    grid_bounds = np.array([lo[0], lo[0] + XG * RES, lo[1], lo[1] + YG * RES], np.float32)
    frames = sorted(int(k) for k in gt.keys())[::STRIDE]
    n_tr = int(len(frames) * SPLIT)
    os.makedirs(f"{OUT}/train", exist_ok=True); os.makedirs(f"{OUT}/val", exist_ok=True)
    index = {"train": [], "val": []}
    print(f"[prep] {len(cam_ids)} cams | grid {XG}x{YG} @ {RES}m bounds {grid_bounds.round(1)} | "
          f"{len(frames)} samples ({n_tr} train / {len(frames)-n_tr} val)", flush=True)

    # Efficient extraction: per camera, walk frames sequentially, cheap grab() to skip
    # non-sampled frames and only decode (retrieve) the sampled ones -- avoids slow random seeks.
    per_cam = {cid: {} for cid in cam_ids}
    for cid in cam_ids:
        cap = caps[cid]; want = iter(frames); nxt = next(want, None); idx = 0
        while nxt is not None:
            if idx == nxt:
                ok, fr = cap.read()
                if not ok:
                    break
                per_cam[cid][nxt] = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                nxt = next(want, None)
            else:
                if not cap.grab():
                    break
            idx += 1
        print(f"  cam {cid}: {len(per_cam[cid])}/{len(frames)} frames", flush=True)

    for i, fidx in enumerate(frames):
        if any(fidx not in per_cam[cid] for cid in cam_ids):
            continue
        imgs = np.stack([per_cam[cid][fidx] for cid in cam_ids])
        bev = rasterize(gt[str(fidx)], lo, XG, YG)
        split = "train" if i < n_tr else "val"
        tok = f"f{fidx:06d}"
        np.savez_compressed(f"{OUT}/{split}/{tok}.npz",
                            imgs=imgs.astype(np.uint8), intrins=Ks,
                            cam_proj=cam_projs, grid_bounds=grid_bounds, bev=bev)
        index[split].append(tok)
        if i % 25 == 0:
            print(f"  {i}/{len(frames)} {split} occ_cells={int(bev.sum())}", flush=True)

    meta = {"scene": os.path.basename(SCENE.rstrip("/")), "cams": cam_ids, "H": H, "W": W,
            "grid": [XG, YG], "res": RES, "grid_bounds": grid_bounds.tolist(), "stride": STRIDE,
            "split": SPLIT, "n_train": len(index["train"]), "n_val": len(index["val"]),
            "note": "per-space self-verification: train=first 70% time, val=last 30% (unseen time, same space)"}
    json.dump(index, open(f"{OUT}/index.json", "w"))
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
    print(f"DONE train={meta['n_train']} val={meta['n_val']} grid={XG}x{YG}", flush=True)


if __name__ == "__main__":
    main()
