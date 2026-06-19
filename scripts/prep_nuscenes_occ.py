"""Harness-owned nuScenes -> 3D OCCUPANCY data adapter (the 6th domain).

Per sample, the fixed on-disk contract the occupancy solver is graded against:
  imgs      (6, H, W, 3) uint8   surround cameras (same as the BEV adapter)
  intrins   (6, 3, 3)   float32  pinhole K scaled to the resized images
  cam2ego   (6, 4, 4)   float32  camera -> ego extrinsic
  occ       (XG, YG, ZG) uint8   HARNESS-OWNED GT: vehicle occupancy in the ego VOXEL grid

The 3D GT fills each vehicle's oriented 3D box extent into the ego voxel grid (validated in the
Phase-0 de-risk; Z tightened to [-2,4] from the occupied-voxel histogram). Held-out = official
nuScenes mini_val scenes. Box-derived vehicle occupancy (not dense semantic Occ3D) -- self-contained
and harness-owned, the honest simplification stated in the report.

usage: python prep_nuscenes_occ.py <dataroot> <out_dir> [H] [W]
"""
import sys, os, json
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

DATAROOT, OUT = sys.argv[1], sys.argv[2]
H = int(sys.argv[3]) if len(sys.argv) > 3 else 128
W = int(sys.argv[4]) if len(sys.argv) > 4 else 352
OW, OH = 1600, 900
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
XB = YB = (-50.0, 50.0)
ZB = (-2.0, 4.0)                                       # tightened from the de-risk Z histogram
RES = 0.5
XG = int((XB[1] - XB[0]) / RES); YG = int((YB[1] - YB[0]) / RES); ZG = int((ZB[1] - ZB[0]) / RES)
_xs = XB[0] + (np.arange(XG) + 0.5) * RES
_ys = YB[0] + (np.arange(YG) + 0.5) * RES
_zs = ZB[0] + (np.arange(ZG) + 0.5) * RES


def voxelize(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    trans = -np.array(ego["translation"]); rot = Quaternion(ego["rotation"]).inverse
    grid = np.zeros((XG, YG, ZG), np.uint8)
    for tok in sample["anns"]:
        ann = nusc.get("sample_annotation", tok)
        if not ann["category_name"].startswith("vehicle"):
            continue
        box = nusc.get_box(ann["token"]); box.translate(trans); box.rotate(rot)
        c = box.center; R = box.rotation_matrix; w, l, h = box.wlh
        half = np.array([l / 2, w / 2, h / 2])
        cor = box.corners().T; mn, mx = cor.min(0), cor.max(0)
        ix0, ix1 = max(np.searchsorted(_xs, mn[0]) - 1, 0), min(np.searchsorted(_xs, mx[0]) + 1, XG)
        iy0, iy1 = max(np.searchsorted(_ys, mn[1]) - 1, 0), min(np.searchsorted(_ys, mx[1]) + 1, YG)
        iz0, iz1 = max(np.searchsorted(_zs, mn[2]) - 1, 0), min(np.searchsorted(_zs, mx[2]) + 1, ZG)
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue
        gx, gy, gz = np.meshgrid(_xs[ix0:ix1], _ys[iy0:iy1], _zs[iz0:iz1], indexing="ij")
        local = (np.stack([gx, gy, gz], -1).reshape(-1, 3) - c) @ R
        inside = np.all(np.abs(local) <= half, 1).reshape(ix1 - ix0, iy1 - iy0, iz1 - iz0)
        grid[ix0:ix1, iy0:iy1, iz0:iz1] |= inside.astype(np.uint8)
    return grid


def cam_record(nusc, sample, cam):
    sd = nusc.get("sample_data", sample["data"][cam])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    img = cv2.resize(cv2.imread(os.path.join(DATAROOT, sd["filename"])), (W, H), interpolation=cv2.INTER_AREA)
    K = np.array(cs["camera_intrinsic"], np.float64).copy()
    K[0] *= W / OW; K[1] *= H / OH
    c2e = np.eye(4); c2e[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix; c2e[:3, 3] = cs["translation"]
    return img[:, :, ::-1], K.astype(np.float32), c2e.astype(np.float32)


def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    splits = create_splits_scenes()
    train_sc, val_sc = set(splits["mini_train"]), set(splits["mini_val"])
    name_of = {s["token"]: s["name"] for s in nusc.scene}
    os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "val"), exist_ok=True)
    index = {"train": [], "val": []}
    for si, sample in enumerate(nusc.sample):
        scene = name_of[sample["scene_token"]]
        split = "train" if scene in train_sc else ("val" if scene in val_sc else None)
        if split is None:
            continue
        imgs, Ks, c2es = [], [], []
        for cam in CAMS:
            im, K, c2e = cam_record(nusc, sample, cam)
            imgs.append(im); Ks.append(K); c2es.append(c2e)
        occ = voxelize(nusc, sample)
        tok = sample["token"]
        np.savez_compressed(os.path.join(OUT, split, tok + ".npz"),
                            imgs=np.stack(imgs).astype(np.uint8), intrins=np.stack(Ks),
                            cam2ego=np.stack(c2es), occ=occ)
        index[split].append(tok)
        if si % 50 == 0:
            print(f"  {si}/{len(nusc.sample)} {split} occ_voxels={int(occ.sum())}", flush=True)
    meta = {"cams": CAMS, "H": H, "W": W, "grid": [XG, YG, ZG], "res": RES,
            "x_bounds": XB, "y_bounds": YB, "z_bounds": ZB,
            "val_scenes": sorted(val_sc), "n_train": len(index["train"]), "n_val": len(index["val"])}
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"DONE train={meta['n_train']} val={meta['n_val']} grid={XG}x{YG}x{ZG}", flush=True)


if __name__ == "__main__":
    main()
