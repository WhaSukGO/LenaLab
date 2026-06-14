"""Harness-owned nuScenes -> BEV data adapter.

For each sample produces the fixed on-disk contract the BEV solver is graded against:
  imgs      (6, H, W, 3) uint8   surround cameras, anisotropically resized
  intrins   (6, 3, 3)   float32  pinhole K SCALED to the resized images
  cam2ego   (6, 4, 4)   float32  camera -> ego extrinsic (rotation + translation)
  bev       (Xg, Yg)    uint8    HARNESS-OWNED GT: vehicle occupancy in the ego BEV grid

The BEV GT is rasterized from the 3D box annotations (footprint polygons) in the EGO frame at
the sample's ego pose -- the solver never sees it. Held-out split = official nuScenes `mini_val`
scenes (disjoint from `mini_train`), so the grader measures generalization to unseen scenes.

usage: python prep_nuscenes_bev.py <dataroot> <out_dir> [H] [W]
out: <out>/train/<token>.npz, <out>/val/<token>.npz, <out>/index.json, <out>/meta.json
"""
import sys, os, json
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

DATAROOT = sys.argv[1]
OUT = sys.argv[2]
H = int(sys.argv[3]) if len(sys.argv) > 3 else 128
W = int(sys.argv[4]) if len(sys.argv) > 4 else 352
OW, OH = 1600, 900                                   # native nuScenes image size
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

# BEV grid: ego x-forward / y-left, 100m x 100m at 0.5 m  -> 200 x 200
XB, YB, RES = (-50.0, 50.0), (-50.0, 50.0), 0.5
XG = int((XB[1] - XB[0]) / RES)                       # rows  (x / forward)
YG = int((YB[1] - YB[0]) / RES)                       # cols  (y / left)


def rasterize_bev(nusc, sample):
    """Vehicle footprints -> binary BEV occupancy in the ego frame of this sample."""
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    trans = -np.array(ego["translation"])
    rot = Quaternion(ego["rotation"]).inverse
    grid = np.zeros((XG, YG), np.uint8)
    for tok in sample["anns"]:
        ann = nusc.get("sample_annotation", tok)
        if not ann["category_name"].startswith("vehicle"):
            continue
        box = nusc.get_box(ann["token"])
        box.translate(trans)                          # global -> ego (translate then rotate)
        box.rotate(rot)
        corners = box.bottom_corners()[:2].T          # (4,2) ego x,y in metres
        gx = (corners[:, 0] - XB[0]) / RES            # x forward -> row
        gy = (corners[:, 1] - YB[0]) / RES            # y left    -> col
        poly = np.stack([gy, gx], 1).astype(np.int32) # cv2 wants (col,row)
        cv2.fillConvexPoly(grid, poly, 1)
    return grid


def cam_record(nusc, sample, cam):
    sd = nusc.get("sample_data", sample["data"][cam])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    img = cv2.imread(os.path.join(DATAROOT, sd["filename"]))   # BGR
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    K = np.array(cs["camera_intrinsic"], np.float64).copy()
    K[0] *= W / OW                                    # scale fx, cx (anisotropic resize)
    K[1] *= H / OH                                    # scale fy, cy
    c2e = np.eye(4)
    c2e[:3, :3] = Quaternion(cs["rotation"]).rotation_matrix
    c2e[:3, 3] = np.array(cs["translation"])
    return img[:, :, ::-1], K.astype(np.float32), c2e.astype(np.float32)   # RGB


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
        bev = rasterize_bev(nusc, sample)
        tok = sample["token"]
        np.savez_compressed(os.path.join(OUT, split, tok + ".npz"),
                            imgs=np.stack(imgs).astype(np.uint8),
                            intrins=np.stack(Ks), cam2ego=np.stack(c2es), bev=bev)
        index[split].append(tok)
        if si % 50 == 0:
            print(f"  {si}/{len(nusc.sample)}  {split}  vehicles_px={int(bev.sum())}", flush=True)
    meta = {"cams": CAMS, "H": H, "W": W, "grid": [XG, YG], "res": RES,
            "x_bounds": XB, "y_bounds": YB,
            "train_scenes": sorted(train_sc), "val_scenes": sorted(val_sc),
            "n_train": len(index["train"]), "n_val": len(index["val"])}
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"DONE train={meta['n_train']} val={meta['n_val']} grid={XG}x{YG}", flush=True)


if __name__ == "__main__":
    main()
