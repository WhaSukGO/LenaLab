"""Prep a KITTI odometry sequence into a PERSISTENT dir (survives /tmp clearing) for the DROID adapter:
left_/right_ symlinks -> cache image_0/image_1, intrinsics.txt (fx fy cx cy baseline from P0/P1), gt.txt.
usage: python prep_kitti_odo.py <seq> [<seq> ...]   -> ~/.cache/vo_lab/kitti_prep/seq<seq>/"""
import sys, os, glob, numpy as np
from pathlib import Path
CACHE = Path(os.path.expanduser("~/.cache/vo_lab/kitti/dataset"))
DST = Path(os.path.expanduser("~/.cache/vo_lab/kitti_prep"))
for seq in sys.argv[1:]:
    calib = {}
    for line in open(CACHE / f"sequences/{seq}/calib.txt"):
        k, *v = line.split(); calib[k.rstrip(':')] = np.array(v, float).reshape(3, 4)
    P0, P1 = calib["P0"], calib["P1"]
    fx, fy, cx, cy = P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2]; baseline = -P1[0, 3] / fx
    out = DST / f"seq{seq}" / "input"; out.mkdir(parents=True, exist_ok=True)
    L = sorted(glob.glob(str(CACHE / f"sequences/{seq}/image_0/*.png")))
    R = sorted(glob.glob(str(CACHE / f"sequences/{seq}/image_1/*.png")))
    for i, (l, r) in enumerate(zip(L, R)):
        for p, t in [(l, 'left'), (r, 'right')]:
            d = out / f"{t}_{i:06d}.png"
            if not d.exists(): os.symlink(p, d)
    np.savetxt(out / "intrinsics.txt", [fx, fy, cx, cy, baseline], fmt="%.6f")
    centres = np.loadtxt(CACHE / f"poses/{seq}.txt").reshape(-1, 3, 4)[:, :, 3]
    np.savetxt(DST / f"seq{seq}" / "gt.txt", centres, fmt="%.6f")
    path = np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()
    print(f"  seq_{seq}: {len(L)} frames, {path:.0f}m -> {DST}/seq{seq}")
