"""Independent grader for 3D vehicle OCCUPANCY — HARNESS-OWNED (the solver never writes this).

The train+infer job writes $LAB_ARTIFACTS/pred_<token>.npy (a XG*YG*ZG uint8 {0,1} voxel mask) for
each held-out sample. This grader reads those, compares each to the held-out GT
($LAB_DATA/<token>_occ.npy -- the secret label), and reports mean per-sample voxel IoU (primary) +
pooled IoU. Restored from the task spec before judging, so a tampered grader earns nothing."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    heldout = Path(os.environ["LAB_DATA"]); art = Path(os.environ["LAB_ARTIFACTS"])
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)
    gts = sorted(heldout.glob("*_occ.npy"))
    per, inter_t, union_t, missing = [], 0, 0, 0
    for gp in gts:
        tok = gp.name[:-len("_occ.npy")]
        gt = np.load(gp).astype(bool)
        pp = art / f"pred_{tok}.npy"
        if not pp.exists():
            per.append(0.0); missing += 1; union_t += int(gt.sum()); continue
        pred = np.load(pp)
        if pred.shape != gt.shape:
            per.append(0.0); missing += 1; union_t += int(gt.sum()); continue
        pred = pred.astype(bool)
        inter = int((pred & gt).sum()); union = int((pred | gt).sum())
        inter_t += inter; union_t += union
        per.append(inter / union if union > 0 else 1.0)
    miou = float(np.mean(per)) if per else 0.0
    global_iou = inter_t / union_t if union_t > 0 else 0.0
    out = {"miou": miou, "global_iou": global_iou, "n_samples": len(gts), "n_missing_pred": missing,
           "metric": "occupancy_voxel_iou",
           "caveats": ["per-sample mean 3D voxel IoU over held-out nuScenes mini_val scenes",
                       "box-derived vehicle occupancy (not dense semantic Occ3D)",
                       "missing/wrong-shape predictions score IoU 0"]}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"held-out mean voxel IoU = {miou:.4f} (global {global_iou:.4f}) over {len(gts)} samples [{missing} missing]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
