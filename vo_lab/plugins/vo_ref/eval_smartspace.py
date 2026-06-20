"""Independent grader for SMART-SPACE 2D floor OCCUPANCY — HARNESS-OWNED (the solver never writes this).

The train+infer job writes $LAB_ARTIFACTS/pred_<token>.npy (an XG*YG uint8 {0,1} floor mask) for each
held-out sample. This grader reads those, compares each to the held-out GT ($LAB_DATA/<token>_bev.npy --
the secret label), and reports mean per-sample IoU (primary) + pooled IoU. Restored from the task spec
before judging, so a tampered grader earns nothing. Held-out = the scene's last 30% of time (unseen),
i.e. per-space self-verification."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    heldout = Path(os.environ["LAB_DATA"]); art = Path(os.environ["LAB_ARTIFACTS"])
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)
    gts = sorted(heldout.glob("*_bev.npy"))
    per, inter_t, union_t, missing = [], 0, 0, 0
    for gp in gts:
        tok = gp.name[:-len("_bev.npy")]
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
           "metric": "smartspace_floor_iou",
           "caveats": ["per-sample mean 2D floor-occupancy IoU over held-out (unseen-time) frames",
                       "per-space self-verification: train=first 70% of the scene timeline, held-out=last 30%",
                       "box-derived agent occupancy (people/forklifts/robots), static multi-camera warehouse",
                       "missing/wrong-shape predictions score IoU 0"]}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"held-out mean floor IoU = {miou:.4f} (global {global_iou:.4f}) over {len(gts)} samples [{missing} missing]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
