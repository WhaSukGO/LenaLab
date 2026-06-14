"""Independent grader for BEV vehicle-occupancy — HARNESS-OWNED (the solver never writes this).

The train+infer job writes $LAB_ARTIFACTS/pred_<token>.npy (a 200x200 uint8 {0,1} BEV occupancy
mask) for each held-out sample <token>. This grader reads those, compares each to the held-out GT
($LAB_DATA/<token>_bev.npy -- the secret label, never in the trainer's input), and reports the
mean per-sample IoU (the primary metric) + the pooled (dataset-level) IoU. Higher is better.

The solver authors only main.py; this file is restored from the task spec before judging, so a
tampered grader earns nothing. The solver picks its own occupancy threshold (a legitimate
algorithm choice); the grader just measures IoU of the binary masks it produced vs hidden GT."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    heldout = Path(os.environ["LAB_DATA"])             # <token>_bev.npy (grader-only labels)
    art = Path(os.environ["LAB_ARTIFACTS"])            # pred_<token>.npy from the train+infer job
    eval_out = Path(os.environ["LAB_EVAL_OUT"]); eval_out.mkdir(parents=True, exist_ok=True)

    gts = sorted(heldout.glob("*_bev.npy"))
    per, inter_t, union_t, missing = [], 0, 0, 0
    for gp in gts:
        tok = gp.name[:-len("_bev.npy")]
        gt = np.load(gp).astype(bool)
        pp = art / f"pred_{tok}.npy"
        if not pp.exists():
            per.append(0.0); missing += 1                # no prediction -> IoU 0 (penalised)
            union_t += int(gt.sum()); continue
        pred = np.load(pp)
        if pred.shape != gt.shape:                       # wrong shape -> invalid -> IoU 0
            per.append(0.0); missing += 1
            union_t += int(gt.sum()); continue
        pred = pred.astype(bool)
        inter = int((pred & gt).sum()); union = int((pred | gt).sum())
        inter_t += inter; union_t += union
        per.append(inter / union if union > 0 else 1.0)  # both empty -> perfect

    miou = float(np.mean(per)) if per else 0.0           # per-sample mean IoU (primary)
    global_iou = inter_t / union_t if union_t > 0 else 0.0
    out = {"miou": miou, "global_iou": global_iou, "n_samples": len(gts),
           "n_missing_pred": missing, "metric": "bev_vehicle_iou",
           "caveats": [
               "per-sample mean IoU over held-out nuScenes mini_val scenes (the primary metric)",
               "missing/wrong-shape predictions score IoU 0 for that sample",
               "binary-mask IoU; the solver chose its own occupancy threshold",
           ]}
    json.dump(out, open(eval_out / "heldout.json", "w"))
    print(f"held-out mean IoU = {miou:.4f}  (global {global_iou:.4f})  over {len(gts)} samples"
          f"  [{missing} missing]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
