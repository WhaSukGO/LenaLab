"""Occupancy calibration (GPU, Docker, non-billed): trains the from-scratch reference Lift-Splat-to-3D
through the real sandbox + grader and confirms the gate discriminates (reference beats all-empty on
held-out voxel IoU). Validates the whole occupancy Track-B pipeline + derives the oracle bar.

  python scripts/prep_nuscenes_occ.py <nuscenes_root> ~/.cache/vo_lab/occ   # once (vo-bev:1)
  python -m vo_lab.run_occ_calibration"""
from __future__ import annotations

import sys

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord

from .agents.occ_implementer import occ_degenerate_author, occ_impl_task, occ_reference_author
from .factory import build_vo_implementer_harness
from .plugins.occ_nuscenes import NuScenesOccProvider


def main(root: str = "./_occ_run", train_max=None, test_max=None, margin: float = 1.3) -> int:
    prov = NuScenesOccProvider(train_max=train_max, test_max=test_max)

    def measure(label, author):
        task = occ_impl_task(0.0, train_max=train_max, test_max=test_max)
        h = build_vo_implementer_harness(root, task=task, provider=prov, author_fn=author,
                                         job_mode="docker", lease_timeout_s=3600.0)
        try:
            h.image_registry.resolve(task.framework)
        except NoImageError as e:
            print(f"Need the GPU image:\n  {e}"); return None
        rec = h.run_experiment(ExperimentRecord(id=f"occ-{label}", hypothesis=label))
        if not (rec.verdict and "miou" in rec.verdict.measured_metrics):
            print(f"{label} failed; log: {rec.log_path}"); return None
        return rec.verdict.measured_metrics

    print("Training from-scratch reference Lift-Splat-to-3D on the GPU (held-out nuScenes mini_val)...")
    pos = measure("ref", occ_reference_author())
    if pos is None:
        return 1
    iou_ref = float(pos["miou"]); bar = round(iou_ref / margin, 4)
    neg = measure("degenerate", occ_degenerate_author())
    iou_neg = float(neg["miou"]) if neg else 0.0
    opened = (iou_ref >= bar) and (iou_neg < bar)
    print("=" * 70)
    print(f"reference occupancy — held-out mean voxel IoU = {iou_ref:.4f}  over {pos.get('n_samples')} samples")
    print(f"derived oracle bar (ref / {margin}) = {bar}")
    print(f"degenerate (all-empty) control      = {iou_neg:.4f}  -> {'REJECTED' if iou_neg < bar else 'PASS?!'}")
    print(f"OCC CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print(f"=> live Track B:  ANTHROPIC_API_KEY=... python -m vo_lab.run_occ_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
