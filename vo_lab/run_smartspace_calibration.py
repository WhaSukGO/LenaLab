"""Smart-space calibration (GPU, non-billed): trains the from-scratch IPM reference through the real
sandbox + grader and confirms the gate discriminates (reference beats all-empty on held-out floor IoU).
Validates the whole smart-space Track-B pipeline + derives the oracle bar.

  python scripts/prep_smartspace.py <scene_dir> ~/.cache/vo_lab/smartspace_occ   # once
  python -m vo_lab.run_smartspace_calibration"""
from __future__ import annotations

import sys

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord

from .agents.smartspace_implementer import (smartspace_degenerate_author, smartspace_impl_task,
                                            smartspace_reference_author)
from .factory import build_vo_implementer_harness
from .plugins.smartspace import SmartSpaceProvider


def main(root: str = "./_smartspace_run", train_max=None, test_max=None, margin: float = 1.3) -> int:
    prov = SmartSpaceProvider(train_max=train_max, test_max=test_max)

    def measure(label, author):
        task = smartspace_impl_task(0.0, train_max=train_max, test_max=test_max)
        h = build_vo_implementer_harness(root, task=task, provider=prov, author_fn=author,
                                         job_mode="docker", lease_timeout_s=3600.0)
        try:
            h.image_registry.resolve(task.framework)
        except NoImageError as e:
            print(f"Need the GPU image:\n  {e}"); return None
        rec = h.run_experiment(ExperimentRecord(id=f"smartspace-{label}", hypothesis=label))
        if not (rec.verdict and "miou" in rec.verdict.measured_metrics):
            print(f"{label} failed; log: {rec.log_path}"); return None
        return rec.verdict.measured_metrics

    print("Training from-scratch IPM reference on the GPU (held-out unseen-time frames)...")
    pos = measure("ref", smartspace_reference_author())
    if pos is None:
        return 1
    iou_ref = float(pos["miou"]); bar = round(iou_ref / margin, 4)
    neg = measure("degenerate", smartspace_degenerate_author())
    iou_neg = float(neg["miou"]) if neg else 0.0
    opened = (iou_ref >= bar) and (iou_neg < bar)
    print("=" * 70)
    print(f"reference smart-space — held-out mean floor IoU = {iou_ref:.4f}  over {pos.get('n_samples')} samples")
    print(f"derived oracle bar (ref / {margin}) = {bar}")
    print(f"degenerate (all-empty) control      = {iou_neg:.4f}  -> {'REJECTED' if iou_neg < bar else 'PASS?!'}")
    print(f"SMARTSPACE CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print(f"=> live Track B:  ANTHROPIC_API_KEY=... python -m vo_lab.run_smartspace_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
