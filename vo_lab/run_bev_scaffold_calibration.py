"""BEV SCAFFOLD calibration (GPU, Docker, non-billed): trains the reference network *through the
locked scaffold* (geometry + augmentation + training all owned by bev_core.py) and confirms the
gate discriminates — the reference network beats a trivial constant network on held-out IoU.

  python -m vo_lab.run_bev_scaffold_calibration

Validates the scaffold harness end-to-end before any billed agent run: proves the locked geometry/
augmentation are correct (the reference network reaches a sane IoU through them) and that a weak
network is REJECTED. Training is a GPU JOB (wall-clock, not tokens)."""
from __future__ import annotations

import sys

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord

from .agents.bev_implementer import (bev_impl_task_scaffold, bev_scaffold_degenerate_author,
                                     bev_scaffold_reference_author)
from .factory import build_vo_implementer_harness
from .plugins.bev_nuscenes import NuScenesBEVProvider


def main(root: str = "./_bev_scaffold_run", train_max=None, test_max=None, margin: float = 1.3) -> int:
    prov = NuScenesBEVProvider(train_max=train_max, test_max=test_max)

    def measure(label, author):
        task = bev_impl_task_scaffold(0.0, train_max=train_max, test_max=test_max)
        h = build_vo_implementer_harness(root, task=task, provider=prov, author_fn=author,
                                         job_mode="docker", lease_timeout_s=3600.0)
        try:
            h.image_registry.resolve(task.framework)
        except NoImageError as e:
            print(f"Need the GPU image:\n  {e}"); return None
        rec = h.run_experiment(ExperimentRecord(id=f"bev-scaffold-{label}", hypothesis=label))
        if not (rec.verdict and "miou" in rec.verdict.measured_metrics):
            print(f"{label} failed; log: {rec.log_path}"); return None
        return rec.verdict.measured_metrics

    print("Training the REFERENCE network through the LOCKED scaffold on the GPU...")
    pos = measure("ref", bev_scaffold_reference_author())
    if pos is None:
        return 1
    iou_ref = float(pos["miou"]); bar = round(iou_ref / margin, 3)
    neg = measure("degenerate", bev_scaffold_degenerate_author())
    iou_neg = float(neg["miou"]) if neg else 0.0
    opened = (iou_ref >= bar) and (iou_neg < bar)

    print("=" * 70)
    print(f"scaffold reference network — held-out mean IoU = {iou_ref:.4f}  over {pos.get('n_samples')} samples")
    print(f"derived oracle bar (ref / {margin}) = {bar}")
    print(f"trivial (constant) network         = {iou_neg:.4f}  -> {'REJECTED' if iou_neg < bar else 'PASS?!'}")
    print(f"SCAFFOLD CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print(f"=> live scaffold Track B:  ANTHROPIC_API_KEY=... python -m vo_lab.run_bev_scaffold_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
