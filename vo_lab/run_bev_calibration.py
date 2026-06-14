"""BEV calibration (GPU, Docker, non-billed): trains the from-scratch reference Lift-Splat on the
GPU through the real sandbox + grader, and confirms the gate discriminates — the trained net beats
the all-zero degenerate control on held-out nuScenes mini_val by IoU.

  python scripts/prep_nuscenes_bev.py <nuscenes_root> ~/.cache/vo_lab/bev   # once (in vo-bev:1)
  python -m vo_lab.run_bev_calibration

This validates the ENTIRE BEV Track-B pipeline (provider -> sandbox train+infer -> harness-owned
IoU grader -> oracle) without spending on the API, and derives the oracle bar for the live run.
The training is a GPU JOB (gpu_lease + CUDA image): wall-clock, not tokens."""
from __future__ import annotations

import sys

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord

from .agents.bev_implementer import (bev_degenerate_author, bev_impl_task,
                                     bev_reference_author)
from .factory import build_vo_implementer_harness
from .plugins.bev_nuscenes import NuScenesBEVProvider


def main(root: str = "./_bev_run", train_max=None, test_max=None, margin: float = 1.3) -> int:
    prov = NuScenesBEVProvider(train_max=train_max, test_max=test_max)

    def measure(label, author):
        task = bev_impl_task(0.0, train_max=train_max, test_max=test_max)
        h = build_vo_implementer_harness(root, task=task, provider=prov, author_fn=author,
                                         job_mode="docker", lease_timeout_s=3600.0)
        try:
            h.image_registry.resolve(task.framework)
        except NoImageError as e:
            print(f"Need the GPU image:\n  {e}\n"
                  "docker build -f docker/Dockerfile.gpu-torch -t vo-gpu-torch:1 ."); return None
        rec = h.run_experiment(ExperimentRecord(id=f"bev-{label}", hypothesis=label))
        if not (rec.verdict and "miou" in rec.verdict.measured_metrics):
            print(f"{label} failed; log: {rec.log_path}"); return None
        return rec.verdict.measured_metrics

    print("Training from-scratch reference Lift-Splat on the GPU (held-out nuScenes mini_val)...")
    pos = measure("ref", bev_reference_author())
    if pos is None:
        return 1
    iou_ref = float(pos["miou"]); bar = round(iou_ref / margin, 3)
    neg = measure("degenerate", bev_degenerate_author())
    iou_neg = float(neg["miou"]) if neg else 0.0
    opened = (iou_ref >= bar) and (iou_neg < bar)

    print("=" * 70)
    print(f"reference BEV — held-out mean IoU = {iou_ref:.4f}  (global {pos.get('global_iou', 0):.4f})  "
          f"over {pos.get('n_samples')} samples")
    print(f"derived oracle bar (ref / {margin}) = {bar}")
    print(f"degenerate (all-zero) control       = {iou_neg:.4f}  -> {'REJECTED' if iou_neg < bar else 'PASS?!'}")
    print(f"CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print(f"=> live Track B:  ANTHROPIC_API_KEY=... python -m vo_lab.run_bev_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
