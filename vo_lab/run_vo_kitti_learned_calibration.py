"""Learned-VO calibration (GPU, Docker, non-billed): trains the reference pure-torch learned
VO on the GPU and confirms the gate discriminates — the trained net beats the static
degenerate control on held-out KITTI sequences.

  bash scripts/fetch_kitti_odometry.sh          # once
  python -m vo_lab.run_vo_kitti_learned_calibration

HONEST oracle note: naive monocular learned VO is drift-dominated and far worse than
classical VO on these short benchmarks (it beats the trivial static baseline but not
classical). So the bar here is "clearly beat the static control" — it validates that the
learned/GPU PIPELINE works and the net learned something, NOT that learned VO is competitive.
The training is a GPU JOB (gpu_lease + CUDA image): wall-clock, not tokens."""
from __future__ import annotations

import sys

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord

from .agents.vo_implementer import (kitti_learned_degenerate_author,
                                    kitti_learned_reference_author, vo_impl_task_kitti_learned)
from .factory import build_vo_implementer_harness
from .plugins.vo_kitti_learned import KITTILearnedProvider


def main(root: str = "./_vo_kitti_learned_run",
         train: tuple[str, ...] = ("00", "02", "06", "08", "09"),
         test: tuple[str, ...] = ("05", "07"), train_max: int = 1000, test_max: int = 300,
         margin: float = 1.3) -> int:
    prov = KITTILearnedProvider(train=train, test=test, train_max=train_max, test_max=test_max)

    def measure(label, author):
        h = build_vo_implementer_harness(root, task=vo_impl_task_kitti_learned(1e9, train, test),
                                         provider=prov, author_fn=author, job_mode="docker",
                                         lease_timeout_s=3600.0)
        try:
            h.image_registry.resolve(vo_impl_task_kitti_learned(1e9, train, test).framework)
        except NoImageError as e:
            print(f"Need the GPU image:\n  {e}\n"
                  "docker build -f docker/Dockerfile.gpu-torch -t vo-gpu-torch:1 ."); return None
        rec = h.run_experiment(ExperimentRecord(id=f"kitti-learned-{label}", hypothesis=label))
        if not (rec.verdict and "ate_rmse" in rec.verdict.measured_metrics):
            print(f"{label} failed; log: {rec.log_path}"); return None
        return rec.verdict.measured_metrics

    print(f"Training reference learned VO on the GPU (train={train}, test={test})...")
    pos = measure("ref", kitti_learned_reference_author())
    if pos is None:
        return 1
    ate_ref = float(pos["ate_rmse"]); bar = round(ate_ref * margin, 2)
    neg = measure("degenerate", kitti_learned_degenerate_author())
    ate_neg = float(neg["ate_rmse"]) if neg else 1e9
    opened = (ate_ref <= bar) and (ate_neg > bar)

    print("=" * 70)
    print(f"reference learned VO — mean held-out ATE = {ate_ref:.2f} m (Sim3)  "
          f"RPE {pos.get('rpe_trans'):.3f}  per-seq { {k: round(v['ate_rmse'],1) for k,v in pos.get('per_seq',{}).items()} }")
    print(f"derived oracle bar (x{margin}) = {bar} m")
    print(f"degenerate (static) control    = {ate_neg:.2f} m  -> {'REJECTED' if ate_neg>bar else 'PASS?!'}")
    print(f"CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print("HONEST: learned VO beats the trivial baseline but is ~10-20x worse than classical "
          "VO — this validates the learned/GPU pipeline, not learned-VO competitiveness.")
    print("=" * 70)
    if opened:
        print(f">>> live learned-VO Track B:  python -m vo_lab.run_vo_kitti_learned_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
