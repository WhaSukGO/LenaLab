"""RGB-D + generalization calibration (LOCAL, non-billed): downloads the dev + held-out
sequences once, measures the reference RGB-D VO's mean held-out ATE, derives the bar, and
checks the gate discriminates (reference passes, degenerate fails) — on UNSEEN sequences.

  python -m vo_lab.run_vo_tum_rgbd_calibration

Proves the whole improvement pipeline (depth exposure + generalization grader + RPE) with
the reference solver, before any billed live run."""
from __future__ import annotations

import sys

from lab.models import ExperimentRecord

from .agents.vo_implementer import degenerate_author, rgbd_reference_author, vo_impl_task_rgbd
from .factory import build_vo_implementer_harness
from .plugins.vo_rgbd import TUMRGBDProvider


def main(root: str = "./_vo_rgbd_run", dev: str = "fr1_xyz",
         heldout: tuple[str, ...] = ("fr1_desk",), max_frames: int = 200,
         margin: float = 1.5) -> int:
    prov = TUMRGBDProvider(dev=dev, heldout=heldout, max_frames=max_frames)

    print(f"Measuring reference RGB-D VO on held-out {heldout} (first run downloads ~0.5 GB/seq)...")
    hp = build_vo_implementer_harness(root, task=vo_impl_task_rgbd(1e9, dev=dev, heldout=heldout),
                                      provider=prov, author_fn=rgbd_reference_author(),
                                      job_mode="local")
    pos = hp.run_experiment(ExperimentRecord(id="rgbd-cal-pos", hypothesis="reference RGB-D VO"))
    if not (pos.verdict and "ate_rmse" in pos.verdict.measured_metrics):
        print("measurement failed; see log:", pos.log_path); return 1
    mm = pos.verdict.measured_metrics
    ate_ref = float(mm["ate_rmse"]); bar = round(ate_ref * margin, 4)

    hn = build_vo_implementer_harness(root, task=vo_impl_task_rgbd(bar, dev=dev, heldout=heldout),
                                      provider=prov, author_fn=degenerate_author(),
                                      job_mode="local")
    neg = hn.run_experiment(ExperimentRecord(id="rgbd-cal-neg", hypothesis="degenerate"))
    ate_neg = float(neg.verdict.measured_metrics.get("ate_rmse", 1e9))
    opened = (ate_ref <= bar) and (ate_neg > bar)

    print("=" * 66)
    print(f"reference RGB-D VO — mean held-out ATE = {ate_ref:.4f} m  "
          f"(RPE {mm.get('rpe_trans'):.4f}, scale_err {mm.get('scale_err'):.3f})")
    print(f"  per-seq: { {k: round(v['ate_rmse'],3) for k,v in mm.get('per_seq',{}).items()} }")
    print(f"derived oracle bar (x{margin}) = {bar} m")
    print(f"degenerate control mean ATE    = {ate_neg:.4f} m  -> {'REJECTED' if ate_neg>bar else 'PASS?!'}")
    print(f"CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print("=" * 66)
    if opened:
        print(f">>> live RGB-D Track B:  python -m vo_lab.run_vo_tum_rgbd_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
