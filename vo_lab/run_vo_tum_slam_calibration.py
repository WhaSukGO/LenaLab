"""SLAM calibration (Docker, non-billed): on a long LOOP sequence, confirm that loop closure
is NECESSARY — the reference SLAM passes, but plain VO-only (and a degenerate control) fail
the bar. Runs in Docker so it also validates the live pipeline (mounts + contract).

  python -m vo_lab.run_vo_tum_slam_calibration

Sets the bar = reference-SLAM ATE x margin, and verifies VO-only ATE > bar (i.e. you can't
clear it without closing loops)."""
from __future__ import annotations

import sys

from lab.models import ExperimentRecord

from .agents.vo_implementer import (degenerate_author, rgbd_reference_author,
                                    slam_reference_author, vo_impl_task_slam)
from .factory import build_vo_implementer_harness
from .plugins.vo_rgbd import TUMRGBDProvider


def main(root: str = "./_vo_slam_run", dev: str = "fr1_room",
         heldout: tuple[str, ...] = ("fr2_desk",), stride: int = 3,
         max_frames: int = 460, margin: float = 1.5) -> int:
    # DISJOINT dev/held-out loop sequences (the original calibration was train-on-test).
    def prov():
        return TUMRGBDProvider(dev=dev, heldout=heldout, stride=stride, max_frames=max_frames)

    def measure(label, author):
        h = build_vo_implementer_harness(root, task=vo_impl_task_slam(1e9, dev=dev, heldout=heldout),
                                         provider=prov(), author_fn=author, job_mode="docker")
        rec = h.run_experiment(ExperimentRecord(id=f"slam-cal-{label}", hypothesis=label))
        if not (rec.verdict and "ate_rmse" in rec.verdict.measured_metrics):
            print(f"{label} failed; log: {rec.log_path}"); return None
        return float(rec.verdict.measured_metrics["ate_rmse"])

    print(f"Calibrating SLAM: dev={dev} held-out={heldout} (loop seqs, stride={stride}, "
          f"{max_frames} frames)...")
    slam = measure("slam", slam_reference_author())          # reference SLAM (loop closure)
    if slam is None:
        return 1
    bar = round(slam * margin, 4)
    vo = measure("voonly", rgbd_reference_author())          # VO-only (should FAIL bar)
    deg = measure("degenerate", degenerate_author())         # static control (should FAIL)

    opened = slam <= bar and (vo is None or vo > bar) and (deg is None or deg > bar)
    print("=" * 66)
    print(f"reference SLAM (loop closure)  ATE = {slam:.4f} m   -> bar (x{margin}) = {bar} m")
    print(f"VO-only (no loop closure)      ATE = {vo:.4f} m   -> {'FAIL (good)' if vo and vo>bar else 'PASS?!'}")
    print(f"degenerate control             ATE = {deg:.4f} m   -> {'REJECTED' if deg and deg>bar else '?!'}")
    print(f"loop closure is NECESSARY to pass: {vo is not None and vo > bar}")
    print(f"CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print("=" * 66)
    if opened:
        print(f">>> live SLAM Track B:  python -m vo_lab.run_vo_tum_slam_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
