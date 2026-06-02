"""Real-data calibration on TUM RGB-D (freiburg1_xyz) — runs LOCALLY (CPU, no Docker/API).

  python -m vo_lab.run_vo_tum_calibration            # downloads ~0.5 GB once, then CPU-only

Establishes the oracle bar from the classical baseline (we have no published number that
fits this simple VO): measure the reference ORB-VO's held-out ATE on the real sequence, set
the bar to a margin above it, then prove the gate discriminates on REAL data — reference VO
PASSES, the degenerate static control is REJECTED. Prints the bar to use for live Track B.

This is the 'baseline-beating' oracle from the design: a Track-B agent must roughly match or
beat the classical baseline on held-out real frames it never sees."""
from __future__ import annotations

import sys

from lab.models import ExperimentRecord, Status

from .factory import build_vo_harness
from .plugins.vo import _contract, vo_calibration_records
from .plugins.vo_real import TUMDatasetProvider, tum_datasets


def main(root: str = "./_vo_tum_run", max_frames: int = 200, margin: float = 1.5) -> int:
    provider = TUMDatasetProvider(max_frames=max_frames)
    h = build_vo_harness(root, job_mode="local", provider=provider)
    ds = tum_datasets()

    print("Measuring the reference ORB-VO on TUM fr1/xyz held-out (first run downloads ~0.5 GB)...")
    meas = ExperimentRecord(id="cal-measure", hypothesis="measure reference VO on TUM",
                            contract=_contract(degenerate=False, threshold=1e9, datasets=ds))
    h.run_experiment(meas)
    rec = h.registry.get("cal-measure")
    if not (rec.verdict and "ate_rmse" in rec.verdict.measured_metrics):
        print("measurement failed; see log:", rec.log_path)
        return 1
    ate_ref = float(rec.verdict.measured_metrics["ate_rmse"])
    bar = round(ate_ref * margin, 4)

    pos, neg = vo_calibration_records(threshold=bar, datasets=ds)
    opened = h.calibration_gate(pos, neg)
    pr, nr = h.registry.get("cal-pos"), h.registry.get("cal-neg")

    print("=" * 64)
    print(f"reference ORB-VO held-out ATE-RMSE (sim3) = {ate_ref:.4f} m")
    print(f"derived oracle bar (x{margin})            = {bar} m")
    print(f"positive (reference VO): {pr.status.value} "
          f"ate={pr.verdict.measured_metrics.get('ate_rmse'):.4f}")
    print(f"negative (degenerate):   {nr.status.value} "
          f"ate={nr.verdict.measured_metrics.get('ate_rmse'):.4f}")
    print(f"CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print("=" * 64)
    if opened:
        print(f">>> live Track B on real data:  python -m vo_lab.run_vo_tum_implement {bar}")
    else:
        print("Gate LOCKED: the reference VO and the degenerate control didn't separate at this "
              "bar (the sequence may be too hard for the simple ORB-VO). Inspect, then adjust.")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
