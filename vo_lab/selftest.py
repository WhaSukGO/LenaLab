"""End-to-end self-test for the VO lab — proves the spine + calibration gate offline.

  python -m vo_lab.selftest

No Docker, no GPU, no API key. Generates a tiny synthetic monocular sequence, runs the
honest ORB-VO (positive control -> evaluator measures held-out ATE -> VERIFIED) and a
degenerate static trajectory (negative control -> large ATE -> REJECTED). The autonomy
gate opens only if BOTH behave correctly — reproduction-first, exactly as ver2 demands."""
from __future__ import annotations

import sys

from lab.models import Status

from .factory import build_vo_harness
from .plugins.vo import ATE_THRESHOLD, vo_calibration_records


def main(root: str = "./_vo_selftest_run") -> int:
    h = build_vo_harness(root, job_mode="local")
    pos, neg = vo_calibration_records()
    opened = h.calibration_gate(pos, neg)

    pos_rec = h.registry.get("cal-pos")
    neg_rec = h.registry.get("cal-neg")
    print("=" * 64)
    print(f"oracle bar: ATE-RMSE (sim3) <= {ATE_THRESHOLD}")
    print(f"positive (honest ORB-VO):  status={pos_rec.status.value} "
          f"verdict={pos_rec.verdict.verdict} measured={pos_rec.verdict.measured_metrics}")
    print(f"negative (static control): status={neg_rec.status.value} "
          f"verdict={neg_rec.verdict.verdict} measured={neg_rec.verdict.measured_metrics}")
    print(f"  -> grader is not a rubber stamp: {neg_rec.verdict.verdict == 'FAIL'}")
    print("=" * 64)
    print(f"CALIBRATION GATE: {'OPEN (autonomy unlocked)' if opened else 'LOCKED'}")
    print(f"tokens spent: {h.budget.state.total_tokens} | "
          f"io wall seconds (uncharged): {h.budget.state.io_wall_seconds:.2f}")

    ok = (opened and pos_rec.status == Status.VERIFIED
          and neg_rec.status == Status.REJECTED)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
