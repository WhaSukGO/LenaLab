"""The core property, offline: the VO calibration gate opens iff the honest solver is
VERIFIED on held-out AND the degenerate control is REJECTED. No Docker/GPU/API."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

import vo_lab  # noqa: E402,F401  -- MUST come first: bootstraps the ver2 (lab) path
from lab.models import Status  # noqa: E402
from vo_lab.factory import build_vo_harness  # noqa: E402
from vo_lab.plugins.vo import ATE_THRESHOLD, vo_calibration_records  # noqa: E402


def test_calibration_gate_opens(tmp_path):
    h = build_vo_harness(tmp_path / "run", job_mode="local")
    pos, neg = vo_calibration_records()
    opened = h.calibration_gate(pos, neg)

    pos_rec = h.registry.get("cal-pos")
    neg_rec = h.registry.get("cal-neg")

    assert pos_rec.status == Status.VERIFIED
    assert neg_rec.status == Status.REJECTED
    assert opened is True
    # honest VO is well under the bar; degenerate control is well over it (non-flaky margin)
    assert pos_rec.verdict.measured_metrics["ate_rmse"] < ATE_THRESHOLD
    assert neg_rec.verdict.measured_metrics["ate_rmse"] > ATE_THRESHOLD


def test_reported_metrics_not_trusted(tmp_path):
    """The verdict comes from the evaluator's held-out measurement, not the solver's
    self-report (the spine's whole point)."""
    h = build_vo_harness(tmp_path / "run", job_mode="local")
    pos, _ = vo_calibration_records()
    h.run_experiment(pos)
    rec = h.registry.get("cal-pos")
    assert rec.verdict.signed_by == "evaluator-vo"
    assert "ate_rmse" in rec.verdict.measured_metrics
