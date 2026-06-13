"""Track A — autonomous VO committee lineage on REAL TUM data (LIVE, billed):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_tum_committee

The validation of the AUTONOMY pillar: unlike the synthetic world (where the ORB params
barely move the metric), on real TUM frames the committee's knobs genuinely matter
(measured: held-out ATE ~0.13 -> ~0.089 m as nfeatures 600 -> 1500). So the autonomous
committee (PI + Geometry/SLAM + Data) runs a lineage that should MEASURABLY IMPROVE the
held-out metric over experiments — discovery, not just a safe loop.

Flow: measure the reference VO on real held-out to set the bar (local, non-billed) ->
reproduction-first calibration gate -> autonomous menu-constrained lineage, each experiment
independently verified on held-out. Budget = tokens + experiments; data/runs/eval are jobs.
Offline machinery test (no API): pytest tests/test_vo_committee.py."""
from __future__ import annotations

import os
import sys

from lab.models import ExperimentRecord, Status

from .factory import build_vo_committee_harness, build_vo_harness
from .plugins.vo import _contract, vo_calibration_records, vo_menu_real
from .plugins.vo_real import TUMDatasetProvider, tum_datasets


def _measure_reference_bar(root: str, provider, ds, margin: float) -> float:
    """Local, non-billed: run the reference ORB-VO (default params) on real held-out, derive
    the oracle bar = ATE x margin."""
    h = build_vo_harness(root, job_mode="local", provider=provider)
    h.run_experiment(ExperimentRecord(id="cal-measure", hypothesis="measure reference VO",
                                      contract=_contract(degenerate=False, threshold=1e9,
                                                         datasets=ds)))
    rec = h.registry.get("cal-measure")
    if not (rec.verdict and "ate_rmse" in rec.verdict.measured_metrics):
        raise RuntimeError(f"reference measurement failed; log: {rec.log_path}")
    return round(float(rec.verdict.measured_metrics["ate_rmse"]) * margin, 4)


def main(root: str = "./_vo_tum_committee_run", max_frames: int = 200, margin: float = 1.5,
         max_stall: int = 3) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("This live run needs ANTHROPIC_API_KEY (billed). Offline machinery test:\n"
              "  PYTHONPATH=. python -m pytest tests/test_vo_committee.py -q")
        return 2

    provider = TUMDatasetProvider(max_frames=max_frames)
    ds = tum_datasets()
    print("Measuring reference ORB-VO on real TUM held-out (local, non-billed; first run "
          "downloads ~0.5 GB)...")
    ate_bar = _measure_reference_bar(root, provider, ds, margin)
    score_bar = 1.0 / (1.0 + ate_bar * 2.0)   # generous recipe bar (loose x2) so reasonable
    #                                            configs pass and the lineage improves vo_score
    print(f"reference bar (gate) = {ate_bar} m ATE | recipe vo_score bar = {score_bar:.4f}\n")

    menu = vo_menu_real(score_bar=score_bar, datasets=ds)
    h = build_vo_committee_harness(root, job_mode="local", provider=provider, menu=menu)

    pos, neg = vo_calibration_records(threshold=ate_bar, datasets=ds)
    if not h.calibration_gate(pos, neg):
        print("CALIBRATION GATE LOCKED — refusing to start autonomy."); return 1
    print("CALIBRATION GATE: OPEN — starting committee lineage on real data.\n")

    h.queue.push(ExperimentRecord(
        id="vo-real-001",
        hypothesis="Minimize held-out monocular VO ATE on real TUM fr1/xyz by tuning the "
                   "ORB-VO recipe (feature count, RANSAC threshold, Lowe ratio)."))
    summary = h.loop(require_gate=True, goal_metric="vo_score", max_stall=max_stall)

    print("=" * 70)
    print(f"experiments ran: {summary['experiments_ran']} | best vo_score: {summary['best']}")
    print("\n--- improvement curve (lineage order) ---")
    verified = sorted(h.registry.query(statuses=[Status.VERIFIED]),
                      key=lambda r: r.created_at)
    for r in verified:
        mm = r.verdict.measured_metrics
        print(f"  {r.id}: ATE={mm.get('ate_rmse'):.4f} m  vo_score={mm.get('vo_score'):.4f}  "
              f"cmd=({r.contract.command})")
    print(f"\ntokens spent: {h.budget.state.total_tokens} | "
          f"io wall (uncharged): {h.budget.state.io_wall_seconds:.1f}s")
    print("=" * 70)
    if h.layout.notebook.exists():
        print("\n--- lab notebook (the meeting + the loop) ---")
        print(h.layout.notebook.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
