"""Track A — autonomous VO committee lineage (LIVE, billed):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_committee

Opens the reproduction-first calibration gate (offline, no API), then runs an autonomous
lineage where the VO expert committee (PI + Geometry/SLAM + Data) proposes menu-constrained
experiments, each independently verified on the held-out split. The committee can only
select + clamp the vetted recipe's params — it cannot invent commands. Budget is measured
in tokens + experiments; downloads/runs/eval are harness jobs (turn-free).

NOTE: on the easy synthetic world the ORB params barely move the metric — this run
demonstrates the LOOP MACHINERY + safety, not an improvement curve. Genuine algorithm
authoring is Track B (the Implementer)."""
from __future__ import annotations

import os
import sys

from lab.models import ExperimentRecord, Status

from .factory import build_vo_committee_harness
from .plugins.vo import vo_calibration_records


def main(root: str = "./_vo_committee_run", max_stall: int = 2) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("This live run needs ANTHROPIC_API_KEY (billed). For the offline machinery "
              "test, run:  PYTHONPATH=. python -m pytest tests/test_vo_committee.py -q")
        return 2

    h = build_vo_committee_harness(root, job_mode="local")
    pos, neg = vo_calibration_records()
    if not h.calibration_gate(pos, neg):
        print("CALIBRATION GATE LOCKED — refusing to start autonomy.")
        return 1
    print("CALIBRATION GATE: OPEN — starting committee lineage.\n")

    h.queue.push(ExperimentRecord(id="vo-001",
                                  hypothesis="Minimize held-out monocular VO trajectory "
                                             "error by tuning the ORB-VO recipe."))
    summary = h.loop(require_gate=True, goal_metric="vo_score", max_stall=max_stall)

    print("=" * 64)
    print(f"experiments ran: {summary['experiments_ran']} | "
          f"best vo_score: {summary['best']}")
    for r in h.registry.query(statuses=[Status.VERIFIED]):
        print(f"  {r.id}: VERIFIED {r.verdict.measured_metrics} cmd=({r.contract.command})")
    print(f"tokens spent: {h.budget.state.total_tokens} | "
          f"io wall (uncharged): {h.budget.state.io_wall_seconds:.1f}s")
    print("=" * 64)
    # Make the "meeting" tangible: show the committee's deliberation + state machine.
    if h.layout.notebook.exists():
        print("\n--- lab notebook (the meeting + the loop) ---")
        print(h.layout.notebook.read_text())
    print("NOTE: on this easy synthetic world the metric stays flat — you are seeing the "
          "live autonomous loop + independent verification work, not an improvement curve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
