"""One-off: backfill the live SLAM attempt that left NO structured record.

The live SLAM Track-B run was REJECTED (its pose graph diverged to ~412 m) but, because the
session errored out under a watchdog misconfiguration, the harness never persisted a registry
row, notebook line, or failed-approaches entry. The only trace was prose in
`claudedocs/trial_track_b_tum_2026-06-02.md` §11. This script reconstructs the record from
the known facts + the archived artifact so the structured store matches reality.

The measured value (412.7 m) is REPRODUCED from the archived artifact on the held-out
sequence (the live run reported ~412 m but recorded no number). Provenance is stated in the
evaluator notes — this is a documented backfill, not a fabricated live result.

    PYTHONPATH=. python -m vo_lab.backfill_slam
"""
from __future__ import annotations

from pathlib import Path

import vo_lab  # noqa: F401  (bootstraps `import lab`)
from lab.models import ExperimentRecord, Status, VerifiedResult
from lab.notebook import Notebook
from lab.paths import Layout
from lab.registry import Registry

from .memory import load_failures, record_failure

ROOT = "./_vo_slam_impl_run"
EXP_ID = "vo-slam-impl-001"
ARTIFACT = "artifacts/agent_authored_vo_slam_v1.py"

MEASURED = 412.7        # reproduced from the archived artifact on held-out fr1_room
BAR = 0.347             # SLAM gate bar (run_vo_tum_slam_calibration)
CREATED_AT = "2026-06-02T20:49:00+00:00"   # artifact archive time; original run left no record

NOTES = (
    "Agent authored a 352-line RGB-D SLAM (SIFT matching + loop detection + pose graph). "
    "The pose-graph optimization DIVERGED (catastrophic blow-up on a ~3 m-scale trajectory) "
    "-> held-out ATE 412.7 m, far past the 0.347 m bar. The verifier correctly REJECTED it "
    "(nothing false accepted). The live run was compounded by a watchdog misconfig "
    "(480 s kill < ~507 s in-container run < 600 s grader), so in-container tests were killed "
    "before completing and the agent never saw its own output to debug the divergence; "
    "watchdog since corrected to 900 s. BACKFILLED 2026-06-03: the original run left no "
    "registry/notebook record; measured value reproduced from the archived artifact "
    f"({ARTIFACT})."
)


def main() -> int:
    layout = Layout(Path(ROOT))
    layout.state.mkdir(parents=True, exist_ok=True)

    oracle = {"metric": "ate_rmse", "op": "<=", "expected": BAR, "tolerance": 0.0,
              "measured": MEASURED, "within": False}
    verdict = VerifiedResult(
        experiment_id=EXP_ID, verdict="FAIL",
        measured_metrics={"ate_rmse": MEASURED, "align": "se3", "n_seqs": 1},
        oracle_comparison=oracle, evaluator_notes=NOTES,
        signed_by="backfill (reproduced from artifact)", signed_at="2026-06-03T00:00:00+00:00")
    rec = ExperimentRecord(
        id=EXP_ID, hypothesis="implement RGB-D SLAM with loop closure",
        status=Status.REJECTED, env_image="vo-cpu-opencv:1",
        datasets=["vo-rgbd-dev-fr1_room", "vo-rgbd-heldout-fr1_room"],
        verdict=verdict, created_at=CREATED_AT)

    reg = Registry(layout.registry_db)
    reg.upsert(rec)
    reg.close()

    nb = Notebook(notebook_path=layout.notebook, failed_path=layout.failed)
    nb.log_event(rec, "REJECTED (backfilled) — pose graph diverged, ATE 412.7 m")
    nb.log_failed(rec, f"evaluator FAIL: {oracle}  [backfilled 2026-06-03]")

    # Seed the cross-run failure memory (guard against duplicate appends on re-run).
    already = any(f.get("exp_id") == EXP_ID for f in load_failures("slam"))
    if not already:
        record_failure(
            "slam", exp_id=EXP_ID,
            what="RGB-D SLAM with loop closure (SIFT + loop detection + pose graph)",
            failure_mode="pose-graph optimization DIVERGED — ATE blew up to 412.7 m on a "
                         "~3 m-scale trajectory (got the structure right, optimizer unstable)",
            metric="ate_rmse", measured=MEASURED, bar=BAR, op="<=",
            fix="constrain/condition the pose-graph solve: robust kernel (Huber), good "
                "initialization from the VO front-end, gauge-fix the first pose, bound step "
                "sizes / use Levenberg-Marquardt damping; sanity-check loop constraints before "
                "adding them. Verify the optimizer REDUCES error vs VO-only before trusting it.",
            artifact=ARTIFACT)

    # Verify
    back = Registry(layout.registry_db).get(EXP_ID)
    print(f"registry: {EXP_ID} -> status={back.status.value}, "
          f"ate={back.verdict.measured_metrics['ate_rmse']} m, verdict={back.verdict.verdict}")
    print(f"notebook: {layout.notebook}")
    print(f"failed:   {layout.failed}")
    print(f"memory:   {len(load_failures('slam'))} slam failure(s) in cross-run ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
