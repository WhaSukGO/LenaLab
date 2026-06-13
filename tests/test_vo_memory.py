"""Offline coverage for the cross-run experience-memory loop (no API, no Docker)."""
from __future__ import annotations

import vo_lab  # noqa: F401  (bootstraps `import lab`)
from lab.models import ExperimentRecord, Status, VerifiedResult

from vo_lab.agents.vo_implementer import vo_impl_task_slam
from vo_lab.memory import (failure_context, inject_failure_memory, inject_memory,
                           load_failures, load_successes, record_failure,
                           record_from_experiment, success_context)


def test_record_and_recall_roundtrip(tmp_path):
    record_failure("slam", exp_id="e1", what="pose graph SLAM",
                   failure_mode="diverged", metric="ate_rmse", measured=412.7, bar=0.347,
                   store=tmp_path)
    fails = load_failures("slam", store=tmp_path)
    assert len(fails) == 1 and fails[0]["measured"] == 412.7
    # other domains are isolated
    assert load_failures("vo-rgbd", store=tmp_path) == []


def test_failure_context_formats_recent(tmp_path):
    for i in range(7):
        record_failure("slam", exp_id=f"e{i}", what=f"try {i}", failure_mode="diverged",
                       metric="ate_rmse", measured=400 + i, bar=0.347, store=tmp_path)
    ctx = failure_context("slam", limit=3, store=tmp_path)
    assert "PRIOR FAILED ATTEMPTS" in ctx
    assert "e6" in ctx and "e5" in ctx and "e4" in ctx   # last 3
    assert "e0" not in ctx                                # trimmed
    assert failure_context("nope", store=tmp_path) == ""


def test_inject_into_task_description_is_nonmutating(tmp_path):
    task = vo_impl_task_slam(0.347)
    base = task.description
    record_failure("slam", exp_id="e1", what="pose graph SLAM", failure_mode="optimizer diverged",
                   metric="ate_rmse", measured=412.7, bar=0.347, store=tmp_path)
    injected = inject_failure_memory(task, "slam", store=tmp_path)
    assert injected is not task
    assert task.description == base                      # original untouched
    assert base in injected.description                  # task preserved
    assert "optimizer diverged" in injected.description  # memory prepended
    # no memory -> task returned unchanged
    assert inject_failure_memory(task, "empty-domain", store=tmp_path).description == base


def test_record_from_experiment_routes_by_verdict(tmp_path):
    # non-terminal status records nothing
    proposed = ExperimentRecord(id="p", hypothesis="h", status=Status.PROPOSED)
    assert record_from_experiment("slam", proposed, store=tmp_path) is None

    v = VerifiedResult(experiment_id="bad", verdict="FAIL",
                       measured_metrics={"ate_rmse": 412.7},
                       oracle_comparison={"metric": "ate_rmse", "op": "<=", "expected": 0.347,
                                          "measured": 412.7, "within": False},
                       evaluator_notes="pose graph diverged")
    bad = ExperimentRecord(id="bad", hypothesis="slam", status=Status.REJECTED, verdict=v)
    entry = record_from_experiment("slam", bad, store=tmp_path)
    assert entry and entry["measured"] == 412.7 and entry["bar"] == 0.347
    assert "diverged" in failure_context("slam", store=tmp_path)


def test_verified_records_success_with_approach(tmp_path):
    art = tmp_path / "main.py"
    art.write_text('"""Linear translation-only pose graph, guaranteed no divergence."""\n'
                   "import numpy as np\n")
    v = VerifiedResult(experiment_id="win", verdict="PASS",
                       measured_metrics={"ate_rmse": 0.185},
                       oracle_comparison={"metric": "ate_rmse", "op": "<=", "expected": 0.347,
                                          "measured": 0.185, "within": True})
    ok = ExperimentRecord(id="win", hypothesis="slam", status=Status.VERIFIED, verdict=v)
    entry = record_from_experiment("slam", ok, artifact=str(art), store=tmp_path)
    assert entry and entry["measured"] == 0.185
    assert "guaranteed no divergence" in entry["approach"]            # docstring auto-extracted
    assert load_failures("slam", store=tmp_path) == []               # not a failure
    assert "VERIFIED at 0.185" in success_context("slam", store=tmp_path)


def test_inject_memory_includes_successes_and_failures(tmp_path):
    task = vo_impl_task_slam(0.347)
    base = task.description
    record_failure("slam", exp_id="f1", what="nonlinear pose graph", failure_mode="diverged",
                   metric="ate_rmse", measured=412.7, bar=0.347, store=tmp_path)
    art = tmp_path / "main.py"
    art.write_text('"""Linear pose graph; VO-only fallback."""\n')
    v = VerifiedResult(experiment_id="s1", verdict="PASS", measured_metrics={"ate_rmse": 0.185},
                       oracle_comparison={"metric": "ate_rmse", "op": "<=", "expected": 0.347,
                                          "measured": 0.185, "within": True})
    record_from_experiment("slam", ExperimentRecord(id="s1", hypothesis="slam",
                           status=Status.VERIFIED, verdict=v), artifact=str(art), store=tmp_path)
    injected = inject_memory(task, "slam", store=tmp_path)
    assert injected is not task and task.description == base          # non-mutating
    assert "PRIOR VERIFIED APPROACHES" in injected.description
    assert "PRIOR FAILED ATTEMPTS" in injected.description
    assert "VO-only fallback" in injected.description                # success approach present
    assert "diverged" in injected.description                        # failure present
    # back-compat: failure-only injector omits successes
    fonly = inject_failure_memory(task, "slam", store=tmp_path)
    assert "PRIOR FAILED ATTEMPTS" in fonly.description
    assert "PRIOR VERIFIED APPROACHES" not in fonly.description
