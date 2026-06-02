"""Track A offline: the committee -> menu -> independent-eval -> decide-next lineage runs
end-to-end with a FAKE run_fn (no API key). The run/eval pipeline is real (CPU ORB-VO on
synthetic data); only the LLM reasoning is stubbed — so this is a genuine integration test
of the autonomous loop's machinery and safety properties."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vo_lab  # noqa: E402,F401  -- bootstraps the ver2 (lab) path first
from lab.agents.sdk import AgentResult  # noqa: E402
from lab.models import ExperimentRecord, Status, Usage  # noqa: E402
from vo_lab.factory import build_vo_committee_harness  # noqa: E402
from vo_lab.plugins.vo import vo_calibration_records  # noqa: E402


def make_fake_committee(decide_calls):
    """Deterministic stand-in for live Claude sessions. Branches on the requested schema:
    proposal / expert-opinion / follow-up decision. Drives a 2-experiment lineage."""
    state = {"decide": 0}

    def fake(prompt, *, system_prompt=None, schema=None, model=None, max_turns=6, **kw):
        req = set((schema or {}).get("required", []))
        if "recipe_id" in req:                       # PROPOSAL_SCHEMA (PI draft)
            data = {"recipe_id": "orb-mono-vo", "params": {"nfeatures": 600},
                    "hypothesis": "baseline ORB-VO"}
        elif "approve" in req:                       # EXPERT_OPINION_SCHEMA
            data = {"approve": True, "param_overrides": {"nfeatures": 1500},
                    "concerns": [], "rationale": "more features for noisy frames"}
        elif "propose_followup" in req:              # DECIDE_SCHEMA
            state["decide"] += 1
            if state["decide"] <= decide_calls:
                data = {"propose_followup": True, "next_id": f"exp-{state['decide']+1}",
                        "hypothesis": "raise RANSAC threshold", "rationale": "explore"}
            else:
                data = {"propose_followup": False, "rationale": "diminishing returns"}
        else:
            data = {}
        return AgentResult(data=data, text="", usage=Usage(tokens_in=120, tokens_out=40))

    return fake


def test_committee_lineage_runs_and_verifies(tmp_path):
    h = build_vo_committee_harness(tmp_path / "run", job_mode="local",
                                   run_fn=make_fake_committee(decide_calls=1))
    # reproduction-first: gate must open before autonomy
    pos, neg = vo_calibration_records()
    assert h.calibration_gate(pos, neg) is True

    h.queue.push(ExperimentRecord(id="exp-1", hypothesis="committee VO exploration"))
    summary = h.loop(require_gate=True, goal_metric="vo_score", max_stall=None)

    assert summary["experiments_ran"] >= 2            # initial + at least one follow-up
    verified = h.registry.query(statuses=[Status.VERIFIED])
    assert any(r.id.startswith("exp-") for r in verified)
    # budget was charged in TOKENS (the fake reported usage), proving turn-free accounting
    assert h.budget.state.total_tokens > 0
    # every committee experiment used the vetted menu command (never an invented one)
    for r in verified:
        if r.id.startswith("exp-"):
            assert "$LAB_CODE/run.py" in r.contract.command
            assert r.verdict.signed_by == "evaluator-vo"


def test_committee_cannot_escape_menu(tmp_path):
    """Even if an expert 'proposes' an out-of-range/unknown param, the Menu clamps/drops it
    — no raw model-authored value reaches the shell."""
    def rogue(prompt, *, system_prompt=None, schema=None, model=None, max_turns=6, **kw):
        req = set((schema or {}).get("required", []))
        if "recipe_id" in req:
            return AgentResult(data={"recipe_id": "orb-mono-vo",
                                     "params": {"nfeatures": 999999, "evil": "rm -rf /"},
                                     "hypothesis": "x"}, text="", usage=Usage(1, 1))
        if "approve" in req:
            return AgentResult(data={"approve": True}, text="", usage=Usage(1, 1))
        return AgentResult(data={"propose_followup": False}, text="", usage=Usage(1, 1))

    h = build_vo_committee_harness(tmp_path / "run", job_mode="local", run_fn=rogue)
    contract, _ = h.planner.propose_contract(ExperimentRecord(id="x", hypothesis="x"))
    assert "evil" not in contract.command and "rm -rf" not in contract.command
    assert "LAB_NFEATURES=3000" in contract.command   # clamped to the recipe's high bound
