"""Track B — LIVE sandboxed VO authoring (BILLED + Docker):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_implement

A sandboxed Claude session WRITES main.py and tests it via a container-only `run` tool
(no host shell, no network, writes confined to the code dir, eval.py off-limits), iterating
until it produces a trajectory. Then the unchanged independent ScriptEvaluator runs it and
measures held-out ATE-RMSE against the fixed oracle — the agent never saw the held-out GT,
and cannot edit the grader.

Requires: (1) ANTHROPIC_API_KEY, (2) a Docker image with numpy+opencv whose registry key
contains "cpu" (see images/registry.yaml). Without those, run the offline proof instead:
  PYTHONPATH=. python -m pytest tests/test_vo_implementer.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.vo_implementer import resilient_sdk_author, vo_impl_task
from .factory import build_vo_implementer_harness


def main(root: str = "./_vo_implement_run", model: str = "claude-sonnet-4-6") -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed). Offline proof:\n"
              "  PYTHONPATH=. python -m pytest tests/test_vo_implementer.py -q")
        return 2

    h = build_vo_implementer_harness(root, job_mode="docker")
    try:
        h.image_registry.resolve(vo_impl_task().framework)
    except NoImageError as e:
        print(f"Live authoring runs in a Docker sandbox but no image is configured:\n  {e}\n"
              "Add a numpy+opencv image (key containing 'cpu') to images/registry.yaml, or "
              "run the offline proof:\n  PYTHONPATH=. python -m pytest tests/test_vo_implementer.py -q")
        return 2

    # wire the real sandboxed author from the harness's own components
    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=40)
    print("Implementer: sandboxed agent authoring + testing monocular VO (no host shell)...\n")
    rec = h.run_experiment(ExperimentRecord(id="vo-impl-001",
                                            hypothesis="implement monocular visual odometry"))

    print("=" * 64)
    print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics,
              "| verdict:", rec.verdict.verdict)
        print("  evaluator notes:", rec.verdict.evaluator_notes[-200:])
    print("  tokens:", h.budget.state.total_tokens,
          "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    if rec.contract:
        main_py = Path(rec.contract.code_dir) / "main.py"
        if main_py.exists():
            print("--- authored main.py (first 30 lines) ---")
            print("\n".join(main_py.read_text().splitlines()[:30]))
    print("=" * 64)
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    sys.exit(main())
