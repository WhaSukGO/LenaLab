"""Track B on REAL data — LIVE sandboxed VO authoring on TUM fr1/xyz (BILLED + Docker):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_tum_implement <bar>

where <bar> is the held-out ATE bar printed by `run_vo_tum_calibration` (the reference
baseline x margin). A sandboxed Claude session writes main.py and tests it in the container
(no host shell, no network, eval.py off-limits); the independent ScriptEvaluator then runs
it on the held-out TUM ground truth and checks ATE-RMSE <= bar. The agent never sees the GT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.vo_implementer import resilient_sdk_author, vo_impl_task_real
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.vo_real import TUMDatasetProvider


def main(bar: float, root: str = "./_vo_tum_impl_run", model: str = "claude-sonnet-4-6",
         max_frames: int = 200) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed).")
        return 2

    task = inject_memory(vo_impl_task_real(bar), "vo-mono")
    h = build_vo_implementer_harness(root, task=task, provider=TUMDatasetProvider(max_frames=max_frames),
                                     job_mode="docker")
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Live authoring needs the Docker sandbox image:\n  {e}\n"
              "Build it: docker build -f docker/Dockerfile.cpu-opencv -t vo-cpu-opencv:1 .")
        return 2

    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=80)
    print(f"Implementer: sandboxed agent authoring monocular VO for TUM fr1/xyz (bar={bar} m)...\n")
    rec = h.run_experiment(ExperimentRecord(id="vo-tum-impl-001",
                                            hypothesis="implement monocular VO on real TUM data"))

    print("=" * 64)
    print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
        print("  evaluator notes:", rec.verdict.evaluator_notes[-200:])
    print("  tokens:", h.budget.state.total_tokens,
          "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    if rec.contract:
        mp = Path(rec.contract.code_dir) / "main.py"
        if mp.exists():
            print("--- authored main.py (first 30 lines) ---")
            print("\n".join(mp.read_text().splitlines()[:30]))
    print("=" * 64)
    art = None
    if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
        art = str(Path(rec.contract.code_dir) / "main.py")
    if record_from_experiment("vo-mono", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=vo-mono, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_vo_tum_implement <bar>   "
              "(get <bar> from run_vo_tum_calibration)")
        sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
