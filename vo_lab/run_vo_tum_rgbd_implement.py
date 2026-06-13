"""RGB-D Track B — LIVE sandboxed authoring (BILLED + Docker):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_tum_rgbd_implement <bar>

A sandboxed Claude agent authors an RGB-D VO (using the depth channel for metric scale).
The independent grader runs it on MULTIPLE held-out sequences it never saw, SE(3)-metric
ATE/RPE, and checks mean ATE <= bar (from run_vo_tum_rgbd_calibration)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.agents.implementer import sdk_author
from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.vo_implementer import resilient_sdk_author, vo_impl_task_rgbd
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.vo_rgbd import TUMRGBDProvider


def main(bar: float, root: str = "./_vo_rgbd_impl_run", model: str = "claude-sonnet-4-6",
         dev: str = "fr1_xyz", heldout: tuple[str, ...] = ("fr1_desk",),
         max_frames: int = 200) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2

    task = inject_memory(vo_impl_task_rgbd(bar, dev=dev, heldout=heldout), "vo-rgbd")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=TUMRGBDProvider(dev=dev, heldout=heldout,
                                                              max_frames=max_frames),
                                     job_mode="docker")
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the Docker sandbox image:\n  {e}\n"
              "docker build -f docker/Dockerfile.cpu-opencv -t vo-cpu-opencv:1 ."); return 2

    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=80)
    print(f"Implementer: sandboxed agent authoring RGB-D VO (bar={bar} m, held-out={heldout})...\n")
    rec = h.run_experiment(ExperimentRecord(id="vo-rgbd-impl-001",
                                            hypothesis="implement RGB-D VO on real TUM data"))
    print("=" * 64)
    print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
    print("  tokens:", h.budget.state.total_tokens, "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
        print("--- authored main.py (first 30 lines) ---")
        print("\n".join((Path(rec.contract.code_dir) / "main.py").read_text().splitlines()[:30]))
    print("=" * 64)
    art = None
    if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
        art = str(Path(rec.contract.code_dir) / "main.py")
    if record_from_experiment("vo-rgbd", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=vo-rgbd, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_vo_tum_rgbd_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
