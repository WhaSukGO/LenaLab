"""SLAM Track B — LIVE sandboxed authoring (BILLED + Docker):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_tum_slam_implement <bar>

A sandboxed Claude agent authors RGB-D SLAM **with loop closure** for a long loop sequence
where VO-only drifts past the bar. Graded on held-out ATE (SE(3) metric). The bar comes from
run_vo_tum_slam_calibration."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.vo_implementer import resilient_sdk_author, vo_impl_task_slam
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.vo_rgbd import TUMRGBDProvider


def main(bar: float, root: str = "./_vo_slam_impl_run", model: str = "claude-sonnet-4-6",
         dev: str = "fr1_room", heldout: tuple[str, ...] = ("fr2_desk",),
         stride: int = 3, max_frames: int = 460) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2

    # DISJOINT dev/held-out (the original run was train-on-test; see trial-doc §12 correction).
    # Feed prior experience (verified approaches + failed attempts) into the author's prompt.
    task = inject_memory(vo_impl_task_slam(bar, dev=dev, heldout=heldout), "slam")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=TUMRGBDProvider(dev=dev, heldout=heldout,
                                                              stride=stride, max_frames=max_frames),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the Docker sandbox image (with scipy):\n  {e}\n"
              "docker build -f docker/Dockerfile.cpu-opencv -t vo-cpu-opencv:1 ."); return 2

    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=100)
    print(f"Implementer: sandboxed agent authoring RGB-D SLAM w/ loop closure "
          f"(bar={bar} m, dev={dev}, held-out={heldout})...\n")
    rec = h.run_experiment(ExperimentRecord(id="vo-slam-impl-001",
                                            hypothesis="implement RGB-D SLAM with loop closure"))
    print("=" * 64)
    print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
    print("  tokens:", h.budget.state.total_tokens, "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
        print("--- authored main.py (first 30 lines) ---")
        print("\n".join((Path(rec.contract.code_dir) / "main.py").read_text().splitlines()[:30]))
    print("=" * 64)
    # Close the loop: a rejected/failed attempt becomes memory for the next session.
    art = None
    if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
        art = str(Path(rec.contract.code_dir) / "main.py")
    entry = record_from_experiment("slam", rec, artifact=art)
    if entry:
        print(f"  recorded outcome to cross-run memory (domain=slam, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_vo_tum_slam_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
