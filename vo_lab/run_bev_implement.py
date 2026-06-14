"""BEV Track B — LIVE sandboxed authoring on the GPU (BILLED + Docker + CUDA):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_bev_implement <bar>

The lab's SECOND problem class: a sandboxed Claude agent authors a multi-camera Bird's-Eye-View
vehicle-occupancy network (Lift-Splat style) that TRAINS ON THE GPU, graded on held-out nuScenes
mini_val scenes by IoU. Proves the verification-first harness generalizes beyond ego-motion (VO/
SLAM) to surround-view perception. Prior experience for the 'bev' domain is injected; the bar comes
from the from-scratch reference (run_bev_calibration)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.bev_implementer import bev_impl_task
from .agents.vo_implementer import resilient_sdk_author
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.bev_nuscenes import NuScenesBEVProvider


def main(bar: float, root: str = "./_bev_impl_run", model: str = "claude-sonnet-4-6",
         train_max: int | None = None, test_max: int | None = None) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2

    task = inject_memory(bev_impl_task(bar, train_max=train_max, test_max=test_max), "bev")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=NuScenesBEVProvider(train_max=train_max, test_max=test_max),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the GPU image:\n  {e}\n"
              "docker build -f docker/Dockerfile.gpu-torch -t vo-gpu-torch:1 ."); return 2

    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=100)
    print(f"Implementer: sandboxed agent authoring a GPU-trained BEV network (bar IoU={bar})...\n")
    rec = h.run_experiment(ExperimentRecord(id="bev-impl-001",
                                            hypothesis="implement a GPU-trained surround-camera BEV network"))
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
    if record_from_experiment("bev", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=bev, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_bev_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
