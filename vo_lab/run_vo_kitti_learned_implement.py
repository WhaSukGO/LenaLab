"""Learned-VO Track B — LIVE sandboxed authoring on the GPU (BILLED + Docker + CUDA):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_kitti_learned_implement <bar>

The faithful learned-VO analogue of the other tracks: a sandboxed Claude agent authors a
PyTorch learned VO that TRAINS ON THE GPU (a new kind of authoring — ML/training code, not
just classical geometry), graded on held-out KITTI sequences (Sim(3) ATE). Prior experience
for the 'learned-vo' domain is injected. The bar comes from run_vo_kitti_learned_calibration."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.vo_implementer import resilient_sdk_author, vo_impl_task_kitti_learned
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.vo_kitti_learned import KITTILearnedProvider


def main(bar: float, root: str = "./_vo_kitti_learned_impl_run", model: str = "claude-sonnet-4-6",
         train: tuple[str, ...] = ("00", "02", "06", "08", "09"),
         test: tuple[str, ...] = ("05", "07"), train_max: int = 1000, test_max: int = 300) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2

    task = inject_memory(vo_impl_task_kitti_learned(bar, train, test), "learned-vo")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=KITTILearnedProvider(train=train, test=test,
                                                                   train_max=train_max, test_max=test_max),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the GPU image:\n  {e}\n"
              "docker build -f docker/Dockerfile.gpu-torch -t vo-gpu-torch:1 ."); return 2

    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=100)
    print(f"Implementer: sandboxed agent authoring a GPU-trained learned VO (bar={bar} m)...\n")
    rec = h.run_experiment(ExperimentRecord(id="vo-learned-impl-001",
                                            hypothesis="implement a GPU-trained learned monocular VO"))
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
    if record_from_experiment("learned-vo", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=learned-vo, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_vo_kitti_learned_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
