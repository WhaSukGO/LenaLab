"""BEV SCAFFOLD Track B — LIVE sandboxed authoring (BILLED + Docker + GPU):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_bev_scaffold_implement <bar>

Tests the n=3 finding's prescription: the locked bev_core.py (geometry + correct flip augmentation
+ training + calibration) is seeded into the agent's workspace; the agent authors ONLY model.py
(the network). If locking the fragile parts collapses the variance the free-form runs showed
(0.085 +/- 0.034, 2/3), that confirms the variance was the agent's design latitude, not the task."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.bev_implementer import bev_impl_task_scaffold, bev_scaffold_seed, seeded
from .agents.vo_implementer import resilient_sdk_author
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.bev_nuscenes import NuScenesBEVProvider


def main(bar: float, root: str = "./_bev_scaffold_impl_run", model: str = "claude-sonnet-4-6",
         train_max=None, test_max=None) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2

    task = inject_memory(bev_impl_task_scaffold(bar, train_max=train_max, test_max=test_max), "bev-scaffold")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=NuScenesBEVProvider(train_max=train_max, test_max=test_max),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the GPU image:\n  {e}"); return 2

    h.planner.author_fn = seeded(
        resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache, model=model, max_turns=100),
        bev_scaffold_seed())
    print(f"Implementer: sandboxed agent authoring ONLY the BEV network (geometry+aug locked, bar IoU={bar})...\n")
    rec = h.run_experiment(ExperimentRecord(id="bev-scaffold-impl-001",
                                            hypothesis="author the BEV network behind a locked Lift-Splat scaffold"))
    print("=" * 64)
    print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
    print("  tokens:", h.budget.state.total_tokens, "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    if rec.contract and (Path(rec.contract.code_dir) / "model.py").exists():
        print("--- authored model.py (first 25 lines) ---")
        print("\n".join((Path(rec.contract.code_dir) / "model.py").read_text().splitlines()[:25]))
    print("=" * 64)
    art = str(Path(rec.contract.code_dir) / "model.py") if rec.contract else None
    if record_from_experiment("bev-scaffold", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=bev-scaffold, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_bev_scaffold_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
