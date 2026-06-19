"""Occupancy SCAFFOLD Track B — LIVE sandboxed authoring (BILLED + Docker + GPU):
  ANTHROPIC_API_KEY=... python -m vo_lab.run_occ_scaffold_implement <bar>
Locked 3D geometry+aug+training (seeded occ_core.py); the agent authors only model.py. Tests whether
the BEV scaffold-fix replicates on the harder 3D task."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.occ_implementer import occ_impl_task_scaffold, occ_scaffold_seed, seeded
from .agents.vo_implementer import resilient_sdk_author
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.occ_nuscenes import NuScenesOccProvider


def main(bar: float, root: str = "./_occ_scaffold_impl_run", model: str = "claude-sonnet-4-6",
         train_max=None, test_max=None) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2
    task = inject_memory(occ_impl_task_scaffold(bar, train_max=train_max, test_max=test_max), "occ-scaffold")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=NuScenesOccProvider(train_max=train_max, test_max=test_max),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the GPU image:\n  {e}"); return 2
    h.planner.author_fn = seeded(
        resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache, model=model, max_turns=100),
        occ_scaffold_seed())
    print(f"Implementer: sandboxed agent authoring ONLY the occupancy network (geometry+aug locked, bar IoU={bar})...\n")
    rec = h.run_experiment(ExperimentRecord(id="occ-scaffold-impl-001",
                                            hypothesis="author the occupancy network behind a locked 3D scaffold"))
    print("=" * 64); print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
    print("  tokens:", h.budget.state.total_tokens, "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    art = str(Path(rec.contract.code_dir) / "model.py") if rec.contract else None
    if record_from_experiment("occ-scaffold", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=occ-scaffold, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_occ_scaffold_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
