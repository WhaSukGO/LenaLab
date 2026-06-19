"""Occupancy Track B — LIVE sandboxed authoring (BILLED + Docker + GPU):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_occ_implement <bar>

The 6th domain: a sandboxed agent authors a camera->3D-occupancy network, graded on held-out
nuScenes mini_val by voxel IoU. Tests whether the harness generalizes to 3D, and (with n=3) whether
the BEV free-form-variance finding replicates on a harder task."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.occ_implementer import occ_impl_task
from .agents.vo_implementer import resilient_sdk_author
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.occ_nuscenes import NuScenesOccProvider


def main(bar: float, root: str = "./_occ_impl_run", model: str = "claude-sonnet-4-6",
         train_max=None, test_max=None) -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2
    task = inject_memory(occ_impl_task(bar, train_max=train_max, test_max=test_max), "occ")
    h = build_vo_implementer_harness(root, task=task,
                                     provider=NuScenesOccProvider(train_max=train_max, test_max=test_max),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the GPU image:\n  {e}"); return 2
    h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                               model=model, max_turns=100)
    print(f"Implementer: sandboxed agent authoring a GPU-trained 3D-occupancy network (bar IoU={bar})...\n")
    rec = h.run_experiment(ExperimentRecord(id="occ-impl-001",
                                            hypothesis="implement a camera->3D-occupancy network"))
    print("=" * 64); print("RESULT:", rec.status.value)
    if rec.verdict:
        print("  measured (held-out):", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
    print("  tokens:", h.budget.state.total_tokens, "| io_wall_s:", round(h.budget.state.io_wall_seconds, 1))
    if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
        print("--- authored main.py (first 25 lines) ---")
        print("\n".join((Path(rec.contract.code_dir) / "main.py").read_text().splitlines()[:25]))
    print("=" * 64)
    art = str(Path(rec.contract.code_dir) / "main.py") if rec.contract else None
    if record_from_experiment("occ", rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain=occ, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_occ_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
