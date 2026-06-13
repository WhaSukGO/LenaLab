"""KITTI stereo Track B — LIVE sandboxed authoring (BILLED + Docker):

  ANTHROPIC_API_KEY=... python -m vo_lab.run_vo_kitti_implement <bar>

A sandboxed Claude agent authors a STEREO VO for outdoor driving (a domain it has never seen
— all prior tracks were indoor TUM). Graded on held-out KITTI sequences, SE(3) metric. Prior
experience (verified approaches + failures) for the 'kitti' domain is injected into its prompt.
The bar comes from run_vo_kitti_calibration."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from lab.image_registry import NoImageError
from lab.models import ExperimentRecord, Status

from .agents.vo_implementer import resilient_sdk_author, vo_impl_task_kitti
from .factory import build_vo_implementer_harness
from .memory import inject_memory, record_from_experiment
from .plugins.vo_kitti import KITTIOdomProvider


def main(bar: float, root: str = "./_vo_kitti_impl_run", model: str = "claude-sonnet-4-6",
         dev: str = "00", heldout: tuple[str, ...] = ("05", "07"),
         stride: int = 3, max_frames: int = 300, task_factory=vo_impl_task_kitti,
         seed_files: dict[str, str] | None = None, provider=None, domain: str = "kitti") -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.path.exists(".env")):
        print("Live Track B needs ANTHROPIC_API_KEY (billed)."); return 2

    task = inject_memory(task_factory(bar, dev=dev, heldout=heldout), domain)
    if provider is None:
        provider = KITTIOdomProvider(dev=dev, heldout=heldout, stride=stride, max_frames=max_frames)
    h = build_vo_implementer_harness(root, task=task, provider=provider,
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"Need the Docker sandbox image:\n  {e}\n"
              "docker build -f docker/Dockerfile.cpu-opencv -t vo-cpu-opencv:1 ."); return 2

    _inner_author = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                         model=model, max_turns=100)
    if seed_files:
        def _seeded_author(task, code_dir, rec):  # seed locked scaffold files before authoring
            from pathlib import Path as _P
            for name, src in seed_files.items():
                (_P(code_dir) / name).write_text(src)
            return _inner_author(task, code_dir, rec)
        h.planner.author_fn = _seeded_author
    else:
        h.planner.author_fn = _inner_author
    print(f"Implementer: sandboxed agent authoring KITTI stereo VO (bar={bar}% t_err, "
          f"dev={dev}, held-out={heldout})...\n")
    rec = h.run_experiment(ExperimentRecord(id="vo-kitti-impl-001",
                                            hypothesis="implement KITTI stereo VO (outdoor driving)"))
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
    if record_from_experiment(domain, rec, artifact=art):
        print(f"  recorded outcome to cross-run memory (domain={domain}, exp={rec.id})")
    return 0 if rec.status == Status.VERIFIED else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m vo_lab.run_vo_kitti_implement <bar>"); sys.exit(2)
    sys.exit(main(float(sys.argv[1])))
