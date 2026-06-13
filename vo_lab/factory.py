"""Assembly — wires ver2's Harness spine with the VO domain plugin.

Mirrors ver2's `build_*_harness` factories. The MVP runs in LOCAL mode (no Docker/GPU):
classical ORB VO is CPU-only, so the entire spine + calibration gate is provable offline.
A `job_mode="docker"` path (with a CUDA image row) is the later, learned-VO upgrade."""
from __future__ import annotations

from pathlib import Path

import vo_lab  # noqa: F401  -- triggers the ver2 (lab) path bootstrap

from lab.budget import Budget
from lab.dataset_cache import DatasetCache
from lab.evaluator import ScriptEvaluator
from lab.gpu_lease import GpuLease
from lab.image_registry import ImageRegistry
from lab.job_runner import JobRunner
from lab.loop import Harness
from lab.models import Usage
from lab.notebook import Notebook
from lab.paths import Layout
from lab.queue import Queue
from lab.registry import Registry
from lab.util import ensure_dir

from .plugins.vo import VODatasetProvider, VOMetricExtractor

_IMAGES = str((Path(__file__).parent.parent / "images" / "registry.yaml").resolve())


class _NoPlanner:
    """The calibration path supplies contracts directly, so the planner is never called.
    (Track A's committee planner is wired in a later increment.)"""
    def propose_contract(self, rec):
        raise RuntimeError("this harness expects pre-supplied contracts (calibration)")

    def decide_next(self, result, rec):
        return None, Usage()


def build_vo_harness(root: str | Path, *, job_mode: str = "local",
                     images_path: str | Path = _IMAGES,
                     max_total_tokens: int = 1_000_000, max_experiments: int = 100,
                     lease_timeout_s: float = 900.0, planner=None,
                     provider=None, seed: int = 1234) -> Harness:
    layout = Layout(Path(root))
    ensure_dir(layout.state)
    registry = Registry(layout.registry_db)
    queue = Queue(registry)
    gpu_lease = GpuLease(layout.gpu_lock)
    image_registry = ImageRegistry(images_path)
    dataset_cache = DatasetCache(layout.cache, provider or VODatasetProvider(seed=seed))
    job_runner = JobRunner(default_mode=job_mode)
    budget = Budget(max_total_tokens=max_total_tokens, max_experiments=max_experiments,
                    state_path=layout.budget_state)
    notebook = Notebook(notebook_path=layout.notebook, failed_path=layout.failed)
    evaluator = ScriptEvaluator(layout, job_runner, dataset_cache, image_registry,
                                mode=job_mode, session_id="evaluator-vo")
    return Harness(
        layout=layout, registry=registry, queue=queue, gpu_lease=gpu_lease,
        image_registry=image_registry, dataset_cache=dataset_cache,
        job_runner=job_runner, budget=budget, notebook=notebook,
        planner=planner or _NoPlanner(), evaluator=evaluator,
        metric_extractor=VOMetricExtractor(), job_mode=job_mode,
        lease_timeout_s=lease_timeout_s,
    )


def build_vo_committee_harness(root: str | Path, *, job_mode: str = "local",
                               images_path: str | Path = _IMAGES, model: str | None = None,
                               run_fn=None, max_total_tokens: int = 4_000_000,
                               max_experiments: int = 50, lease_timeout_s: float = 900.0,
                               seed: int = 1234, provider=None, menu=None) -> Harness:
    """Track A: experiments proposed by the VO expert committee (menu-constrained), judged
    by the independent ScriptEvaluator on the held-out split. Calibration still uses the
    fixed reference contracts (vo_calibration_records). `run_fn` is injectable so the whole
    lineage is testable offline without an API key. `provider`/`menu` let the same machinery
    run on the synthetic world (default) or REAL TUM data with a real recipe."""
    from lab.agents.sdk import DEFAULT_MODEL, run_agent
    from lab.history import ResearchHistory

    from .agents.vo_committee import vo_committee

    h = build_vo_harness(root, job_mode=job_mode, images_path=images_path,
                         max_total_tokens=max_total_tokens, max_experiments=max_experiments,
                         lease_timeout_s=lease_timeout_s, seed=seed, provider=provider)
    h.planner = vo_committee(model=model or DEFAULT_MODEL, run_fn=run_fn or run_agent,
                             notebook=h.notebook, history=ResearchHistory(h.registry), menu=menu)
    return h


def build_vo_implementer_harness(root: str | Path, *, author_fn=None, task=None, provider=None,
                                 job_mode: str = "local", images_path: str | Path = _IMAGES,
                                 max_total_tokens: int = 2_000_000, max_experiments: int = 20,
                                 lease_timeout_s: float = 900.0) -> Harness:
    """Track B: the solver AUTHORS a VO algorithm (main.py), graded by the unchanged
    independent ScriptEvaluator on the held-out split against the task's fixed oracle
    (ATE-RMSE <= bar). Defaults to synthetic data + the reference author (offline, no API).
    Pass task=vo_impl_task_real(bar) + provider=TUMDatasetProvider() for real data. For a
    live sandboxed Claude author, leave author_fn=None then set:
        h.planner.author_fn = sdk_author(h.job_runner, h.image_registry, h.dataset_cache, ...)
    """
    from lab.factory import build_implementer_harness

    from .agents.vo_implementer import reference_author, vo_impl_task

    return build_implementer_harness(
        root, task or vo_impl_task(), author_fn or reference_author(),
        job_mode=job_mode, images_path=str(images_path),
        provider=provider or VODatasetProvider(),
        max_total_tokens=max_total_tokens, max_experiments=max_experiments,
        lease_timeout_s=lease_timeout_s)
