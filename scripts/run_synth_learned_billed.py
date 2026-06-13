"""Rung-3 BILLED: agent authors+trains a learned VO on the contamination-clean synthetic domain
(GPU sandbox), graded by held-out Sim3 ATE on unseen synthetic test seqs. Bar 4.24m (ref x1.3)."""
import sys
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  bootstrap lab
from lab.models import ExperimentRecord, Status
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_synth_learned, resilient_sdk_author
from vo_lab.plugins.vo_synth import SyntheticLearnedProvider
from vo_lab.memory import inject_memory, record_from_experiment

bar = 4.24
task = inject_memory(vo_impl_task_synth_learned(bar), "synth-learned")
h = build_vo_implementer_harness("./_vo_synth_learned_impl_run", task=task,
                                 provider=SyntheticLearnedProvider(), job_mode="docker",
                                 lease_timeout_s=3600.0)
h.image_registry.resolve(task.framework)
h.planner.author_fn = resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                           model="claude-sonnet-4-6", max_turns=100)
print(f"BILLED rung-3: agent authoring+training a learned VO on synthetic (bar={bar}m ATE)...", flush=True)
rec = h.run_experiment(ExperimentRecord(id="synth-learned-impl-001", hypothesis="learned VO on synthetic"))
print("RESULT:", rec.status.value)
if rec.verdict:
    print("  measured:", rec.verdict.measured_metrics, "| verdict:", rec.verdict.verdict)
print("  tokens:", h.budget.state.total_tokens)
from pathlib import Path
art = None
if rec.contract and (Path(rec.contract.code_dir) / "main.py").exists():
    art = str(Path(rec.contract.code_dir) / "main.py")
record_from_experiment("synth-learned", rec, artifact=art)
