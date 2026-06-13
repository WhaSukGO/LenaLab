"""Rung-3 de-risk (GPU, non-billed): train the REFERENCE learned VO on the contamination-clean
synthetic data and grade on the unseen synthetic test seqs — confirms the GPU learned pipeline works
and sets the baseline/bar before the billed agent run. Writes result to /tmp/synth_learned_derisk.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  -- bootstraps the ver2 `lab` package onto sys.path
from lab.models import ExperimentRecord
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import (vo_impl_task_synth_learned, kitti_learned_reference_author,
                                          kitti_learned_degenerate_author)
from vo_lab.plugins.vo_synth import SyntheticLearnedProvider

ROOT = "./_vo_synth_learned_run"
prov = SyntheticLearnedProvider()


def measure(label, author):
    task = vo_impl_task_synth_learned(1e9)
    h = build_vo_implementer_harness(ROOT, task=task, provider=prov, author_fn=author,
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print("Need GPU image:", e); return None
    rec = h.run_experiment(ExperimentRecord(id=f"synthlearned-{label}", hypothesis=label))
    return rec.verdict.measured_metrics if rec.verdict else None


print("Training REFERENCE learned VO on synthetic (GPU)...", flush=True)
ref = measure("ref", kitti_learned_reference_author())
if not (ref and "ate_rmse" in ref):
    Path("/tmp/synth_learned_derisk.txt").write_text("REFERENCE FAILED — see run log\n")
    print("reference failed"); sys.exit(1)
ate = float(ref["ate_rmse"]); bar = round(ate * 1.3, 2)
neg = measure("degenerate", kitti_learned_degenerate_author())
ate_neg = float(neg["ate_rmse"]) if neg and "ate_rmse" in neg else 1e9
opened = (ate <= bar) and (ate_neg > bar)
res = (f"REFERENCE learned VO (synthetic, Sim3 ATE): {ate:.2f} m | per-seq "
       f"{ {k: round(v['ate_rmse'],1) for k,v in (ref.get('per_seq') or {}).items()} }\n"
       f"degenerate control: {ate_neg:.2f} m | derived bar (x1.3): {bar} m | "
       f"GATE {'OPEN' if opened else 'LOCKED'}\n"
       f"-> learned/GPU pipeline {'WORKS on clean synthetic; billed agent run worth it' if opened else 'did not validate'}")
Path("/tmp/synth_learned_derisk.txt").write_text(res + "\n")
print(res)
