"""Fidelity-ladder GAUSSIAN+AUG rung: the augmentation REAL DATA CANNOT DO. GSplatModule renders the
GT trajectory PLUS two parallel offset paths (+/-0.8 m lateral) per scene = 3x diverse training views
with exact poses. Trains the agent VO, tests held-out real 07/09. Does manufactured viewpoint diversity
push BELOW the real-data ceiling (27.2 m)? Writes /tmp/gaussian_aug_rung.txt. GPU run."""
import sys, dataclasses
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401
from lab.models import DatasetRef, ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_synth_learned
from vo_lab.plugins.vo_gaussian import GaussianProvider

KITTI_TEST = ("07", "09")
PERTURB = [(0.8, 0.0, 0.0), (-0.8, 0.0, 0.0)]   # two parallel paths (novel viewpoints real data lacks)
ROOT = "./_vo_gaussian_aug_run"
AGENT_MAIN = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py")).read_text()
print(f"agent model: {len(AGENT_MAIN.splitlines())} lines | GT + {len(PERTURB)} parallel paths -> real {KITTI_TEST}", flush=True)


def seed_agent_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(AGENT_MAIN)
        return Usage()
    return author


def main():
    base = vo_impl_task_synth_learned(1e9)
    datasets = [DatasetRef(name="gaussian-aug-train", source="gaussian-aug:train"),
                DatasetRef(name="gaussian-aug-test-" + "_".join(KITTI_TEST),
                           source=";".join(f"gaussian-aug:{s}" for s in KITTI_TEST), held_out=True)]
    task = dataclasses.replace(base, datasets=datasets)
    prov = GaussianProvider(perturbations=PERTURB)
    h = build_vo_implementer_harness(ROOT, task=task, provider=prov, author_fn=seed_agent_author(),
                                     job_mode="docker", lease_timeout_s=5400.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print("Need GPU image:", e); return 1
    print("GAUSSIAN+AUG RUNG: rendering GT+parallel paths, training agent VO, inferring on real 07/09...", flush=True)
    rec = h.run_experiment(ExperimentRecord(id="gaussian-aug-rung-001", hypothesis="rendered viewpoint augmentation -> real"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    ate = mm.get("ate_rmse")
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()}
    res = (f"GAUSSIAN+AUG RUNG (GT + 2 parallel-path renders -> held-out REAL {KITTI_TEST}):\n"
           f"  held-out Sim3 ATE = {ate} m | per-seq {per}\n"
           f"  LADDER  procedural 69.6  ->  gaussian 27.1  ->  gaussian+aug {ate}  ->  real 27.2")
    Path("/tmp/gaussian_aug_rung.txt").write_text(res + "\n")
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
