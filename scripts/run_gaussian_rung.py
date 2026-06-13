"""Fidelity-ladder GAUSSIAN rung (Phase 1): train the agent's SAME learned VO on RENDERED real-KITTI
scenes (GSplatModule stereo-depth reprojection of scenes 00/02/06/08), test on held-out REAL KITTI
07/09. Fills the pending middle rung: does real-appearance RENDERED training data close the
69.6 -> 27.2 m sim-to-real gap? Writes /tmp/gaussian_rung.txt. GPU run (mirrors sim_to_real_kitti.py)."""
import sys, dataclasses
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  bootstrap lab
from lab.models import DatasetRef, ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_synth_learned
from vo_lab.plugins.vo_gaussian import GaussianProvider

KITTI_TEST = ("07", "09")
ROOT = "./_vo_gaussian_rung_run"
AGENT_MAIN = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py")).read_text()
print(f"agent model: {len(AGENT_MAIN.splitlines())} lines | RENDERED train -> real {KITTI_TEST}", flush=True)


def seed_agent_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(AGENT_MAIN)   # the agent's exact rung-3 model
        return Usage()
    return author


def main():
    base = vo_impl_task_synth_learned(1e9)
    datasets = [DatasetRef(name="gaussian-train", source="gaussian:train"),
                DatasetRef(name="gaussian-test-" + "_".join(KITTI_TEST),
                           source=";".join(f"gaussian:{s}" for s in KITTI_TEST), held_out=True)]
    task = dataclasses.replace(base, datasets=datasets)
    h = build_vo_implementer_harness(ROOT, task=task, provider=GaussianProvider(),
                                     author_fn=seed_agent_author(), job_mode="docker",
                                     lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print("Need GPU image:", e); return 1
    print("GAUSSIAN RUNG: rendering real-KITTI scenes, training agent VO, inferring on real 07/09...", flush=True)
    rec = h.run_experiment(ExperimentRecord(id="gaussian-rung-001", hypothesis="rendered-real -> real transfer"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    ate = mm.get("ate_rmse")
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()}
    res = (f"GAUSSIAN RUNG (RENDERED real-KITTI train -> held-out REAL {KITTI_TEST}):\n"
           f"  held-out Sim3 ATE = {ate} m | per-seq {per}\n"
           f"  LADDER  procedural 69.6  ->  gaussian {ate}  ->  real 27.2  (in-domain synth 0.45)")
    Path("/tmp/gaussian_rung.txt").write_text(res + "\n")
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
