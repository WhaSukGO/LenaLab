"""SIM-TO-REAL TEST (the user's rung-8/'lung8' request): take the learned VO the agent authored +
trained on contamination-clean SYNTHETIC data, and run it on REAL KITTI driving photos (Lodestar
rung 8). Same pipeline, test domain swapped synthetic -> real. Graded by held-out Sim(3) ATE on
KITTI 07 + 09. Honest expectation: learned VO transfers POORLY sim->real (appearance gap); this
measures exactly how much. Writes result to /tmp/sim_to_real_kitti.txt.
"""
import sys, dataclasses
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  bootstrap lab
from lab.models import DatasetRef, ExperimentRecord, Usage
from lab.util import ensure_dir
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_synth_learned
from vo_lab.plugins.vo_synth import SyntheticLearnedProvider
from vo_lab.plugins.vo_kitti_learned import KITTILearnedProvider
from vo_lab.plugins.vo_ref.synthetic_stereo import generate_sequence

KITTI_TEST = ("07", "09")
ROOT = "./_vo_sim2real_run"
AGENT_CODE = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py"))
AGENT_MAIN = AGENT_CODE.read_text()
print(f"agent model from: {AGENT_CODE} ({len(AGENT_MAIN.splitlines())} lines)", flush=True)


class SimToRealProvider:
    """Train on the SAME clean synthetic seqs; TEST on real KITTI driving photos."""

    def __init__(self):
        self.synth = SyntheticLearnedProvider()
        self.kitti = KITTILearnedProvider(train=(), test=KITTI_TEST, test_max=300, test_stride=3)

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            self.kitti.fetch(ref, dest)                       # KITTI gt centres -> seq_<s>/gt.txt
        else:
            for name, kind, n, seed in self.synth.train:      # synthetic train (supervision)
                sub = ensure_dir(dest / "train" / f"seq_{name}")
                generate_sequence(sub, gt_dir=sub, kind=kind, n=n, seed=seed)
                (sub / "gt_poses.txt").rename(sub / "poses.txt")
                (sub / "gt.txt").unlink(missing_ok=True)
            self.kitti.fetch(ref, dest)                       # REAL KITTI frames -> test_input/seq_<s>/


def seed_agent_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(AGENT_MAIN)   # the agent's exact trained pipeline
        return Usage()
    return author


def main():
    base = vo_impl_task_synth_learned(1e9)
    datasets = [DatasetRef(name="sim2real-train", source="sim2real:train"),
                DatasetRef(name="sim2real-test-" + "_".join(KITTI_TEST),
                           source=";".join(f"sim2real:{s}" for s in KITTI_TEST), held_out=True)]
    task = dataclasses.replace(base, datasets=datasets)
    h = build_vo_implementer_harness(ROOT, task=task, provider=SimToRealProvider(),
                                     author_fn=seed_agent_author(), job_mode="docker",
                                     lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print("Need GPU image:", e); return 1
    print("SIM-TO-REAL: training agent's VO on synthetic, inferring on REAL KITTI 07+09...", flush=True)
    rec = h.run_experiment(ExperimentRecord(id="sim2real-kitti-001", hypothesis="sim->real transfer"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    ate = mm.get("ate_rmse")
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()}
    res = (f"SIM-TO-REAL (synthetic-trained learned VO -> REAL KITTI {KITTI_TEST}):\n"
           f"  held-out Sim3 ATE = {ate} m | per-seq {per}\n"
           f"  (rung-3 on synthetic was the in-domain number; this is the appearance-gap transfer)")
    Path("/tmp/sim_to_real_kitti.txt").write_text(res + "\n")
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
