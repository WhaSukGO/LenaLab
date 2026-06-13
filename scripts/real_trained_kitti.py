"""Close the sim-to-real loop (Experiment A — isolation test): train the agent's SAME authored
learned VO on REAL KITTI (00/02/06/08), test on held-out REAL KITTI 07/09 — the SAME test seqs as
the sim-trained run, so the ONLY thing that changes is training data (synthetic -> real). Isolates
'sim-to-real appearance gap' from 'learned VO can't do real driving on one GPU'. Writes
/tmp/real_trained_kitti.txt.
"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  bootstrap lab
from lab.models import ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_kitti_learned
from vo_lab.plugins.vo_kitti_learned import KITTILearnedProvider

TRAIN = ("00", "02", "06", "08")
TEST = ("07", "09")
ROOT = "./_vo_real_learned_run"
AGENT_MAIN = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py")).read_text()
print(f"agent model: {len(AGENT_MAIN.splitlines())} lines | train REAL {TRAIN} -> test REAL {TEST}", flush=True)


def seed_agent_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(AGENT_MAIN)   # the EXACT model from rung 3
        return Usage()
    return author


def main():
    task = vo_impl_task_kitti_learned(1e9, TRAIN, TEST)
    prov = KITTILearnedProvider(train=TRAIN, test=TEST, train_max=250, train_stride=2,
                                test_max=300, test_stride=3)
    h = build_vo_implementer_harness(ROOT, task=task, provider=prov, author_fn=seed_agent_author(),
                                     job_mode="docker", lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print("Need GPU image:", e); return 1
    print(f"REAL-TRAINED: training agent's VO on REAL KITTI {TRAIN}, inferring on {TEST}...", flush=True)
    rec = h.run_experiment(ExperimentRecord(id="real-learned-kitti-001", hypothesis="real-trained learned VO"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    ate = mm.get("ate_rmse")
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()}
    res = (f"REAL-TRAINED learned VO (train REAL KITTI {TRAIN} -> test held-out REAL {TEST}):\n"
           f"  held-out Sim3 ATE = {ate} m | per-seq {per}\n"
           f"  COMPARE  sim-trained -> real 07/09 = 69.6 m   |   synthetic in-domain = 0.45 m")
    Path("/tmp/real_trained_kitti.txt").write_text(res + "\n")
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
