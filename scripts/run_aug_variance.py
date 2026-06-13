"""Variance for the gaussian+AUG rung (Ep17: 26.88m, 'augmentation ties rendered, doesn't beat real').
Re-trains the agent VO on GT + 2 parallel-path renders, tests real 07/09, across seeds -> mean+/-std.
Confirms whether aug is distinguishable from plain rendered (27.35+/-1.49) or within noise. Resume-safe.

  GPU. ~12 min/training. Env: AUGVAR_SEEDS (default 42,1,2). Writes artifacts/fidelity_ladder/aug_variance.json.
"""
import sys, os, json, dataclasses, statistics
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401
from lab.models import DatasetRef, ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_synth_learned
from vo_lab.plugins.vo_gaussian import GaussianProvider

KITTI_TEST = ("07", "09")
PERTURB = [(0.8, 0.0, 0.0), (-0.8, 0.0, 0.0)]
AGENT_CODE = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py"))
OUT = Path("/home/ws/devel/whasuk/LenaLab/artifacts/fidelity_ladder/aug_variance.json")
SEEDS = [int(s) for s in os.environ.get("AUGVAR_SEEDS", "42,1,2").split(",")]


def author_for_seed(seed: int):
    src = AGENT_CODE.read_text()
    for kw in ("torch.manual_seed", "random.seed", "np.random.seed"):
        src = src.replace(f"{kw}(42)", f"{kw}({seed})")
    assert f"manual_seed({seed})" in src

    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src); return Usage()
    return author


def run_once(seed: int):
    base = vo_impl_task_synth_learned(1e9)
    datasets = [DatasetRef(name="gaussian-aug-train", source="gaussian-aug:train"),
                DatasetRef(name="gaussian-aug-test-" + "_".join(KITTI_TEST),
                           source=";".join(f"gaussian-aug:{s}" for s in KITTI_TEST), held_out=True)]
    task = dataclasses.replace(base, datasets=datasets)
    h = build_vo_implementer_harness(f"./_vo_augvar_s{seed}", task=task,
                                     provider=GaussianProvider(perturbations=PERTURB),
                                     author_fn=author_for_seed(seed), job_mode="docker", lease_timeout_s=5400.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"[s{seed}] need GPU image:", e); return None, None
    rec = h.run_experiment(ExperimentRecord(id=f"augvar-s{seed}", hypothesis=f"aug seed {seed}"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()} or None
    return mm.get("ate_rmse"), per


def main():
    rows = json.load(open(OUT)) if OUT.exists() else []
    done = {r["seed"] for r in rows}
    for seed in [s for s in SEEDS if s not in done]:
        print(f"\ntraining gaussian+aug seed={seed} (~12 min)...", flush=True)
        ate, per = run_once(seed)
        rows = [r for r in rows if r["seed"] != seed] + [{"seed": seed, "ate": ate, "per_seq": per}]
        OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(rows, open(OUT, "w"), indent=2)
        print(f"  -> s{seed}: ATE {ate} m  {per}", flush=True)
    a = [r["ate"] for r in rows if r["ate"] is not None]
    if a:
        print(f"\n=== GAUSSIAN+AUG (Ep17 26.88m): n={len(a)} mean {statistics.mean(a):.2f} "
              f"std {statistics.stdev(a) if len(a)>1 else 0:.2f} | vs plain rendered 27.35+/-1.49 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
