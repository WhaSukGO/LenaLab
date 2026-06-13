"""Variance for Ep8 (learned VO on GPU, optical-flow pose CNN, KITTI train -> test 05/07, recorded 19.77m
Sim3 ATE). The blog's own stat note flagged this as the one learned number with unmeasured variance. Re-train
the agent's EXACT Ep8 model across seeds -> mean+/-std. Resume-safe.

  GPU. Env: EP8VAR_SEEDS (default 42,1,2). Writes artifacts/fidelity_ladder/ep8_variance.json.
"""
import sys, os, json, statistics
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401
from lab.models import ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_kitti_learned
from vo_lab.plugins.vo_kitti_learned import KITTILearnedProvider

AGENT_CODE = Path("_vo_kitti_learned_impl_run/workspaces/vo-learned-impl-001/code/main.py")
OUT = Path(os.environ.get("EP8VAR_OUT",
           "/home/ws/devel/whasuk/LenaLab/artifacts/fidelity_ladder/ep8_variance.json"))
SEEDS = [int(s) for s in os.environ.get("EP8VAR_SEEDS", "42,1,2").split(",")]


def author_for_seed(seed: int):
    src = AGENT_CODE.read_text()
    for kw in ("torch.manual_seed", "np.random.seed", "random.seed"):
        src = src.replace(f"{kw}(42)", f"{kw}({seed})")
    assert f"manual_seed({seed})" in src, "seed patch failed"

    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src); return Usage()
    return author


def run_once(seed: int):
    task = vo_impl_task_kitti_learned(1e9)                    # defaults: train 00/02/06/08/09, test 05/07
    h = build_vo_implementer_harness(f"./_vo_ep8var_s{seed}", task=task,
                                     provider=KITTILearnedProvider(),
                                     author_fn=author_for_seed(seed), job_mode="docker",
                                     lease_timeout_s=5400.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"[s{seed}] need GPU image:", e); return None, None
    rec = h.run_experiment(ExperimentRecord(id=f"ep8var-s{seed}", hypothesis=f"ep8 seed {seed}"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()} or None
    return mm.get("ate_rmse"), per


def main():
    rows = json.load(open(OUT)) if OUT.exists() else []
    done = {r["seed"] for r in rows}
    for seed in [s for s in SEEDS if s not in done]:
        print(f"\ntraining Ep8 learned VO seed={seed}...", flush=True)
        ate, per = run_once(seed)
        rows = [r for r in rows if r["seed"] != seed] + [{"seed": seed, "ate": ate, "per_seq": per}]
        OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(rows, open(OUT, "w"), indent=2)
        print(f"  -> s{seed}: ATE {ate} m  {per}", flush=True)
    a = [r["ate"] for r in rows if r["ate"] is not None]
    if a:
        print(f"\n=== EP8 (recorded 19.77m): n={len(a)} mean {statistics.mean(a):.2f} "
              f"std {statistics.stdev(a) if len(a)>1 else 0:.2f} | vs ref 31.5 (gap large) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
