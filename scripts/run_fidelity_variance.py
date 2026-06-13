"""Fidelity-ladder VARIANCE run (audit Priority 1). The blog's headline conclusion — 'rendered (27.1) ~=
real (27.2); the sim-to-real gap is appearance, not capacity' — rests on those numbers being within
TRAINING NOISE of each other. But each was a SINGLE training run (fixed seed 42, but GPU training is not
bit-deterministic). This re-trains the close rungs across several seeds and reports mean +/- std, so we know
whether 27.1 vs 27.2 is signal or noise.

Reuses the exact agent model + providers + grader from run_fidelity_ladder.py; only the seed varies.
Incremental + resume-safe: each (domain, seed) writes to artifacts/fidelity_ladder/variance.json; re-running
skips completed cells. Env: FIDVAR_DOMAINS (default gaussian,gaussian_aug,real), FIDVAR_SEEDS (default 42,1,2).

  GPU run. ~12 min/training. Default 3 domains x 3 seeds = 9 trainings ~ 2-2.5 h.
"""
import sys, os, json, dataclasses, statistics
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401
from lab.models import DatasetRef, ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import vo_impl_task_synth_learned, vo_impl_task_kitti_learned
from vo_lab.plugins.vo_gaussian import make_domain_provider

TEST_SEQS = ("07", "09")
TRAIN_REAL = ("00", "02", "06", "08")
AGENT_CODE = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py"))
OUT = Path(os.environ.get("FIDVAR_OUT",
           "/home/ws/devel/whasuk/LenaLab/artifacts/fidelity_ladder/variance.json"))
DOMAINS = os.environ.get("FIDVAR_DOMAINS", "gaussian,gaussian_aug,real").split(",")
SEEDS = [int(s) for s in os.environ.get("FIDVAR_SEEDS", "42,1,2").split(",")]


def make_domain_task(domain: str):
    if domain == "real":
        return vo_impl_task_kitti_learned(1e9, TRAIN_REAL, TEST_SEQS)
    base = vo_impl_task_synth_learned(1e9)
    datasets = [DatasetRef(name=f"{domain}-train", source=f"{domain}:train"),
                DatasetRef(name=f"{domain}-test-" + "_".join(TEST_SEQS),
                           source=";".join(f"{domain}:{s}" for s in TEST_SEQS), held_out=True)]
    return dataclasses.replace(base, datasets=datasets)


def author_for_seed(seed: int):
    src = AGENT_CODE.read_text()
    for kw in ("torch.manual_seed", "random.seed", "np.random.seed"):
        src = src.replace(f"{kw}(42)", f"{kw}({seed})")
    assert f"manual_seed({seed})" in src, "seed patch failed"

    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
        return Usage()
    return author


def run_once(domain: str, seed: int):
    task = make_domain_task(domain)
    h = build_vo_implementer_harness(f"./_vo_fidvar_{domain}_s{seed}", task=task,
                                     provider=make_domain_provider(domain),
                                     author_fn=author_for_seed(seed), job_mode="docker",
                                     lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"[{domain} s{seed}] need GPU image:", e); return None, None
    rec = h.run_experiment(ExperimentRecord(id=f"fidvar-{domain}-s{seed}",
                                            hypothesis=f"{domain} learned VO, seed {seed}"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()} or None
    return mm.get("ate_rmse"), per


def load():
    return json.load(open(OUT)) if OUT.exists() else []


def save(rows):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=2)


def summarize(rows):
    print("\n=== FIDELITY-LADDER VARIANCE (held-out real KITTI 07/09, Sim3 ATE m) ===")
    print(f"  {'domain':<14} {'n':>2}  {'mean':>7} {'std':>6} {'min':>7} {'max':>7}   seeds->ATE")
    stats = {}
    for dom in DOMAINS:
        vals = [(r["seed"], r["ate"]) for r in rows if r["domain"] == dom and r["ate"] is not None]
        if not vals:
            print(f"  {dom:<14}  (none yet)"); continue
        a = [v for _, v in vals]
        mean = statistics.mean(a); sd = statistics.stdev(a) if len(a) > 1 else 0.0
        stats[dom] = (mean, sd, len(a))
        seeds = " ".join(f"s{s}={v:.1f}" for s, v in sorted(vals))
        print(f"  {dom:<14} {len(a):>2}  {mean:>7.2f} {sd:>6.2f} {min(a):>7.2f} {max(a):>7.2f}   {seeds}")
    if "gaussian" in stats and "real" in stats:
        (gm, gs, gn), (rm, rs, rn) = stats["gaussian"], stats["real"]
        diff = abs(gm - rm); pooled = (gs + rs) / 2 or 0.01
        print(f"\n  VERDICT: rendered {gm:.2f}±{gs:.2f}  vs  real {rm:.2f}±{rs:.2f}  "
              f"| diff {diff:.2f} m = {diff/pooled:.1f}x pooled-std")
        print("  -> 'rendered ~= real' " + ("HOLDS (diff < 1 std, within noise)" if diff <= pooled
              else "is QUESTIONABLE (diff > 1 std)" if diff <= 2 * pooled
              else "FAILS (diff > 2 std)"))


def main():
    rows = load()
    done = {(r["domain"], r["seed"]) for r in rows}
    todo = [(d, s) for d in DOMAINS for s in SEEDS if (d, s) not in done]
    print(f"variance run: {len(done)} done, {len(todo)} to do  ({DOMAINS} x {SEEDS})", flush=True)
    for i, (dom, seed) in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] training {dom} seed={seed} (~12 min)...", flush=True)
        ate, per = run_once(dom, seed)
        rows = [r for r in rows if not (r["domain"] == dom and r["seed"] == seed)]
        rows.append({"domain": dom, "seed": seed, "ate": ate, "per_seq": per})
        save(rows)
        print(f"  -> {dom} s{seed}: ATE {ate} m  {per}", flush=True)
        summarize(rows)
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
