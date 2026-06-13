"""FIDELITY LADDER runner — three training-domain rungs, ONE held-out real test (KITTI 07+09),
ONE grader (eval_learned Sim3 ATE). Only the TRAINING domain changes:
  procedural : procedural synthetic -> real   (== sim_to_real_kitti.py SimToRealProvider)
  gaussian   : 3DGS-rendered     -> real      (Phase-0 STUB: render seam falls back to procedural)
  real       : real KITTI 00/02/06/08 -> real (== KITTILearnedProvider train/test)

Phase 0 writes artifacts/fidelity_ladder/results.json from KNOWN cached numbers (NO GPU). The live
seam (run_domain_live) is implemented for later phases but MUST NOT be called here.
"""
import sys, json, argparse, dataclasses
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  bootstrap lab (puts ver2 `lab` on sys.path)
from lab.models import DatasetRef, ExperimentRecord, Usage
from lab.image_registry import NoImageError
from vo_lab.factory import build_vo_implementer_harness
from vo_lab.agents.vo_implementer import (vo_impl_task_synth_learned,
                                          vo_impl_task_kitti_learned)
from vo_lab.plugins.vo_gaussian import make_domain_provider  # single source of truth for ladder providers

TEST_SEQS = ("07", "09")
TRAIN_REAL = ("00", "02", "06", "08")
ROOT = "./_vo_fidelity_ladder_run"
RESULTS_PATH = Path("/home/ws/devel/whasuk/LenaLab/artifacts/fidelity_ladder/results.json")

DOMAINS = [
    {"name": "procedural",   "label": "procedural synth -> real"},
    {"name": "gaussian",     "label": "rendered -> real"},
    {"name": "gaussian_aug", "label": "rendered +viewpoint-aug -> real"},
    {"name": "real",         "label": "real -> real"},
]

# KNOWN cached results — Phase 0 reads these; NO new GPU runs this phase.
KNOWN = {
    "procedural": {"ate_rmse": 69.57, "per_seq": {"seq_07": 72.35, "seq_09": 66.78},
                   "status": "cached"},
    "real":       {"ate_rmse": 27.24, "per_seq": {"seq_07": 31.03, "seq_09": 23.45},
                   "status": "cached"},
    "gaussian":   {"ate_rmse": 27.11, "per_seq": {"seq_07": 34.31, "seq_09": 19.91},
                   "status": "done"},   # Phase-1: GSplatModule reprojection render (run_gaussian_rung.py)
    "gaussian_aug": {"ate_rmse": 26.88, "per_seq": {"seq_07": 31.82, "seq_09": 21.93},
                     "status": "done"},  # GT + 2 parallel-path renders; ~ties (within noise)
}
REFERENCE = {"degenerate": 63.29, "classical_vo_approx": 3.5, "synthetic_in_domain": 0.45}

# Agent's exact authored+trained learned-VO pipeline (same model across all rungs).
AGENT_CODE = next(Path("_vo_synth_learned_impl_run").rglob("code/main.py"))


# Providers + the ladder registry live in vo_lab/plugins/vo_gaussian.py (single source of truth);
# make_domain_provider is imported above. The runner only orchestrates, grades, and reports.


def make_domain_task(domain: str):
    """Build the matching eval_learned-graded task for the domain (same grader, same test seqs)."""
    if domain == "real":
        return vo_impl_task_kitti_learned(1e9, TRAIN_REAL, TEST_SEQS)
    # procedural + gaussian both ride the synth-learned task spine, retargeted at real KITTI test.
    base = vo_impl_task_synth_learned(1e9)
    datasets = [
        DatasetRef(name=f"{domain}-train", source=f"{domain}:train"),
        DatasetRef(name=f"{domain}-test-" + "_".join(TEST_SEQS),
                   source=";".join(f"{domain}:{s}" for s in TEST_SEQS), held_out=True),
    ]
    return dataclasses.replace(base, datasets=datasets)


def seed_agent_author():
    main_src = AGENT_CODE.read_text()                         # the agent's exact trained pipeline

    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(main_src)
        return Usage()
    return author


def run_domain_live(domain: str):
    """LIVE seam — builds harness with make_domain_provider + seeded agent main.py + eval_learned,
    runs it, returns measured held-out Sim3 ate_rmse (+ per-seq). NOT called in Phase 0 (GPU)."""
    task = make_domain_task(domain)
    prov = make_domain_provider(domain)
    root = f"{ROOT}_{domain}"
    h = build_vo_implementer_harness(root, task=task, provider=prov,
                                     author_fn=seed_agent_author(), job_mode="docker",
                                     lease_timeout_s=3600.0)
    try:
        h.image_registry.resolve(task.framework)
    except NoImageError as e:
        print(f"[{domain}] need GPU image:", e)
        return None, None
    print(f"[{domain}] training learned VO ({domain} domain) -> testing real KITTI {TEST_SEQS}...",
          flush=True)
    rec = h.run_experiment(ExperimentRecord(id=f"fidelity-{domain}-001",
                                            hypothesis=f"{domain}-trained learned VO on real"))
    mm = rec.verdict.measured_metrics if rec.verdict else {}
    ate = mm.get("ate_rmse")
    per = {k: round(v.get("ate_rmse", 0), 2) for k, v in (mm.get("per_seq") or {}).items()} or None
    return ate, per


def build_results(use_cached: bool) -> dict:
    domains = []
    for d in DOMAINS:
        name = d["name"]
        if use_cached:
            k = KNOWN[name]
            ate, per, status = k["ate_rmse"], k["per_seq"], k["status"]
        else:
            ate, per = run_domain_live(name)
            status = "measured" if ate is not None else "failed"
        domains.append({"name": name, "label": d["label"],
                        "ate_rmse": ate, "per_seq": per, "status": status})
    return {"test_seqs": list(TEST_SEQS), "domains": domains, "reference": dict(REFERENCE)}


def print_summary(results: dict) -> None:
    print(f"\nFIDELITY LADDER — held-out real KITTI {results['test_seqs']} (Sim3 ATE, metres)")
    print(f"  {'domain':<11} {'label':<26} {'ATE':>8}  {'seq_07':>7} {'seq_09':>7}  status")
    print("  " + "-" * 72)
    for d in results["domains"]:
        ate = "—" if d["ate_rmse"] is None else f"{d['ate_rmse']:.2f}"
        ps = d["per_seq"] or {}
        s7 = "—" if not ps else f"{ps.get('seq_07', float('nan')):.2f}"
        s9 = "—" if not ps else f"{ps.get('seq_09', float('nan')):.2f}"
        print(f"  {d['name']:<11} {d['label']:<26} {ate:>8}  {s7:>7} {s9:>7}  {d['status']}")
    r = results["reference"]
    print("  " + "-" * 72)
    print(f"  ref: degenerate={r['degenerate']}  classical_vo~{r['classical_vo_approx']}  "
          f"synthetic_in_domain={r['synthetic_in_domain']}")


def main():
    ap = argparse.ArgumentParser(description="Fidelity ladder runner (Phase 0: cached, no GPU)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--use-cached", dest="use_cached", action="store_true",
                   help="Phase-0 default: populate from KNOWN cached numbers, no GPU")
    g.add_argument("--live", dest="use_cached", action="store_false",
                   help="run each domain live on GPU (NOT for Phase 0)")
    ap.set_defaults(use_cached=True)
    args = ap.parse_args()

    results = build_results(args.use_cached)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results)
    print(f"\nwrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
