"""Parallel multi-agent orchestration for the lab — two modes:

  • COMPETITION (tournament): N different approaches author the SAME task in parallel; the
    independent verifier grades each on the held-out and the BEST verified result wins. This
    directly attacks why M2 stalled — four *sequential* single-approach attempts; a tournament
    surfaces a working approach (or confirms the wall) in one round.

  • COOPERATION (cooperative_pipeline): the task is DECOMPOSED into stages (e.g. front-end →
    loop-closure); each stage runs its own tournament, and the stage's winning module is LOCKED
    and handed to the next stage (via the scaffold seed mechanism). Division of labour: each agent
    owns one piece, building on the verified work of the previous — work compounds instead of being
    re-derived (the incremental-build default, parallelised).

Both reuse the unchanged harness + independent verifier; nothing here grades anything itself.
Build/validate offline with non-billed reference/degenerate authors; run billed with sdk_factory.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from lab.models import ExperimentRecord, Status

from .factory import build_vo_implementer_harness


# ---- author factories: (harness) -> author_fn -----------------------------------------------
def ref_factory(author_fn) -> Callable:
    """Wrap a ready non-billed author (e.g. kitti_stereo_reference_author()) as a factory."""
    return lambda h: author_fn


def sdk_factory(model: str = "claude-sonnet-4-6", max_turns: int = 100) -> Callable:
    """A live sandboxed Claude author (billed). Built per-harness so each parallel run is isolated."""
    from .agents.vo_implementer import resilient_sdk_author

    def make(h):
        return resilient_sdk_author(h.job_runner, h.image_registry, h.dataset_cache,
                                    model=model, max_turns=max_turns)
    return make


def _seeded(author_fn, seed_files: dict[str, str]):
    """Lock files into code_dir before authoring (scaffold / cooperation hand-off)."""
    def author(task, code_dir, rec):
        for name, src in seed_files.items():
            (Path(code_dir) / name).write_text(src)
        return author_fn(task, code_dir, rec)
    return author


# ---- one run --------------------------------------------------------------------------------
def _run_variant(variant: dict, *, provider, root: Path, job_mode: str,
                 lease_timeout_s: float) -> dict:
    label = variant["label"]
    task = variant["task"]
    sub = Path(root) / label
    h = build_vo_implementer_harness(sub, task=task, provider=provider,
                                     job_mode=job_mode, lease_timeout_s=lease_timeout_s)
    if job_mode == "docker":
        from lab.image_registry import NoImageError
        try:
            h.image_registry.resolve(task.framework)
        except NoImageError as e:
            return {"label": label, "status": "NO_IMAGE", "error": str(e), "metric": None,
                    "passed": False, "code_path": None}
    author = variant["author_factory"](h)
    if variant.get("seed_files"):
        author = _seeded(author, variant["seed_files"])
    h.planner.author_fn = author
    rec = h.run_experiment(ExperimentRecord(id=f"par-{label}", hypothesis=variant.get("hypothesis", label)))
    mm = (rec.verdict.measured_metrics if rec.verdict else {}) or {}
    metric = task.metric
    code = (Path(rec.contract.code_dir) / task.entry_filename) if rec.contract else None
    result = {
        "label": label,
        "status": rec.status.value,
        "passed": rec.status == Status.VERIFIED,
        "metric": mm.get(metric),
        "measured": mm,
        "code_path": str(code) if (code and code.exists()) else None,
        "tokens": getattr(h.budget.state, "total_tokens", None),
    }
    try:  # checkpoint: lets a re-run skip this completed variant (resume after crash/restart)
        (sub / "result.json").write_text(json.dumps(result))
    except Exception:  # noqa: BLE001
        pass
    return result


def _rank(results: list[dict], op: str) -> list[dict]:
    """Verified first; then by the metric (op '<=' => lower is better)."""
    big = float("inf")
    def key(r):
        m = r["metric"] if r["metric"] is not None else big
        return (0 if r["passed"] else 1, m if op == "<=" else -m)
    return sorted(results, key=key)


# ---- COMPETITION ----------------------------------------------------------------------------
def tournament(variants: list[dict], *, provider, root: str | Path, op: str = "<=",
               job_mode: str = "docker", lease_timeout_s: float = 3600.0,
               max_workers: int = 4, resume: bool = True, log=print) -> dict:
    """Run all `variants` in parallel on the same provider/held-out; return ranked results + winner.
    Each variant: {label, task, author_factory, [seed_files], [hypothesis]}.
    resume=True (default): variants with a saved result.json (root/<label>/result.json) are skipped
    and their result reused — so a re-run after a crash/restart only does the unfinished work."""
    root = Path(root)
    pending, results = [], []
    for v in variants:
        rj = root / v["label"] / "result.json"
        if resume and rj.exists():
            try:
                results.append(json.loads(rj.read_text()))
                log(f"[tournament] {v['label']}: RESUMED from checkpoint "
                    f"({results[-1].get('status')} {results[-1].get('metric')})")
                continue
            except Exception:  # noqa: BLE001
                pass
        pending.append(v)
    log(f"[tournament] {len(pending)} to run in parallel (workers={min(max_workers,max(1,len(pending)))}), "
        f"{len(results)} resumed: {[v['label'] for v in pending]}")
    if pending:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pending))) as ex:
            futs = {ex.submit(_run_variant, v, provider=provider, root=root,
                              job_mode=job_mode, lease_timeout_s=lease_timeout_s): v for v in pending}
            for f in as_completed(futs):
                v = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001
                    r = {"label": v["label"], "status": "ERROR", "error": str(e)[:300],
                         "passed": False, "metric": None, "code_path": None}
                log(f"[tournament] {r['label']}: {r['status']} "
                    f"({r.get('metric')}{' ✓' if r['passed'] else ''})")
                results.append(r)
    ranked = _rank(results, op)
    winner = ranked[0] if ranked and ranked[0]["passed"] else (ranked[0] if ranked else None)
    log(f"[tournament] winner: {winner['label'] if winner else None} "
        f"({winner.get('metric') if winner else None})")
    return {"winner": winner, "ranked": ranked}


# ---- COOPERATION ----------------------------------------------------------------------------
def cooperative_pipeline(stages: list[dict], *, provider, root: str | Path, op: str = "<=",
                         job_mode: str = "docker", lease_timeout_s: float = 3600.0,
                         max_workers: int = 4, log=print) -> dict:
    """Division of labour: run each stage's tournament, then LOCK its winning module and hand it to
    the next stage (scaffold seed). Each stage: {name, variants, [seed_as]} — seed_as is the filename
    the previous winner's authored module is locked as for this stage's authors (e.g. 'frontend.py').
    Returns each stage's result + the final winner."""
    root = Path(root)
    prev = None
    stage_results = []
    for i, stage in enumerate(stages):
        variants = [dict(v) for v in stage["variants"]]  # copy (we may inject seed_files)
        seed_as = stage.get("seed_as")
        if prev is not None and seed_as and prev.get("code_path"):
            locked = Path(prev["code_path"]).read_text()
            log(f"[cooperate] locking stage-{i} input '{seed_as}' from winner '{prev['label']}'")
            for v in variants:
                v["seed_files"] = {**(v.get("seed_files") or {}), seed_as: locked}
        log(f"[cooperate] === stage {i}: {stage['name']} ===")
        res = tournament(variants, provider=provider, root=root / f"stage{i}_{stage['name']}",
                         op=op, job_mode=job_mode, lease_timeout_s=lease_timeout_s,
                         max_workers=max_workers, log=log)
        stage_results.append({"stage": stage["name"], **res})
        prev = res["winner"]
        if prev is None:
            log(f"[cooperate] stage {i} produced no winner; stopping."); break
    return {"final": prev, "stages": stage_results}
