"""Cross-run experience memory — the 'structured handoff' the harness was missing.

ver2's `Notebook` logs failures to a *per-run* `failed_approaches.md`, but a fresh relaunch
(new run root) starts blank, so a re-attempted task never learns from the previous session.
This module keeps a **repo-level, cross-run** ledger keyed by `domain` and injects it into the
next author session's task description (the seam `_author_prompt` reads):

  - **failures** — what diverged / fell short, so the next agent avoids the dead end (this is
    what let the SLAM agent stop diverging);
  - **successes** — what *worked* (a concise approach summary, auto-extracted from the verified
    artifact's docstring), so the next agent can build on or improve it instead of re-deriving.

No ver2 edits — pure solver-side plumbing; the verifier is untouched. The agent still authors
working code that is graded independently on the held-out split, so an injected approach is a
reference/hint, not a shortcut around verification.

Layout (repo-level, stable across run roots):
    lab_memory/failures.{jsonl,md}    failed attempts (avoid these)
    lab_memory/successes.{jsonl,md}   verified approaches (build on these)
"""
from __future__ import annotations

import ast
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = _REPO_ROOT / "lab_memory"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(store: str | Path | None, kind: str) -> tuple[Path, Path]:
    base = Path(store) if store is not None else DEFAULT_STORE
    base.mkdir(parents=True, exist_ok=True)
    stem = "failures" if kind == "fail" else "successes"
    return base / f"{stem}.jsonl", base / f"{stem}.md"


def _append(jsonl: Path, md: Path, rec: dict, md_line: str) -> dict:
    with jsonl.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    with md.open("a") as f:
        f.write(md_line)
    return rec


def _extract_approach(artifact: str | Path | None, *, max_chars: int = 480) -> str:
    """Pull a short 'what worked' summary from a verified artifact: its module docstring if
    present, else the leading comment block. Agents commonly head main.py with exactly this."""
    if not artifact:
        return ""
    p = Path(artifact)
    if not p.exists():
        return ""
    text = p.read_text(errors="replace")
    summary = ""
    try:
        summary = ast.get_docstring(ast.parse(text)) or ""
    except (SyntaxError, ValueError):
        summary = ""
    if not summary:
        lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if s.startswith("#"):
                lines.append(s.lstrip("# ").rstrip())
            elif s and not s.startswith(('"""', "'''")):
                break
        summary = "\n".join(lines)
    summary = summary.strip()
    return (summary[:max_chars] + " …") if len(summary) > max_chars else summary


# ── recording ────────────────────────────────────────────────────────────────

def record_failure(domain: str, *, exp_id: str, what: str, failure_mode: str,
                   metric: str | None = None, measured: float | None = None,
                   bar: float | None = None, op: str = "<=", fix: str | None = None,
                   artifact: str | None = None, store: str | Path | None = None,
                   ts: str | None = None) -> dict[str, Any]:
    """Append one failed attempt to the cross-run ledger (jsonl + human md)."""
    rec = {
        "ts": ts or _now(), "domain": domain, "exp_id": exp_id, "what": what,
        "failure_mode": failure_mode, "metric": metric, "measured": measured,
        "bar": bar, "op": op, "fix": fix, "artifact": artifact,
    }
    jsonl, md = _paths(store, "fail")
    target = f"{metric} {op} {bar}" if (metric and bar is not None) else "(bar n/a)"
    got = f"{measured:g}" if measured is not None else "—"
    line = (f"- `{rec['ts'][:19]}` **[{domain}]** `{exp_id}` — {what}\n"
            f"  - measured **{got}** vs needed **{target}** → {failure_mode}\n")
    if fix:
        line += f"  - fix/hint: {fix}\n"
    if artifact:
        line += f"  - artifact: `{artifact}`\n"
    return _append(jsonl, md, rec, line)


def record_success(domain: str, *, exp_id: str, what: str, approach: str = "",
                   metric: str | None = None, measured: float | None = None,
                   bar: float | None = None, op: str = "<=", artifact: str | None = None,
                   store: str | Path | None = None, ts: str | None = None) -> dict[str, Any]:
    """Append one VERIFIED approach to the cross-run ledger (jsonl + human md)."""
    rec = {
        "ts": ts or _now(), "domain": domain, "exp_id": exp_id, "what": what,
        "approach": approach, "metric": metric, "measured": measured, "bar": bar,
        "op": op, "artifact": artifact,
    }
    jsonl, md = _paths(store, "success")
    target = f"{metric} {op} {bar}" if (metric and bar is not None) else ""
    got = f"{measured:g}" if measured is not None else "—"
    line = (f"- `{rec['ts'][:19]}` **[{domain}]** `{exp_id}` — VERIFIED at **{got}**"
            f"{(' (' + target + ')') if target else ''} — {what}\n")
    if approach:
        line += "  - approach: " + approach.replace("\n", " ").strip()[:500] + "\n"
    if artifact:
        line += f"  - artifact: `{artifact}`\n"
    return _append(jsonl, md, rec, line)


def record_from_experiment(domain: str, rec, *, what: str | None = None,
                           fix: str | None = None, artifact: str | None = None,
                           store: str | Path | None = None) -> dict[str, Any] | None:
    """Record a finished `ExperimentRecord`: VERIFIED → success ledger, REJECTED/FAILED →
    failure ledger. Returns the ledger entry, or None for non-terminal status."""
    status = getattr(rec.status, "value", str(rec.status))
    label = what or getattr(rec, "hypothesis", rec.id)
    v = getattr(rec, "verdict", None)
    oc = (getattr(v, "oracle_comparison", None) or {}) if v is not None else {}
    metric, measured, bar = oc.get("metric"), oc.get("measured"), oc.get("expected")
    op = oc.get("op", "<=")

    if status == "VERIFIED":
        return record_success(domain, exp_id=rec.id, what=label,
                              approach=_extract_approach(artifact), metric=metric,
                              measured=measured, bar=bar, op=op, artifact=artifact, store=store)
    if status in ("REJECTED", "FAILED"):
        if v is not None:
            notes = (getattr(v, "evaluator_notes", "") or "").strip()
            mode = notes[-200:] if notes else f"evaluator {getattr(v, 'verdict', '')} ({status})"
        else:
            mode = f"run {status} before grading (no verdict)"
        return record_failure(domain, exp_id=rec.id, what=label, failure_mode=mode,
                             metric=metric, measured=measured, bar=bar, op=op, fix=fix,
                             artifact=artifact, store=store)
    return None


# ── recall / injection ───────────────────────────────────────────────────────

def _load(kind: str, domain: str | None, store: str | Path | None) -> list[dict]:
    jsonl, _ = _paths(store, kind)
    if not jsonl.exists():
        return []
    out = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if domain is None or d.get("domain") == domain:
            out.append(d)
    return out


def load_failures(domain: str | None = None, *, store: str | Path | None = None) -> list[dict]:
    return _load("fail", domain, store)


def load_successes(domain: str | None = None, *, store: str | Path | None = None) -> list[dict]:
    return _load("success", domain, store)


def failure_context(domain: str, *, limit: int = 5, store: str | Path | None = None) -> str:
    """A prompt block summarizing prior failed attempts on this domain. '' if none."""
    fails = load_failures(domain, store=store)
    if not fails:
        return ""
    lines = ["PRIOR FAILED ATTEMPTS ON THIS TASK — learn from these, do NOT repeat them:"]
    for i, d in enumerate(fails[-limit:], 1):
        tgt = (f"{d.get('metric')} {d.get('op','<=')} {d.get('bar')}"
               if d.get("metric") and d.get("bar") is not None else "the bar")
        got = f"{d['measured']:g}" if d.get("measured") is not None else "—"
        lines.append(f"  {i}. [{d.get('exp_id')}] {d.get('what','')} — measured {got} "
                     f"(needed {tgt}). Failure: {d.get('failure_mode','')}")
        if d.get("fix"):
            lines.append(f"     hint: {d['fix']}")
    return "\n".join(lines)


def success_context(domain: str, *, limit: int = 3, store: str | Path | None = None) -> str:
    """A prompt block summarizing prior VERIFIED approaches. '' if none. The agent may build on
    or improve these — it still must author working code graded on the held-out split."""
    wins = load_successes(domain, store=store)
    if not wins:
        return ""
    lines = ["PRIOR VERIFIED APPROACHES ON THIS TASK — you may build on or improve these "
             "(you still author the code; it is graded on a held-out split you can't see):"]
    for i, d in enumerate(wins[-limit:], 1):
        got = f"{d['measured']:g}" if d.get("measured") is not None else "—"
        lines.append(f"  {i}. [{d.get('exp_id')}] {d.get('what','')} — VERIFIED at {got}")
        if d.get("approach"):
            lines.append("     approach: " + d["approach"].replace("\n", " ").strip()[:480])
    return "\n".join(lines)


def memory_context(domain: str, *, include_success: bool = True, include_failure: bool = True,
                   fail_limit: int = 5, success_limit: int = 3,
                   store: str | Path | None = None) -> str:
    """Combined prior-experience block: successes (build on) + failures (avoid). '' if none."""
    blocks = []
    if include_success:
        s = success_context(domain, limit=success_limit, store=store)
        if s:
            blocks.append(s)
    if include_failure:
        f = failure_context(domain, limit=fail_limit, store=store)
        if f:
            blocks.append(f)
    return "\n\n".join(blocks)


def inject_memory(task, domain: str, *, include_success: bool = True,
                  include_failure: bool = True, store: str | Path | None = None):
    """Return `task` with the prior-experience block (successes + failures) prepended to its
    description. Non-mutating; returns the task unchanged if there is no memory."""
    ctx = memory_context(domain, include_success=include_success,
                         include_failure=include_failure, store=store)
    if not ctx:
        return task
    return replace(task, description=f"{task.description}\n\n{ctx}")


def inject_failure_memory(task, domain: str, *, limit: int = 5, store: str | Path | None = None):
    """Back-compat: inject failures only (kept for callers/tests that predate success memory)."""
    return inject_memory(task, domain, include_success=False, include_failure=True, store=store)
