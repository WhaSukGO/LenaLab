"""vo_lab — a Visual-Odometry research lab (the SOLVER) built on top of Touchstone (ver2).

ver3 does NOT reimplement the verifier. It imports ver2's spine (`lab`) — Harness,
ScriptEvaluator, registry, budget, gpu_lease, image_registry, dataset_cache, job_runner,
VerifiedResult — and adds only the VO domain: a dataset provider, harness-owned reference
code (run.py / eval.py), expert prompts, and a thin factory. See
`claudedocs/design_ver3_harness_architecture_2026-06-02.md`.

This bootstrap puts the sibling ver2 checkout on sys.path so `import lab` works without a
formal install. Override with VER2_PATH if ver2 lives elsewhere."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_ver2() -> Path:
    env = os.environ.get("VER2_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    # sibling layout: .../whasuk/blueberry_ver3/vo_lab/__init__.py -> .../whasuk/blueberry_ver2
    candidates.append(Path(__file__).resolve().parents[2] / "blueberry_ver2")
    for c in candidates:
        if (c / "lab" / "__init__.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise ImportError(
        "cannot locate ver2 (Touchstone). Set VER2_PATH to the blueberry_ver2 checkout. "
        f"tried: {[str(c) for c in candidates]}")


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader: populate os.environ from blueberry_ver3/.env
    for keys not already set (so ANTHROPIC_API_KEY in .env actually reaches the Agent SDK)."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


VER2_ROOT = _bootstrap_ver2()
_load_dotenv()
