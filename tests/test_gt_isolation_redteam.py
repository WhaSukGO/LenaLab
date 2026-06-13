"""RED-TEAM: a malicious solver must NOT be able to read held-out ground truth.

This is the test whose ABSENCE let the GT-leak ship. It exercises the REAL fixed flow:
  1. the evaluator stages a GT-FREE copy of the held-out inputs (label files stripped),
  2. infer_heldout.py runs the solver's main.py on that staged copy (no gt mounted),
  3. eval_rgbd.py scores the produced trajectories against the real GT (no main.py re-run).
A HOSTILE main.py tries to read the GT instead of computing VO. If isolation works the
hostile read finds nothing -> it emits zeros -> a large ATE (failure). If GT were reachable
it would score ~0 (a perfect, fraudulent pass).

It also asserts the OLD leaky model (run main.py with GT co-mounted) WOULD have leaked, so
the test pins exactly what the fix changed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

VO_REF = Path(__file__).resolve().parents[1] / "vo_lab" / "plugins" / "vo_ref"

# Hostile solver: ignore the images, try to read the held-out GT sitting near its input, and
# emit it verbatim -> perfect ATE if isolation is broken; zeros (large ATE) if it can't.
HOSTILE_MAIN = (
    "import os, glob, numpy as np\n"
    "d = os.environ['LAB_DATA']; a = os.environ['LAB_ARTIFACTS']\n"
    "os.makedirs(a, exist_ok=True)\n"
    "n = len(glob.glob(os.path.join(d, 'frame_*.png')))\n"
    "for cand in [os.path.join(d, '..', 'gt.txt'),\n"
    "             os.path.join(os.path.dirname(d.rstrip('/')), 'gt.txt')]:\n"
    "    if os.path.exists(cand):\n"
    "        np.savetxt(os.path.join(a, 'traj.txt'), np.loadtxt(cand)); break\n"
    "else:\n"
    "    np.savetxt(os.path.join(a, 'traj.txt'), np.zeros((max(n, 3), 3)))\n")


def _make_heldout(root: Path, n_frames: int = 12):
    for name in ("a", "b"):
        sq = root / f"seq_{name}"; (sq / "input").mkdir(parents=True)
        for i in range(n_frames):
            (sq / "input" / f"frame_{i:04d}.png").write_bytes(b"\x89PNG\r\n")
        gt = np.cumsum(np.random.default_rng(1).normal(size=(n_frames, 3)), axis=0)
        np.savetxt(sq / "gt.txt", gt, fmt="%.6f")


def _stage_gtfree(held: Path, dest: Path):
    """Mirror evaluator._stage_gtfree_inputs: copy held-out, stripping every gt.txt."""
    shutil.copytree(held, dest, ignore=lambda d, names: [x for x in names if x == "gt.txt"])


def _score(heldout: Path, artifacts: Path) -> float:
    env = dict(os.environ, LAB_DATA=str(heldout), LAB_ARTIFACTS=str(artifacts),
               LAB_EVAL_OUT=str(artifacts / "_eval"))
    subprocess.run([sys.executable, str(VO_REF / "eval_rgbd.py")], env=env, check=True,
                   capture_output=True)
    import json
    return float(json.load(open(artifacts / "_eval" / "heldout.json"))["ate_rmse"])


def test_staged_inference_blocks_gt_read(tmp_path):
    """The FIX: infer runs main.py over GT-free staged inputs; the hostile read fails."""
    held = tmp_path / "heldout"; _make_heldout(held)
    code = tmp_path / "code"; code.mkdir(); (code / "main.py").write_text(HOSTILE_MAIN)
    staged = tmp_path / "staged"; _stage_gtfree(held, staged)
    art = tmp_path / "art"; art.mkdir()

    # 1) GT-free inference job (what runs in the isolated container)
    env = dict(os.environ, LAB_DATA=str(staged), LAB_ARTIFACTS=str(art), LAB_CODE=str(code))
    subprocess.run([sys.executable, str(VO_REF / "infer_heldout.py")], env=env, check=True,
                   capture_output=True)
    assert not (staged / "seq_a" / "gt.txt").exists()          # staging stripped the labels
    # 2) scoring against the REAL gt
    ate = _score(held, art)
    assert ate > 1.0, (f"GT LEAK: hostile main.py scored ATE={ate:.4f} on staged inputs "
                       "(should be large) — staging failed to hide the labels.")


def test_old_leaky_model_would_have_leaked(tmp_path):
    """Pin the vulnerability the fix removes: running main.py with GT co-located DOES leak."""
    held = tmp_path / "heldout"; _make_heldout(held)
    code = tmp_path / "code"; code.mkdir(); (code / "main.py").write_text(HOSTILE_MAIN)
    # OLD model: run main.py with LAB_DATA = the *real* seq/input (gt.txt is its sibling)
    ates = []
    for sq in sorted(held.glob("seq_*")):
        run_art = Path(tempfile.mkdtemp())
        env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(run_art))
        subprocess.run([sys.executable, str(code / "main.py")], env=env, check=True,
                       capture_output=True)
        est = np.loadtxt(run_art / "traj.txt").reshape(-1, 3)
        gt = np.loadtxt(sq / "gt.txt").reshape(-1, 3)
        m = min(len(est), len(gt))
        ates.append(float(np.sqrt(((est[:m] - gt[:m]) ** 2).sum(1).mean())))
    assert np.mean(ates) < 1e-6   # confirms the old path leaked (hostile read GT verbatim)
