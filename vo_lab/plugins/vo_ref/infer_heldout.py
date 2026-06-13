"""Held-out INFERENCE runner — runs in a GT-FREE job (the evaluator stages the held-out
inputs with label files stripped, so ground truth is NOT mounted here). For each held-out
sequence it runs the solver's main.py and saves that sequence's trajectory. The scoring
grader (eval*.py) then reads these trajectories + the real GT in a SEPARATE job that never
runs the solver — closing the GT-read leak.

$LAB_DATA = staged held-out inputs (seq_<n>/input/, NO gt.txt).
$LAB_CODE/main.py = the solver's authored algorithm (single-sequence contract, unchanged).
Writes $LAB_ARTIFACTS/traj_<n>.txt per sequence."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    data = Path(os.environ["LAB_DATA"])               # GT-free staged inputs
    art = Path(os.environ["LAB_ARTIFACTS"]); art.mkdir(parents=True, exist_ok=True)
    code = Path(os.environ["LAB_CODE"])
    seqs = sorted(p for p in data.glob("seq_*") if (p / "input").is_dir())
    if not seqs:
        print("infer_heldout: no seq_*/input under LAB_DATA", file=sys.stderr); return 2
    for sq in seqs:
        s = sq.name.replace("seq_", "")
        tmp = Path(tempfile.mkdtemp())
        env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(tmp))
        try:
            subprocess.run([sys.executable, str(code / "main.py")], env=env,
                           timeout=900, check=True)
            produced = tmp / "traj.txt"
            if produced.exists():
                shutil.copy(produced, art / f"traj_{s}.txt")
                print(f"infer_heldout: wrote traj_{s}.txt")
            else:
                print(f"infer_heldout: main.py produced no traj.txt for seq {s}", file=sys.stderr)
            poses = tmp / "poses.txt"            # optional full 6-DoF poses (for official metric)
            if poses.exists():
                shutil.copy(poses, art / f"poses_{s}.txt")
        except Exception as e:  # noqa: BLE001 - a failed seq -> no traj_<s>; grader scores it 1e9
            print(f"infer_heldout: seq {s} failed: {str(e)[:160]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
