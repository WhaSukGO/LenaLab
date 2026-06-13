"""Loop-detection ORACLE provider — for the M2 decomposition experiment.

Wraps the KITTI provider and writes a `loops.txt` oracle into each sequence's input dir: the
ground-truth loop pairs (frame i revisited at frame j) plus their GT relative pose. This is a
LABELED SCAFFOLD INPUT (like the locked front-end), NOT the answer: the agent gets sparse loop
constraints but never the held-out trajectory. It isolates the sub-skill question — given CORRECT
loops, can the agent author the pose-graph optimisation? — from loop DETECTION.

loops.txt format (one loop per line):  i j  r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
where (r|t) is the 3x4 relative pose T_ij = inv(Twc_i) @ Twc_j (so Twc_j ≈ Twc_i @ T_ij).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .vo_kitti import KITTIOdomProvider


def compute_loop_constraints(poses_3x4: np.ndarray, *, min_gap: int = 80, max_dist: float = 15.0,
                             max_loops: int = 8) -> list[tuple[int, int, np.ndarray]]:
    """poses_3x4: (n,12) row-major 3x4 cam->world per frame. Return up to max_loops strongest
    revisit constraints (i, j, rel_3x4) with j-i >= min_gap and centre distance < max_dist."""
    P = poses_3x4.reshape(-1, 3, 4)
    n = len(P)
    centres = P[:, :, 3]
    cands = []
    for j in range(min_gap + 1, n):
        upto = j - min_gap
        if upto < 1:
            continue
        d = np.linalg.norm(centres[:upto] - centres[j], axis=1)
        i = int(d.argmin())
        if d[i] < max_dist:
            cands.append((i, j, float(d[i])))
    cands.sort(key=lambda c: c[2])                       # strongest (closest) first
    chosen, used_j = [], []
    for i, j, _ in cands:                                # spread them out: skip near-duplicate j
        if all(abs(j - jj) > min_gap // 2 for jj in used_j):
            Ti = np.eye(4); Ti[:3, :] = P[i]
            Tj = np.eye(4); Tj[:3, :] = P[j]
            rel = (np.linalg.inv(Ti) @ Tj)[:3, :].reshape(-1)
            chosen.append((i, j, rel)); used_j.append(j)
        if len(chosen) >= max_loops:
            break
    return chosen


def _write_loops(out_dir: Path, poses_3x4: np.ndarray, **kw) -> int:
    loops = compute_loop_constraints(poses_3x4, **kw)
    lines = [f"{i} {j} " + " ".join(f"{v:.8e}" for v in rel) for i, j, rel in loops]
    (Path(out_dir) / "loops.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(loops)


class LoopOracleKITTIProvider(KITTIOdomProvider):
    """KITTI provider that also writes loops.txt (the GT loop-detection oracle) into each seq input."""

    def __init__(self, *, min_gap: int = 80, max_dist: float = 15.0, max_loops: int = 8, **kw):
        super().__init__(**kw)
        self._loop_kw = dict(min_gap=min_gap, max_dist=max_dist, max_loops=max_loops)

    def _materialize(self, seq: str, out_dir: Path, gt_path: Path | None = None) -> int:
        n = super()._materialize(seq, out_dir, gt_path=gt_path)
        # strided GT poses in the agent's frame numbering (same idx the materialiser used)
        poses_file = self.raw_root / "dataset" / "poses" / f"{seq}.txt"
        nframes = len(sorted((self._seq_dir(seq) / "image_0").glob("*.png")))
        idx = self._indices(nframes)
        full = np.loadtxt(poses_file).reshape(-1, 12)[idx]
        k = _write_loops(out_dir, full, **self._loop_kw)   # loops.txt -> survives GT-staging (not gt*)
        return n
