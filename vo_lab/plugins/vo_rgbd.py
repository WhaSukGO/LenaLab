"""RGB-D + generalization data provider (the improvement experiment).

Two changes vs the monocular `TUMDatasetProvider`:
  1. EXPOSE DEPTH — materializes the TUM depth channel so the solver can do RGB-D VO, where
     absolute scale is OBSERVABLE (no monocular scale ambiguity). depth_%04d.png is the raw
     16-bit TUM depth (metres = pixel / 5000); the scale is written into intrinsics.txt.
  2. GENERALIZATION HELD-OUT — the held-out split is one or more *different sequences* than
     the dev sequence the agent authors against. Each held-out sequence is materialized as
     its own subdir `seq_<name>/` (frames + depth + gt.txt + intrinsics.txt). The grader runs
     the agent's code on each unseen sequence, so a result must generalize, not overfit one
     scene.

On-disk contract:
  dev (held_out=False):  frame_%04d.png · depth_%04d.png · intrinsics.txt(fx fy cx cy depth_scale)
  held-out (held_out=True): seq_<name>/{frame_*.png, depth_*.png, gt.txt, intrinsics.txt}

The actual downloads (~0.5 GB/sequence) happen on the user's machine, cached once at
~/.cache/vo_lab/tum; the association/materialization logic is unit-tested offline on a tiny
fabricated fixture."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

from .vo_real import _read_pairs

TUM_BASE = "https://cvg.cit.tum.de/rgbd/dataset"
TUM_DEPTH_SCALE = 5000.0  # TUM convention: depth_metres = depth_png_uint16 / 5000
# (fx, fy, cx, cy) per freiburg camera; images are 640x480.
TUM_INTRINSICS = {"freiburg1": (517.306408, 516.469215, 318.643040, 255.313989),
                  "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764)}

# Sequence catalogue: name -> (url, freiburg camera).
SEQS = {
    "fr1_xyz":  (f"{TUM_BASE}/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",  "freiburg1"),
    "fr1_desk": (f"{TUM_BASE}/freiburg1/rgbd_dataset_freiburg1_desk.tgz", "freiburg1"),
    "fr2_xyz":  (f"{TUM_BASE}/freiburg2/rgbd_dataset_freiburg2_xyz.tgz",  "freiburg2"),
}


def rgbd_datasets(dev: str = "fr1_xyz", heldout: tuple[str, ...] = ("fr1_desk",)) -> list[DatasetRef]:
    """Dev sequence (agent authors here) + held-out sequences (graded, never authored on)."""
    return [
        DatasetRef(name=f"vo-rgbd-dev:{dev}", source=SEQS[dev][0]),
        DatasetRef(name="vo-rgbd-heldout:" + "+".join(heldout),
                   source=";".join(SEQS[s][0] for s in heldout), held_out=True),
    ]


class TUMRGBDProvider:
    def __init__(self, *, dev: str = "fr1_xyz", heldout: tuple[str, ...] = ("fr1_desk",),
                 raw_root: str | Path | None = None, stride: int = 1,
                 max_frames: int | None = 200, assoc_max_dt: float = 0.02):
        self.dev = dev
        self.heldout = heldout
        self.raw_root = Path(raw_root) if raw_root else (Path.home() / ".cache" / "vo_lab" / "tum")
        self.stride = stride
        self.max_frames = max_frames
        self.assoc_max_dt = assoc_max_dt

    # --- download once -------------------------------------------------------
    def _ensure_raw(self, seq_key: str) -> tuple[Path, str]:
        url, cam = SEQS[seq_key]
        name = Path(url).name[:-4]
        root = ensure_dir(self.raw_root)
        seq_dir = root / name
        if not ((seq_dir / "groundtruth.txt").exists() and (seq_dir / "rgb.txt").exists()):
            tgz = root / f"{name}.tgz"
            if not tgz.exists():
                print(f"downloading {url} (~0.5 GB, once) ...")
                urllib.request.urlretrieve(url, tgz)
            with tarfile.open(tgz, "r:gz") as t:
                t.extractall(root)
        return seq_dir, cam

    # --- 3-way timestamp association (rgb <-> depth <-> gt) ------------------
    def _associate(self, seq_dir: Path, *, need_gt: bool):
        rgb = _read_pairs(seq_dir / "rgb.txt")
        depth = _read_pairs(seq_dir / "depth.txt")
        d_ts = np.array([t for t, _ in depth]) if depth else np.zeros(0)
        gt = _read_pairs(seq_dir / "groundtruth.txt")
        g_ts = np.array([t for t, _ in gt]) if gt else np.zeros(0)
        g_xyz = np.array([[float(v) for v in s.split()[:3]] for _, s in gt]) if gt else np.zeros((0, 3))
        out = []  # (rgb_fname, depth_fname, xyz_or_None)
        for ts, fname in rgb:
            j = int(np.argmin(np.abs(d_ts - ts))) if len(d_ts) else -1
            if j < 0 or abs(d_ts[j] - ts) > self.assoc_max_dt:
                continue
            dfile = depth[j][1]
            xyz = None
            if need_gt:
                k = int(np.argmin(np.abs(g_ts - ts))) if len(g_ts) else -1
                if k < 0 or abs(g_ts[k] - ts) > self.assoc_max_dt:
                    continue
                xyz = g_xyz[k]
            out.append((fname, dfile, xyz))
        out = out[:: self.stride]
        if self.max_frames is not None:
            out = out[: self.max_frames]
        return out

    def _materialize(self, seq_key: str, frames_dir: Path, gt_path: Path | None = None) -> int:
        """Write frames+depth+intrinsics into frames_dir; if gt_path, write GT there (kept
        OUTSIDE frames_dir so the solver's code never receives the ground truth)."""
        seq_dir, cam = self._ensure_raw(seq_key)
        matched = self._associate(seq_dir, need_gt=gt_path is not None)
        if not matched:
            raise RuntimeError(f"{seq_key}: no rgb/depth/gt associations within {self.assoc_max_dt}s")
        ensure_dir(frames_dir)
        fx, fy, cx, cy = TUM_INTRINSICS[cam]
        np.savetxt(frames_dir / "intrinsics.txt",
                   np.array([fx, fy, cx, cy, TUM_DEPTH_SCALE]), fmt="%.6f")
        gts = []
        for i, (rgbf, depthf, xyz) in enumerate(matched):
            g = cv2.imread(str(seq_dir / rgbf), cv2.IMREAD_GRAYSCALE)
            d = cv2.imread(str(seq_dir / depthf), cv2.IMREAD_UNCHANGED)  # 16-bit depth
            if g is None or d is None:
                raise RuntimeError(f"{seq_key}: cannot read {rgbf} / {depthf}")
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), g)
            cv2.imwrite(str(frames_dir / f"depth_{i:04d}.png"), d)
            if gt_path is not None:
                gts.append(xyz)
        if gt_path is not None:
            np.savetxt(gt_path, np.array(gts), fmt="%.6f")
        return len(matched)

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            # one subdir per held-out sequence: seq_<name>/input/ (given to the solver's
            # code) + seq_<name>/gt.txt (grader-only, never in the code's LAB_DATA)
            for seq_key in self.heldout:
                sub = ensure_dir(dest / f"seq_{seq_key}")
                self._materialize(seq_key, sub / "input", gt_path=sub / "gt.txt")
        else:
            self._materialize(self.dev, dest, gt_path=None)
