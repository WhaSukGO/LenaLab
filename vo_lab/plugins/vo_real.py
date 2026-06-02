"""Real-data VO provider — TUM RGB-D (the minimal standard monocular VO/SLAM benchmark).

KITTI's odometry images are a single ~22 GB download (no per-sequence option), which
violates the "minimum download" constraint. TUM RGB-D ships per-sequence (~0.5 GB) with
ground-truth trajectories, and `freiburg1_xyz` is the recommended easy translational
sequence — the right first real monocular sequence.

Design: the provider DOWNLOADS once (a harness job, zero tokens; cached by DatasetCache)
and PRE-ASSOCIATES ground-truth poses to RGB frames by timestamp, emitting the SAME on-disk
contract as the synthetic provider:
  visible (held_out=False): frame_%04d.png (grayscale) + intrinsics.txt (fx fy cx cy)
  held-out (held_out=True): gt.txt with one `tx ty tz` per frame, index-aligned
So vo_ref/run.py and vo_ref/eval.py work unchanged. The solver sees only frames; the
ground-truth trajectory stays evaluator-only. Absolute scale is unobservable (monocular)
and is handled by eval.py's Sim(3) alignment.

The actual 0.5 GB download happens on the user's machine; the association/parse logic here
is unit-tested offline against a tiny fabricated TUM-format fixture."""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

# freiburg1_xyz: ~0.47 GB, simple translatory motion (recommended for debugging).
TUM_FR1_XYZ_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
# Calibrated Freiburg-1 RGB intrinsics (fx, fy, cx, cy); images are 640x480.
TUM_FR1_INTRINSICS = (517.306408, 516.469215, 318.643040, 255.313989)


def tum_datasets() -> list[DatasetRef]:
    return [
        DatasetRef(name="vo-tum-frames", source=TUM_FR1_XYZ_URL),
        DatasetRef(name="vo-tum-gt", source=TUM_FR1_XYZ_URL, held_out=True),
    ]


def _read_pairs(path: Path) -> list[tuple[float, str]]:
    """Read a TUM index file (`timestamp value...`), skipping # comments."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append((float(parts[0]), " ".join(parts[1:])))
    return out


class TUMDatasetProvider:
    """Materializes a TUM RGB-D sequence into the lab's frame/gt contract.

    max_frames + stride keep the first run light (default ~200 frames). intrinsics default
    to the fr1 calibration. assoc_max_dt is the max timestamp gap (s) for an rgb<->gt match."""

    def __init__(self, *, url: str = TUM_FR1_XYZ_URL, raw_root: str | Path | None = None,
                 intrinsics=TUM_FR1_INTRINSICS, stride: int = 1, max_frames: int | None = 200,
                 assoc_max_dt: float = 0.02):
        self.url = url
        # Shared lifetime cache so the ~0.5 GB download happens ONCE, not once per DatasetRef
        # (frames + gt) and not again on re-runs. Override with raw_root if desired.
        self.raw_root = Path(raw_root) if raw_root else (Path.home() / ".cache" / "vo_lab" / "tum")
        self.intrinsics = intrinsics
        self.stride = stride
        self.max_frames = max_frames
        self.assoc_max_dt = assoc_max_dt
        self._seq_name = Path(url).name[:-4] if url.endswith(".tgz") else Path(url).name

    # --- download once -------------------------------------------------------
    def _ensure_raw(self, dest_parent: Path) -> Path:
        root = self.raw_root or (dest_parent / "_raw")
        ensure_dir(root)
        seq_dir = root / self._seq_name
        if (seq_dir / "groundtruth.txt").exists() and (seq_dir / "rgb.txt").exists():
            return seq_dir
        tgz = root / f"{self._seq_name}.tgz"
        if not tgz.exists():
            print(f"downloading {self.url} (~0.5 GB, once) ...")
            urllib.request.urlretrieve(self.url, tgz)
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(root)
        return seq_dir

    # --- timestamp association (deterministic, harness-side) -----------------
    def _associate(self, seq_dir: Path):
        rgb = _read_pairs(seq_dir / "rgb.txt")                 # (ts, "rgb/<ts>.png")
        gt = _read_pairs(seq_dir / "groundtruth.txt")          # (ts, "tx ty tz qx qy qz qw")
        gt_ts = np.array([t for t, _ in gt])
        gt_xyz = np.array([[float(v) for v in s.split()[:3]] for _, s in gt])
        matched = []  # (rgb_fname, xyz)
        for ts, fname in rgb:
            j = int(np.argmin(np.abs(gt_ts - ts)))
            if abs(gt_ts[j] - ts) <= self.assoc_max_dt:
                matched.append((fname, gt_xyz[j]))
        matched = matched[:: self.stride]
        if self.max_frames is not None:
            matched = matched[: self.max_frames]
        return matched

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        seq_dir = self._ensure_raw(dest.parent)
        matched = self._associate(seq_dir)
        if not matched:
            raise RuntimeError(f"no rgb<->gt associations within {self.assoc_max_dt}s")
        if ref.held_out:
            gt = np.array([xyz for _, xyz in matched])
            np.savetxt(dest / "gt.txt", gt, fmt="%.6f")
        else:
            fx, fy, cx, cy = self.intrinsics
            np.savetxt(dest / "intrinsics.txt", np.array([fx, fy, cx, cy]), fmt="%.6f")
            for i, (fname, _xyz) in enumerate(matched):
                img = cv2.imread(str(seq_dir / fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"could not read frame {fname}")
                cv2.imwrite(str(dest / f"frame_{i:04d}.png"), img)
