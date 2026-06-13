"""KITTI odometry (grayscale STEREO) data provider — the cross-domain generalization test.

TUM is indoor, hand-held, small-scale. KITTI is outdoor *driving*: long forward motion, large
scale, very different image statistics. If an agent's VO transfers here, that is real
generalization, not a tuned-to-one-dataset result.

Stereo (not mono): KITTI's calibrated stereo baseline makes absolute scale OBSERVABLE, so we
grade with SE(3) (metric, no scale freebie) — the same honest bar as the RGB-D experiment, on
a brand-new domain. The agent gets left+right frames; depth/scale must come from the stereo
baseline it is given.

On-disk contract (mirrors the RGB-D provider so the grader `eval_rgbd.py` is REUSED unchanged):
  dev (held_out=False):      left_%06d.png · right_%06d.png · intrinsics.txt(fx fy cx cy baseline)
  held-out (held_out=True):  seq_<NN>/{input/{left_*.png,right_*.png,intrinsics.txt}, gt.txt}

`gt.txt` is one `tx ty tz` (camera centre) per frame, OUTSIDE the input dir the solver sees.

The ~21.6 GB grayscale set is one monolithic download (no per-sequence option); fetched once
by `scripts/fetch_kitti_odometry.sh` and cached at ~/.cache/vo_lab/kitti, from which only the
used sequences are extracted. The materialization logic is unit-tested offline on a tiny
fabricated fixture (no 22 GB needed for tests)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

# KITTI odometry has ground-truth poses for sequences 00–10.
DEFAULT_CACHE = Path.home() / ".cache" / "vo_lab" / "kitti"


def kitti_datasets(dev: str = "00", heldout: tuple[str, ...] = ("05", "07")) -> list[DatasetRef]:
    """Dev sequence (agent authors here) + held-out sequences (graded, never authored on).
    Names are path/mount-safe (no ':') — they become cache dirs and Docker -v mount paths."""
    return [
        DatasetRef(name=f"vo-kitti-dev-{dev}", source=f"kitti:{dev}"),
        DatasetRef(name="vo-kitti-heldout-" + "_".join(heldout),
                   source=";".join(f"kitti:{s}" for s in heldout), held_out=True),
    ]


def _read_calib(calib_path: Path) -> tuple[float, float, float, float, float]:
    """Parse KITTI sequences/<NN>/calib.txt -> (fx, fy, cx, cy, baseline_m).
    P0 = left gray projection [fx 0 cx 0; 0 fy cy 0; 0 0 1 0]; P1 = right gray, with
    P1[0,3] = -fx * baseline. KITTI gray baseline is ~0.54 m."""
    P = {}
    for line in calib_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        if key in ("P0", "P1"):
            P[key] = np.array([float(x) for x in vals.split()], dtype=np.float64).reshape(3, 4)
    fx, fy, cx, cy = P["P0"][0, 0], P["P0"][1, 1], P["P0"][0, 2], P["P0"][1, 2]
    baseline = float(-P["P1"][0, 3] / fx)        # metres
    return float(fx), float(fy), float(cx), float(cy), baseline


def _read_poses(poses_path: Path) -> np.ndarray:
    """KITTI poses/<NN>.txt -> (N,3) camera centres. Each line is a row-major 3x4 [R|t]
    cam->world transform; the camera centre in world is its translation column (idx 3,7,11)."""
    M = np.loadtxt(poses_path).reshape(-1, 3, 4)
    return M[:, :, 3].copy()                     # (N,3) translation = camera centre


class KITTIOdomProvider:
    def __init__(self, *, dev: str = "00", heldout: tuple[str, ...] = ("05", "07"),
                 raw_root: str | Path | None = None, stride: int = 3,
                 max_frames: int | None = 300):
        self.dev = dev
        self.heldout = heldout
        self.raw_root = Path(raw_root) if raw_root else DEFAULT_CACHE
        self.stride = stride
        self.max_frames = max_frames

    def _seq_dir(self, seq: str) -> Path:
        d = self.raw_root / "dataset" / "sequences" / seq
        if not (d / "image_0").is_dir():
            raise RuntimeError(
                f"KITTI sequence {seq} not found at {d}. Fetch once with "
                f"`bash scripts/fetch_kitti_odometry.sh` (~21.6 GB).")
        return d

    def _indices(self, n: int) -> list[int]:
        idx = list(range(0, n, self.stride))
        if self.max_frames is not None:
            idx = idx[: self.max_frames]
        return idx

    def _materialize(self, seq: str, out_dir: Path, gt_path: Path | None = None) -> int:
        seq_dir = self._seq_dir(seq)
        left = sorted((seq_dir / "image_0").glob("*.png"))
        right = sorted((seq_dir / "image_1").glob("*.png"))
        if not left or len(left) != len(right):
            raise RuntimeError(f"KITTI {seq}: missing/mismatched stereo frames "
                               f"({len(left)} left, {len(right)} right)")
        idx = self._indices(len(left))
        ensure_dir(out_dir)
        fx, fy, cx, cy, baseline = _read_calib(seq_dir / "calib.txt")
        np.savetxt(out_dir / "intrinsics.txt",
                   np.array([fx, fy, cx, cy, baseline]), fmt="%.6f")
        for j, i in enumerate(idx):
            li = cv2.imread(str(left[i]), cv2.IMREAD_GRAYSCALE)
            ri = cv2.imread(str(right[i]), cv2.IMREAD_GRAYSCALE)
            if li is None or ri is None:
                raise RuntimeError(f"KITTI {seq}: cannot read frame {i}")
            cv2.imwrite(str(out_dir / f"left_{j:06d}.png"), li)
            cv2.imwrite(str(out_dir / f"right_{j:06d}.png"), ri)
        if gt_path is not None:
            poses_file = self.raw_root / "dataset" / "poses" / f"{seq}.txt"
            centres = _read_poses(poses_file)
            np.savetxt(gt_path, centres[idx], fmt="%.6f")
            # also write the FULL 3x4 cam->world poses (KITTI format) for the OFFICIAL metric
            # (t_err + r_err need orientations); name shares the 'gt*.txt' label glob so the
            # GT-isolation staging strips it too. gt_path is .../gt.txt -> .../gt_poses.txt.
            full = np.loadtxt(poses_file).reshape(-1, 12)[idx]
            np.savetxt(Path(gt_path).with_name("gt_poses.txt"), full, fmt="%.8e")
        return len(idx)

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            for seq in self.heldout:
                sub = ensure_dir(dest / f"seq_{seq}")
                self._materialize(seq, sub / "input", gt_path=sub / "gt.txt")
        else:
            self._materialize(self.dev, dest, gt_path=None)
