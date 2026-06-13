"""KITTI LEARNED-VO data provider (the GPU / learned-research track).

Unlike classical VO (where the solver never sees ANY ground truth), supervised learned VO
legitimately trains on a TRAIN split's poses — only the TEST labels stay secret. So the
solver-visible split here contains:
  - train/      : left frames + poses.txt (GT 3x4 per frame) + intrinsics.txt   (training signal)
  - test_input/seq_<s>/ : left frames + intrinsics.txt, NO gt                    (you may see test
                                                                                  inputs, not labels)
and the HELD-OUT split (grader-only) contains just:
  - seq_<s>/gt.txt : camera centres for each test sequence (the secret labels)

This is exactly standard ML evaluation: train on (X_train, y_train), predict on X_test, get
scored on the held-out y_test. The trainer is a JOB on the GPU (gpu_lease + CUDA image) — it
consumes wall-clock, not the model's token budget.

Reuses the KITTI odometry download (scripts/fetch_kitti_odometry.sh); offline-tested on a
fabricated fixture."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

from .vo_kitti import DEFAULT_CACHE, _read_calib, _read_poses


def kitti_learned_datasets(train: tuple[str, ...] = ("00", "02", "06", "08", "09"),
                           test: tuple[str, ...] = ("05", "07")) -> list[DatasetRef]:
    return [
        DatasetRef(name="vo-kitti-learn-train-" + "_".join(train), source="kitti-learn:train"),
        DatasetRef(name="vo-kitti-learn-test-" + "_".join(test),
                   source=";".join(f"kitti-learn:{s}" for s in test), held_out=True),
    ]


class KITTILearnedProvider:
    def __init__(self, *, train: tuple[str, ...] = ("00", "02", "06", "08", "09"),
                 test: tuple[str, ...] = ("05", "07"),
                 raw_root: str | Path | None = None, train_stride: int = 2,
                 train_max: int | None = 1200, test_stride: int = 3, test_max: int | None = 300):
        self.train = (train,) if isinstance(train, str) else tuple(train)
        self.test = test
        self.raw_root = Path(raw_root) if raw_root else DEFAULT_CACHE
        self.train_stride = train_stride
        self.train_max = train_max
        self.test_stride = test_stride
        self.test_max = test_max

    def _seq_dir(self, seq: str) -> Path:
        d = self.raw_root / "dataset" / "sequences" / seq
        if not (d / "image_0").is_dir():
            raise RuntimeError(f"KITTI sequence {seq} not found at {d}. Fetch once with "
                               f"`bash scripts/fetch_kitti_odometry.sh`.")
        return d

    def _indices(self, n: int, stride: int, cap: int | None) -> list[int]:
        idx = list(range(0, n, stride))
        return idx[:cap] if cap is not None else idx

    def _write_frames(self, seq: str, idx: list[int], out: Path) -> None:
        seq_dir = self._seq_dir(seq)
        left = sorted((seq_dir / "image_0").glob("*.png"))
        ensure_dir(out)
        fx, fy, cx, cy, _b = _read_calib(seq_dir / "calib.txt")
        np.savetxt(out / "intrinsics.txt", np.array([fx, fy, cx, cy]), fmt="%.6f")
        for j, i in enumerate(idx):
            im = cv2.imread(str(left[i]), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out / f"left_{j:06d}.png"), im)

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            # grader-only: camera centres for each test sequence
            for seq in self.test:
                n = len(list((self._seq_dir(seq) / "image_0").glob("*.png")))
                idx = self._indices(n, self.test_stride, self.test_max)
                centres = _read_poses(self.raw_root / "dataset" / "poses" / f"{seq}.txt")[idx]
                sub = ensure_dir(dest / f"seq_{seq}")
                np.savetxt(sub / "gt.txt", centres, fmt="%.6f")
        else:
            # train split: each train sequence in its OWN subdir (frames + GT poses), so the
            # trainer forms relative targets only between consecutive frames of the same seq.
            for seq in self.train:
                tn = len(list((self._seq_dir(seq) / "image_0").glob("*.png")))
                tidx = self._indices(tn, self.train_stride, self.train_max)
                self._write_frames(seq, tidx, dest / "train" / f"seq_{seq}")
                # full 3x4 cam->world poses (12 cols/frame) for relative rot+trans targets.
                allp = np.loadtxt(self.raw_root / "dataset" / "poses" / f"{seq}.txt")
                np.savetxt(dest / "train" / f"seq_{seq}" / "poses.txt", allp[tidx], fmt="%.8e")
            # test INPUT frames (no labels)
            for seq in self.test:
                n = len(list((self._seq_dir(seq) / "image_0").glob("*.png")))
                idx = self._indices(n, self.test_stride, self.test_max)
                self._write_frames(seq, idx, dest / "test_input" / f"seq_{seq}")
