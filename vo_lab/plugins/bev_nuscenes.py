"""nuScenes BEV data provider (the multi-camera perception track — a SECOND problem class).

Like learned VO, supervised BEV legitimately trains on a TRAIN split's labels; only the TEST
labels stay secret. The solver-visible split contains:
  - train/<token>.npz       : imgs(6,H,W,3) + intrins(6,3,3) + cam2ego(6,4,4) + bev(200,200)
                              (the full training signal -- 6 surround cams + calib + GT occupancy)
  - test_input/<token>.npz  : imgs + intrins + cam2ego, NO bev  (test inputs, not labels)
and the HELD-OUT split (grader-only) contains just:
  - <token>_bev.npy         : the secret BEV vehicle-occupancy GT for each test sample

This is standard ML eval: train on (X_train, y_train), predict on X_test, score on held-out
y_test. The trainer is a JOB on the GPU (gpu_lease + CUDA image) -- wall-clock, not tokens.

Source data is produced once by `scripts/prep_nuscenes_bev.py` (nuScenes mini -> npz cache at
~/.cache/vo_lab/bev). Held-out = official nuScenes mini_val scenes (disjoint from mini_train)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

DEFAULT_CACHE = Path.home() / ".cache" / "vo_lab" / "bev"


def bev_datasets() -> list[DatasetRef]:
    return [
        DatasetRef(name="bev-nuscenes-train", source="bev:train"),
        DatasetRef(name="bev-nuscenes-heldout", source="bev:val", held_out=True),
    ]


class NuScenesBEVProvider:
    def __init__(self, *, cache: str | Path | None = None,
                 train_max: int | None = None, test_max: int | None = None):
        self.cache = Path(cache) if cache else DEFAULT_CACHE
        self.train_max = train_max
        self.test_max = test_max

    def _tokens(self, split: str, cap: int | None) -> list[Path]:
        d = self.cache / split
        if not d.is_dir():
            raise RuntimeError(f"prepped BEV cache missing at {d}. Build it once with "
                               f"`python scripts/prep_nuscenes_bev.py <nuscenes_root> {self.cache}` "
                               f"(in vo-bev:1).")
        files = sorted(d.glob("*.npz"))
        return files[:cap] if cap is not None else files

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            # grader-only: the secret BEV occupancy GT per held-out (val) sample
            ensure_dir(dest)
            for f in self._tokens("val", self.test_max):
                tok = f.stem
                np.save(dest / f"{tok}_bev.npy", np.load(f)["bev"].astype(np.uint8))
        else:
            # train split: full npz (cams + calib + GT occupancy) -- the supervised signal
            tr = ensure_dir(dest / "train")
            for f in self._tokens("train", self.train_max):
                shutil.copy(f, tr / f.name)
            # test INPUT: same val samples but with the BEV label STRIPPED
            ti = ensure_dir(dest / "test_input")
            for f in self._tokens("val", self.test_max):
                d = np.load(f)
                np.savez_compressed(ti / f.name, imgs=d["imgs"],
                                    intrins=d["intrins"], cam2ego=d["cam2ego"])
