"""Smart-space 2D floor-occupancy data provider (the 7th domain). Mirrors the occupancy provider:
train split carries the full npz (cams + cam_proj + bev GT), test_input strips the GT, held-out split
is the secret per-sample floor GT (<token>_bev.npy). Source: scripts/prep_smartspace.py ->
~/.cache/vo_lab/smartspace_occ. Held-out = the scene's last 30% of time (per-space self-verification)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

DEFAULT_CACHE = Path.home() / ".cache" / "vo_lab" / "smartspace_occ"


def smartspace_datasets() -> list[DatasetRef]:
    return [DatasetRef(name="smartspace-train", source="smartspace:train"),
            DatasetRef(name="smartspace-heldout", source="smartspace:val", held_out=True)]


class SmartSpaceProvider:
    def __init__(self, *, cache: str | Path | None = None, train_max=None, test_max=None):
        self.cache = Path(cache) if cache else DEFAULT_CACHE
        self.train_max = train_max; self.test_max = test_max

    def _tokens(self, split, cap):
        d = self.cache / split
        if not d.is_dir():
            raise RuntimeError(f"prepped smartspace cache missing at {d}. Build it with "
                               f"`python scripts/prep_smartspace.py <scene_dir> {self.cache}`.")
        files = sorted(d.glob("*.npz"))
        return files[:cap] if cap is not None else files

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            ensure_dir(dest)
            for f in self._tokens("val", self.test_max):
                np.save(dest / f"{f.stem}_bev.npy", np.load(f)["bev"].astype(np.uint8))
        else:
            tr = ensure_dir(dest / "train")
            for f in self._tokens("train", self.train_max):
                shutil.copy(f, tr / f.name)
            ti = ensure_dir(dest / "test_input")
            for f in self._tokens("val", self.test_max):
                d = np.load(f)
                np.savez_compressed(ti / f.name, imgs=d["imgs"], intrins=d["intrins"],
                                    cam_proj=d["cam_proj"], grid_bounds=d["grid_bounds"])
