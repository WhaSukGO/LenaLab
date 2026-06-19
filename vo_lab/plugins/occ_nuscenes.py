"""nuScenes 3D-occupancy data provider (the 6th domain). Mirrors the BEV provider: train split
carries the full npz (cams + calib + occ GT), test_input strips the GT, held-out split is the
secret per-sample voxel GT (<token>_occ.npy). Source: scripts/prep_nuscenes_occ.py -> ~/.cache/vo_lab/occ."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from lab.models import DatasetRef
from lab.util import ensure_dir

DEFAULT_CACHE = Path.home() / ".cache" / "vo_lab" / "occ"


def occ_datasets() -> list[DatasetRef]:
    return [DatasetRef(name="occ-nuscenes-train", source="occ:train"),
            DatasetRef(name="occ-nuscenes-heldout", source="occ:val", held_out=True)]


class NuScenesOccProvider:
    def __init__(self, *, cache: str | Path | None = None, train_max=None, test_max=None):
        self.cache = Path(cache) if cache else DEFAULT_CACHE
        self.train_max = train_max; self.test_max = test_max

    def _tokens(self, split, cap):
        d = self.cache / split
        if not d.is_dir():
            raise RuntimeError(f"prepped occ cache missing at {d}. Build it with "
                               f"`python scripts/prep_nuscenes_occ.py <nuscenes_root> {self.cache}` (vo-bev:1).")
        files = sorted(d.glob("*.npz"))
        return files[:cap] if cap is not None else files

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            ensure_dir(dest)
            for f in self._tokens("val", self.test_max):
                np.save(dest / f"{f.stem}_occ.npy", np.load(f)["occ"].astype(np.uint8))
        else:
            tr = ensure_dir(dest / "train")
            for f in self._tokens("train", self.train_max):
                shutil.copy(f, tr / f.name)
            ti = ensure_dir(dest / "test_input")
            for f in self._tokens("val", self.test_max):
                d = np.load(f)
                np.savez_compressed(ti / f.name, imgs=d["imgs"], intrins=d["intrins"], cam2ego=d["cam2ego"])
