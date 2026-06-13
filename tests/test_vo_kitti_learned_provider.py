"""KITTI learned-VO provider — verified OFFLINE on a fabricated fixture (no 22 GB, no GPU).
Checks: train split exposes frames + GT poses per sequence (the supervision), test inputs
have frames but NO labels, and the held-out split carries only the secret gt.txt."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vo_lab  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from lab.models import DatasetRef  # noqa: E402
from vo_lab.plugins.vo_kitti_learned import KITTILearnedProvider  # noqa: E402


def _fixture(raw: Path, seqs=("00", "02", "05"), n: int = 8):
    fx, fy, cx, cy, base = 700.0, 700.0, 600.0, 180.0, 0.54
    for s in seqs:
        seq = raw / "dataset" / "sequences" / s
        (seq / "image_0").mkdir(parents=True); (seq / "image_1").mkdir()
        for i in range(n):
            cv2.imwrite(str(seq / f"image_0/{i:06d}.png"), np.full((32, 64), i * 4, np.uint8))
            cv2.imwrite(str(seq / f"image_1/{i:06d}.png"), np.full((32, 64), i * 4, np.uint8))
        (seq / "calib.txt").write_text(
            f"P0: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0\n"
            f"P1: {fx} 0 {cx} {-fx*base} 0 {fy} {cy} 0 0 0 1 0\n")
        (raw / "dataset" / "poses").mkdir(parents=True, exist_ok=True)
        (raw / "dataset" / "poses" / f"{s}.txt").write_text(
            "\n".join(f"1 0 0 {i} 0 1 0 {2*i} 0 0 1 {3*i}" for i in range(n)))


def test_learned_provider_train_test_split(tmp_path):
    _fixture(tmp_path / "raw")
    prov = KITTILearnedProvider(train=("00", "02"), test=("05",), raw_root=tmp_path / "raw",
                                train_stride=1, train_max=None, test_stride=1, test_max=None)

    vis = tmp_path / "vis"; vis.mkdir()
    prov.fetch(DatasetRef("train", "s"), vis)
    # train: per-seq frames + poses (the supervision is legitimately visible)
    for s in ("00", "02"):
        assert (vis / "train" / f"seq_{s}" / "left_000000.png").exists()
        poses = np.loadtxt(vis / "train" / f"seq_{s}" / "poses.txt")
        assert poses.shape == (8, 12)                          # full 3x4 per frame
    # test inputs: frames present, NO labels
    assert (vis / "test_input" / "seq_05" / "left_000000.png").exists()
    assert not (vis / "test_input" / "seq_05" / "gt.txt").exists()
    assert not (vis / "test_input" / "seq_05" / "poses.txt").exists()

    ho = tmp_path / "ho"; ho.mkdir()
    prov.fetch(DatasetRef("test", "s", held_out=True), ho)
    gt = np.loadtxt(ho / "seq_05" / "gt.txt")                  # only the secret labels
    assert gt.shape == (8, 3) and np.allclose(gt[2], [2.0, 4.0, 6.0])
    assert not (ho / "seq_05" / "left_000000.png").exists()    # no frames in held-out
