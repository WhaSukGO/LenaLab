"""KITTI stereo provider — verified OFFLINE on a fabricated KITTI-odometry fixture (no 22 GB).
Checks: stereo pairs are exposed, intrinsics carry the metric baseline, camera centres come
from the pose translation column, and held-out is laid out as seq_*/input (left+right, NO gt)
+ seq_*/gt.txt (grader-only) so the solver can't read GT."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vo_lab  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from lab.models import DatasetRef  # noqa: E402
from vo_lab.plugins.vo_kitti import KITTIOdomProvider  # noqa: E402


def _fixture(raw: Path, seqs=("00", "05"), n: int = 6):
    """Fabricate the KITTI odometry on-disk layout: sequences/<NN>/{image_0,image_1,calib.txt}
    and poses/<NN>.txt (3x4 row-major cam->world per frame)."""
    fx, fy, cx, cy, base = 700.0, 700.0, 600.0, 180.0, 0.54
    for s in seqs:
        seq = raw / "dataset" / "sequences" / s
        (seq / "image_0").mkdir(parents=True); (seq / "image_1").mkdir()
        for i in range(n):
            cv2.imwrite(str(seq / f"image_0/{i:06d}.png"), np.full((32, 64), i * 5, np.uint8))
            cv2.imwrite(str(seq / f"image_1/{i:06d}.png"), np.full((32, 64), i * 5, np.uint8))
        # KITTI calib: P0 (left), P1 (right) with P1[0,3] = -fx*baseline
        p0 = f"P0: {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0"
        p1 = f"P1: {fx} 0 {cx} {-fx*base} 0 {fy} {cy} 0 0 0 1 0"
        (seq / "calib.txt").write_text(p0 + "\n" + p1 + "\n")
        # poses: identity rotation, translation = (i, 2i, 3i)
        (raw / "dataset" / "poses").mkdir(parents=True, exist_ok=True)
        lines = [f"1 0 0 {i} 0 1 0 {2*i} 0 0 1 {3*i}" for i in range(n)]
        (raw / "dataset" / "poses" / f"{s}.txt").write_text("\n".join(lines))


def test_kitti_provider_exposes_stereo_and_isolates_gt(tmp_path):
    _fixture(tmp_path / "raw")
    prov = KITTIOdomProvider(dev="00", heldout=("05",), raw_root=tmp_path / "raw",
                             stride=1, max_frames=None)

    dev = tmp_path / "dev"; dev.mkdir()
    prov.fetch(DatasetRef("vo-kitti-dev-00", "s"), dev)
    assert len(list(dev.glob("left_*.png"))) == 6
    assert len(list(dev.glob("right_*.png"))) == 6              # stereo pair exposed
    intr = np.loadtxt(dev / "intrinsics.txt")
    assert intr.shape == (5,) and abs(intr[4] - 0.54) < 1e-6    # metric baseline carried
    assert abs(intr[0] - 700.0) < 1e-6                          # fx from P0
    assert not (dev / "gt.txt").exists()                        # dev has no GT

    ho = tmp_path / "ho"; ho.mkdir()
    prov.fetch(DatasetRef("vo-kitti-heldout-05", "s", held_out=True), ho)
    sub = ho / "seq_05"
    assert (sub / "input" / "left_000000.png").exists()
    assert (sub / "input" / "right_000000.png").exists()
    assert not (sub / "input" / "gt.txt").exists()              # GT NOT in solver's input
    gt = np.loadtxt(sub / "gt.txt")                              # GT is grader-only, alongside
    assert gt.shape == (6, 3)
    assert np.allclose(gt[2], [2.0, 4.0, 6.0])                  # camera centre = pose translation


def test_kitti_stride_and_maxframes(tmp_path):
    _fixture(tmp_path / "raw", seqs=("00",), n=10)
    prov = KITTIOdomProvider(dev="00", heldout=("00",), raw_root=tmp_path / "raw",
                             stride=2, max_frames=3)
    dev = tmp_path / "dev"; dev.mkdir()
    prov.fetch(DatasetRef("vo-kitti-dev-00", "s"), dev)
    assert len(list(dev.glob("left_*.png"))) == 3              # stride 2, capped at 3
