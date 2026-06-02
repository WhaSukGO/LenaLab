"""Real-data provider parsing/association — verified OFFLINE on a tiny fabricated TUM-format
fixture (no 0.5 GB download). The real download just swaps in real files of the same format;
this proves the provider emits the lab's on-disk contract (frame_%04d.png + intrinsics.txt
visible; index-aligned gt.txt held-out) with timestamps correctly associated."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vo_lab  # noqa: E402,F401  -- bootstraps the ver2 (lab) path first
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from lab.models import DatasetRef  # noqa: E402
from vo_lab.plugins.vo_real import TUMDatasetProvider  # noqa: E402


def _make_fixture(root: Path, n: int = 5):
    seq = root / "rgbd_dataset_freiburg1_xyz"
    (seq / "rgb").mkdir(parents=True)
    ts = [1305031910.0 + 0.1 * i for i in range(n)]
    rgb_lines = ["# color images", "# timestamp filename"]
    for i, t in enumerate(ts):
        fn = f"rgb/{t:.6f}.png"
        cv2.imwrite(str(seq / fn), np.full((16, 16), i * 10, dtype=np.uint8))
        rgb_lines.append(f"{t:.6f} {fn}")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines))
    # GT timestamps offset by 5ms (within assoc window); positions = (i, 2i, 3i)*0.1
    gt_lines = ["# ground truth", "# timestamp tx ty tz qx qy qz qw"]
    for i, t in enumerate(ts):
        gt_lines.append(f"{t + 0.005:.6f} {i*0.1:.3f} {i*0.2:.3f} {i*0.3:.3f} 0 0 0 1")
    (seq / "groundtruth.txt").write_text("\n".join(gt_lines))


def test_tum_provider_emits_contract_with_correct_association(tmp_path):
    _make_fixture(tmp_path / "raw", n=5)
    prov = TUMDatasetProvider(raw_root=tmp_path / "raw", max_frames=None, stride=1,
                              assoc_max_dt=0.02)
    frames = tmp_path / "frames"; gt = tmp_path / "gt"
    frames.mkdir(); gt.mkdir()

    prov.fetch(DatasetRef("vo-tum-frames", "s"), frames)
    prov.fetch(DatasetRef("vo-tum-gt", "s", held_out=True), gt)

    # visible split: index-named grayscale frames + intrinsics, NO ground truth
    assert sorted(p.name for p in frames.glob("frame_*.png")) == [f"frame_{i:04d}.png" for i in range(5)]
    assert (frames / "intrinsics.txt").exists()
    assert not (frames / "gt.txt").exists()

    # held-out split: index-aligned positions, correctly associated by timestamp
    poses = np.loadtxt(gt / "gt.txt")
    assert poses.shape == (5, 3)
    assert np.allclose(poses[2], [0.2, 0.4, 0.6])      # i=2 -> (0.2, 0.4, 0.6)


def test_tum_provider_respects_max_frames(tmp_path):
    _make_fixture(tmp_path / "raw", n=5)
    prov = TUMDatasetProvider(raw_root=tmp_path / "raw", max_frames=3, stride=1)
    gt = tmp_path / "gt"; gt.mkdir()
    prov.fetch(DatasetRef("vo-tum-gt", "s", held_out=True), gt)
    assert np.loadtxt(gt / "gt.txt").shape == (3, 3)
