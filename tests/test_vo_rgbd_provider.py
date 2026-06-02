"""RGB-D + generalization provider — verified OFFLINE on a fabricated TUM-RGBD fixture.
Checks: depth is exposed, intrinsics carry the depth scale, held-out is laid out as
seq_*/input (frames+depth, NO gt) + seq_*/gt.txt (grader-only) so the solver can't read GT."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vo_lab  # noqa: E402,F401
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from lab.models import DatasetRef  # noqa: E402
from vo_lab.plugins.vo_rgbd import TUMRGBDProvider  # noqa: E402


def _fixture(raw: Path, n: int = 5):
    seq = raw / "rgbd_dataset_freiburg1_xyz"
    (seq / "rgb").mkdir(parents=True); (seq / "depth").mkdir()
    ts = [1305031910.0 + 0.1 * i for i in range(n)]
    rgb_l, dep_l, gt_l = ["# rgb"], ["# depth"], ["# gt"]
    for i, t in enumerate(ts):
        cv2.imwrite(str(seq / f"rgb/{t:.6f}.png"), np.full((16, 16), i * 10, np.uint8))
        cv2.imwrite(str(seq / f"depth/{t:.6f}.png"), np.full((16, 16), 5000, np.uint16))  # 1.0 m
        rgb_l.append(f"{t:.6f} rgb/{t:.6f}.png")
        dep_l.append(f"{t + 0.001:.6f} depth/{t:.6f}.png")
        gt_l.append(f"{t + 0.002:.6f} {i*0.1:.3f} {i*0.2:.3f} {i*0.3:.3f} 0 0 0 1")
    (seq / "rgb.txt").write_text("\n".join(rgb_l))
    (seq / "depth.txt").write_text("\n".join(dep_l))
    (seq / "groundtruth.txt").write_text("\n".join(gt_l))


def test_rgbd_provider_exposes_depth_and_isolates_gt(tmp_path):
    _fixture(tmp_path / "raw")
    prov = TUMRGBDProvider(dev="fr1_xyz", heldout=("fr1_xyz",), raw_root=tmp_path / "raw",
                           max_frames=None, assoc_max_dt=0.02)

    dev = tmp_path / "dev"; dev.mkdir()
    prov.fetch(DatasetRef("vo-rgbd-dev:fr1_xyz", "s"), dev)
    assert len(list(dev.glob("frame_*.png"))) == 5
    assert len(list(dev.glob("depth_*.png"))) == 5            # depth is exposed
    intr = np.loadtxt(dev / "intrinsics.txt")
    assert intr.shape == (5,) and intr[4] == 5000.0           # depth scale carried
    assert not (dev / "gt.txt").exists()                      # dev has no GT
    # depth round-trips as 16-bit metric
    d = cv2.imread(str(dev / "depth_0000.png"), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and d[0, 0] == 5000

    ho = tmp_path / "ho"; ho.mkdir()
    prov.fetch(DatasetRef("vo-rgbd-heldout:fr1_xyz", "s", held_out=True), ho)
    sub = ho / "seq_fr1_xyz"
    assert (sub / "input" / "frame_0000.png").exists()
    assert (sub / "input" / "depth_0000.png").exists()
    assert not (sub / "input" / "gt.txt").exists()            # GT NOT in the solver's input
    gt = np.loadtxt(sub / "gt.txt")                            # GT is grader-only, alongside
    assert gt.shape == (5, 3) and np.allclose(gt[2], [0.2, 0.4, 0.6])
