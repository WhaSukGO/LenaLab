"""Synthetic-stereo data provider — the CONTAMINATION-CONTROLLED domain.

Procedurally generated stereo sequences with exact ground truth (vo_ref/synthetic_stereo.py),
which no model can have memorized. Same on-disk contract as the KITTI provider so the harness +
grader (eval_kitti.py) are reused unchanged:
  dev (held_out=False):      left_%06d.png · right_%06d.png · intrinsics.txt(fx fy cx cy baseline)
  held-out (held_out=True):  seq_<name>/{input/{left_*.png,right_*.png,intrinsics.txt}, gt.txt, gt_poses.txt}

The reference VO scores ~1.7% t_err here (positive control passed), so it is a sound, recoverable,
exactly-graded stereo-VO problem on data outside any training set.
"""
from __future__ import annotations

from pathlib import Path

from lab.models import DatasetRef
from lab.util import ensure_dir

from .vo_ref.synthetic_stereo import generate_sequence

# (name, trajectory-kind, n_frames, seed). dev and held-out are disjoint (different seeds + kinds).
DEFAULT_DEV = ("synthdev", "A", 280, 111)
DEFAULT_HELDOUT = (("synth1", "B", 260, 222), ("synth2", "C", 300, 333))

# VIO: same idea but each sequence also has an IMU stream (imu.txt) + vision blackouts.
DEFAULT_VIO_DEV = ("viodev", "A", 280, 411)
DEFAULT_VIO_HELDOUT = (("vio1", "B", 300, 522), ("vio2", "C", 320, 633))


def synth_datasets(dev=DEFAULT_DEV, heldout=DEFAULT_HELDOUT) -> list[DatasetRef]:
    return [
        DatasetRef(name=f"vo-synth-dev-{dev[0]}", source=f"synth:{dev[0]}"),
        DatasetRef(name="vo-synth-heldout-" + "_".join(s[0] for s in heldout),
                   source=";".join(f"synth:{s[0]}" for s in heldout), held_out=True),
    ]


class SyntheticStereoProvider:
    def __init__(self, *, dev=DEFAULT_DEV, heldout=DEFAULT_HELDOUT):
        self.dev = dev
        self.heldout = heldout

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            for name, kind, n, seed in self.heldout:
                sub = ensure_dir(dest / f"seq_{name}")
                generate_sequence(sub / "input", gt_dir=sub, kind=kind, n=n, seed=seed)
        else:
            name, kind, n, seed = self.dev
            generate_sequence(dest, gt_dir=None, kind=kind, n=n, seed=seed)  # dev: frames only, no GT


# LEARNED: train on several synthetic seqs (with GT poses), test on disjoint UNSEEN synthetic seqs.
# Contamination-clean: a network cannot have memorised our procedural synthetic world.
DEFAULT_LEARN_TRAIN = (("ltrA", "A", 240, 711), ("ltrB", "B", 240, 722),
                       ("ltrC", "C", 240, 733), ("ltrD", "A", 240, 744))
DEFAULT_LEARN_TEST = (("lte1", "B", 260, 822), ("lte2", "C", 260, 833))


def synth_learn_datasets(train=DEFAULT_LEARN_TRAIN, test=DEFAULT_LEARN_TEST) -> list[DatasetRef]:
    return [
        DatasetRef(name="vo-synthlearn-train", source="synthlearn:train"),
        DatasetRef(name="vo-synthlearn-test-" + "_".join(s[0] for s in test),
                   source=";".join(f"synthlearn:{s[0]}" for s in test), held_out=True),
    ]


class SyntheticLearnedProvider:
    """Learned-VO on the contamination-clean synthetic domain. Lays out the same train/test_input/gt
    structure the learned task + eval_learned expect, but from procedural synthetic sequences:
      dev (held_out=False): train/seq_<name>/{left frames, poses.txt, intrinsics}  (supervision)
                            test_input/seq_<name>/{left frames, intrinsics}          (NO labels)
      held-out:             seq_<name>/gt.txt (camera centres of the test seqs — grader-only secret)
    """

    def __init__(self, *, train=DEFAULT_LEARN_TRAIN, test=DEFAULT_LEARN_TEST):
        self.train = train
        self.test = test

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        from .vo_ref.synthetic_stereo import generate_sequence, make_trajectory
        import numpy as np
        dest = Path(dest)
        if ref.held_out:
            for name, kind, n, seed in self.test:                 # grader secret: just the centres
                centres = np.array([T[:3, 3] for T in make_trajectory(kind, n)])
                sub = ensure_dir(dest / f"seq_{name}")
                np.savetxt(sub / "gt.txt", centres, fmt="%.6f")
        else:
            for name, kind, n, seed in self.train:                # train: frames + poses.txt (3x4)
                sub = ensure_dir(dest / "train" / f"seq_{name}")
                generate_sequence(sub, gt_dir=sub, kind=kind, n=n, seed=seed)
                (sub / "gt_poses.txt").rename(sub / "poses.txt")  # learned trainer reads poses.txt
                (sub / "gt.txt").unlink(missing_ok=True)
            for name, kind, n, seed in self.test:                 # test inputs: frames only, no labels
                generate_sequence(dest / "test_input" / f"seq_{name}", gt_dir=None,
                                  kind=kind, n=n, seed=seed)


def vio_datasets(dev=DEFAULT_VIO_DEV, heldout=DEFAULT_VIO_HELDOUT) -> list[DatasetRef]:
    return [
        DatasetRef(name=f"vo-vio-dev-{dev[0]}", source=f"vio:{dev[0]}"),
        DatasetRef(name="vo-vio-heldout-" + "_".join(s[0] for s in heldout),
                   source=";".join(f"vio:{s[0]}" for s in heldout), held_out=True),
    ]


class SyntheticVIOProvider:
    """Like SyntheticStereoProvider but each sequence also carries an IMU stream (input/imu.txt) and
    vision blackouts — for the M3 visual-inertial fusion experiment. GT stays isolated; imu.txt is a
    provided sensor (survives gt* staging)."""

    def __init__(self, *, dev=DEFAULT_VIO_DEV, heldout=DEFAULT_VIO_HELDOUT):
        self.dev = dev
        self.heldout = heldout

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        from .vo_ref.synthetic_vio import generate_vio_sequence
        dest = Path(dest)
        if ref.held_out:
            for name, kind, n, seed in self.heldout:
                sub = ensure_dir(dest / f"seq_{name}")
                generate_vio_sequence(sub / "input", gt_dir=sub, kind=kind, n=n, seed=seed)
        else:
            name, kind, n, seed = self.dev
            generate_vio_sequence(dest, gt_dir=None, kind=kind, n=n, seed=seed)  # dev: no GT
