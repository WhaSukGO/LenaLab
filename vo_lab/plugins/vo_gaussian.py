"""3DGS FIDELITY-LADDER domain — the middle rung between procedural-synth and real.

The sim-to-real experiment compares three TRAINING domains, all tested on the SAME held-out REAL
KITTI 07 + 09 and graded by the SAME eval_learned Sim3-ATE grader. Only the TRAIN side differs:

  procedural synth -> real   (SimToRealProvider; appearance gap is huge: ate_rmse 69.57)
  3DGS-rendered  -> real      (THIS provider; photoreal renders should narrow the gap)
  real           -> real      (KITTILearnedProvider; in-domain ceiling: ate_rmse 27.24)

The hypothesis is a fidelity ladder: as the TRAIN imagery gets photometrically closer to KITTI
(procedural -> 3DGS renders -> real KITTI), sim-to-real transfer should improve monotonically. This
provider is the 3DGS rung. It mirrors SimToRealProvider exactly — synthetic-looking TRAIN frames +
REAL KITTI test_input + REAL KITTI held-out gt centres — but swaps the TRAIN imagery source for a
3D-Gaussian-Splatting render seam.

PHASE-0 STUB: the render seam (`_render_train`) currently FALLS BACK to the procedural synthetic
generator, identical to SimToRealProvider's train side, so the harness/grader integrate now with no
GPU. The seam is isolated and prominently marked TODO(phase1) so a real GSplatModule.render(...) drops
in later WITHOUT touching fetch(), the registry, or the rest of the pipeline.

Construction is import-clean: building any provider here triggers NO rendering and NO GPU.
"""
from __future__ import annotations

from pathlib import Path

from lab.models import DatasetRef
from lab.util import ensure_dir

from .vo_kitti_learned import KITTILearnedProvider

# Real KITTI test sequences (shared across all three ladder domains).
KITTI_TEST = ("07", "09")
# Train scenes whose 3DGS reconstruction we render from (Phase-1); KITTI seq ids reused for naming.
TRAIN_SCENES = ("00", "02", "06", "08")

# --- procedural fallback config (Phase-0 stub; matches SimToRealProvider's synth train side) ------
# (name, trajectory-kind, n_frames, seed) per train scene — disjoint kinds/seeds, contamination-clean.
_FALLBACK_TRAIN = (("00", "A", 240, 711), ("02", "B", 240, 722),
                   ("06", "C", 240, 733), ("08", "A", 240, 744))


class GaussianProvider:
    """3DGS-rendered TRAIN -> REAL KITTI test. Same on-disk contract as SimToRealProvider /
    SyntheticLearnedProvider so the learned harness + eval_learned grader are reused unchanged:
      dev (held_out=False): train/seq_<s>/{left frames, poses.txt(3x4), intrinsics}  (supervision)
                            test_input/seq_<s>/{left frames, intrinsics}  = REAL KITTI (no labels)
      held-out:             seq_<s>/gt.txt  = REAL KITTI camera centres (grader-only secret)
    """

    def __init__(self, *, kitti_test=KITTI_TEST, train_scenes=TRAIN_SCENES,
                 frames_per_scene=200, win_start=100, stride=2, src_radius=2, perturbations=()):
        self.kitti_test = tuple(kitti_test)
        self.train_scenes = tuple(train_scenes)
        self.frames_per_scene = frames_per_scene
        self.win_start = win_start
        self.stride = stride
        self.src_radius = src_radius
        # Optional world-frame viewpoint offsets (metres). Each renders an extra PARALLEL-PATH training
        # sequence per scene — the novel-viewpoint augmentation real data cannot provide. () = GT only.
        self.perturbations = tuple(perturbations)
        # Real KITTI supplies BOTH the test_input frames and the held-out gt centres (no KITTI train).
        self.kitti = KITTILearnedProvider(train=(), test=self.kitti_test,
                                          test_max=300, test_stride=3)

    # --- THE 3DGS RENDER SEAM (Phase-1: wired to GSplatModule) -------------------------------------
    def _render_train(self, dest: Path) -> None:
        """Write train/seq_<s>/{left frames, poses.txt(3x4), intrinsics} by RENDERING real KITTI scenes.

        Phase-1 wires the real renderer: for each train scene, GSplatModule reconstructs a colour point
        cloud from real KITTI stereo and renders a left-camera trajectory of REAL-appearance novel views
        with EXACT GT poses (stereo-depth reprojection / point-splat — the achievable precursor to
        optimised 3DGS; see vo_ref/gaussian_kitti.py). TODO(phase2): swap GSplatModule internals for
        optimised gsplat — fetch/registry/grader stay unchanged. Falls back to procedural only if KITTI
        is unavailable, so the harness still integrates in a bare environment.
        """
        from .vo_kitti import DEFAULT_CACHE
        import numpy as np
        dest = Path(dest)
        root = Path(DEFAULT_CACHE) / "dataset"
        try:
            from .vo_ref.gaussian_kitti import GSplatModule
            gm = GSplatModule(step=2, src_radius=self.src_radius, splat=2)
            for scene in self.train_scenes:
                poses = np.loadtxt(root / "poses" / f"{scene}.txt").reshape(-1, 3, 4)
                n = poses.shape[0]
                stop = min(self.win_start + self.frames_per_scene * self.stride, n - self.src_radius)
                idx = list(range(self.win_start, stop, self.stride))
                seq_dir = root / "sequences" / scene
                gm.render_sequence(seq_dir, poses, idx, dest / "train" / f"seq_{scene}")     # GT path
                for k, off in enumerate(self.perturbations):                                  # parallel paths
                    gm.render_sequence(seq_dir, poses, idx, dest / "train" / f"seq_{scene}_p{k}",
                                       world_offset=off)
        except Exception as e:  # pragma: no cover - graceful fallback if KITTI/stereo missing
            print(f"[GaussianProvider] render unavailable ({e}); falling back to procedural train")
            from .vo_ref.synthetic_stereo import generate_sequence
            for scene, (kind, seed) in zip(self.train_scenes, [("A", 711), ("B", 722), ("C", 733), ("A", 744)]):
                sub = ensure_dir(dest / "train" / f"seq_{scene}")
                generate_sequence(sub, gt_dir=sub, kind=kind, n=240, seed=seed)
                (sub / "gt_poses.txt").rename(sub / "poses.txt")
                (sub / "gt.txt").unlink(missing_ok=True)

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        dest = Path(dest)
        if ref.held_out:
            self.kitti.fetch(ref, dest)        # REAL KITTI gt centres -> seq_<s>/gt.txt (grader secret)
        else:
            self._render_train(dest)           # 3DGS-rendered (Phase-0: procedural) train supervision
            self.kitti.fetch(ref, dest)        # REAL KITTI frames -> test_input/seq_<s>/ (no labels)


# --- ladder registry ------------------------------------------------------------------------------
# Prefer importing SimToRealProvider from the existing script; replicate minimally if unimportable.
try:
    from scripts.sim_to_real_kitti import SimToRealProvider
except Exception:  # pragma: no cover - replicate the procedural-synth -> real provider minimally
    from .vo_synth import SyntheticLearnedProvider
    from .vo_ref.synthetic_stereo import generate_sequence

    class SimToRealProvider:
        """Procedural-synth TRAIN -> REAL KITTI test (minimal replica of scripts/sim_to_real_kitti.py)."""

        def __init__(self):
            self.synth = SyntheticLearnedProvider()
            self.kitti = KITTILearnedProvider(train=(), test=KITTI_TEST, test_max=300, test_stride=3)

        def fetch(self, ref: DatasetRef, dest: Path) -> None:
            dest = Path(dest)
            if ref.held_out:
                self.kitti.fetch(ref, dest)
            else:
                for name, kind, n, seed in self.synth.train:
                    sub = ensure_dir(dest / "train" / f"seq_{name}")
                    generate_sequence(sub, gt_dir=sub, kind=kind, n=n, seed=seed)
                    (sub / "gt_poses.txt").rename(sub / "poses.txt")
                    (sub / "gt.txt").unlink(missing_ok=True)
                self.kitti.fetch(ref, dest)


DOMAINS = ["procedural", "gaussian", "real"]


def make_domain_provider(domain: str):
    """Fidelity-ladder provider registry: domain name -> provider. All test on REAL KITTI 07 + 09;
    only the TRAIN domain differs. Construction triggers no rendering / no GPU."""
    if domain == "procedural":
        return SimToRealProvider()
    if domain == "gaussian":
        return GaussianProvider()
    if domain == "real":
        return KITTILearnedProvider(train=("00", "02", "06", "08"), test=("07", "09"),
                                    train_max=250, train_stride=2, test_max=300, test_stride=3)
    raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
