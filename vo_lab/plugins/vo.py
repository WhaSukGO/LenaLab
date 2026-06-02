"""Visual-Odometry domain plugin (the SOLVER's domain seam).

Mirrors ver2's CIFAR plugin, but for monocular VO. The harness owns the held-out split
and the grader; the solver only ever sees the visible image frames.

MVP data policy (real-first + synthetic fallback, per the design): this increment ships
the SYNTHETIC generator only — deterministic, tiny, offline (no 100GB download), so the
whole spine + calibration gate runs locally with no Docker/GPU/API. A real KITTI/TartanAir
provider plugs into the same `fetch(ref, dest)` seam later.

Anti-gaming boundary: the synthetic world (camera trajectory + 3-D points) is generated
from a HARNESS-OWNED seed. The visible dataset gets rendered FRAMES only; the held-out
dataset gets the ground-truth camera centers (gt.txt). The solver never receives GT."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lab.menu import Menu, ParamSpec, Recipe
from lab.models import (
    BudgetSpec, Criterion, DatasetRef, ExperimentContract, ExperimentRecord, FrameworkSpec,
    OracleRef, Status, Usage, VerifiedResult,
)

VO_CODE_DIR = str((Path(__file__).parent / "vo_ref").resolve())

# Image / camera config (kept small and fixed for a robust, non-flaky calibration).
_W, _H = 640, 480
_FX = _FY = 500.0
_N_FRAMES = 40
_N_POINTS = 700
_TILE = 7  # px; each landmark gets a distinct random texture tile -> distinct ORB descriptors
_NOISE_STD = 8.0  # per-frame Gaussian sensor noise (gray levels); makes ORB params matter

# Fixed oracle bar. Set with a WIDE margin (cf. ver2 CIFAR's 0.45 vs random-0.10): the
# honest ORB-VO lands well below this after Sim(3) alignment, while the degenerate
# (static) negative control scores ~the GT spread (several units) -> clean separation.
ATE_THRESHOLD = 0.40
_CRITERION = Criterion(metric="ate_rmse", op="<=", value=ATE_THRESHOLD, tolerance=0.0)

_HARNESS_SEED = 1234  # owned by the harness; the solver cannot read or influence it


# --- synthetic world (deterministic) ----------------------------------------------------

def _make_world(seed: int):
    """Return (K, poses, points, tiles). poses are world->cam (R,t); points Nx3 world."""
    rng = np.random.default_rng(seed)
    K = np.array([[_FX, 0, _W / 2], [0, _FY, _H / 2], [0, 0, 1]], dtype=np.float64)

    # 3-D landmarks spread in a volume in front of the camera path.
    pts = np.column_stack([
        rng.uniform(-9, 9, _N_POINTS),
        rng.uniform(-6, 6, _N_POINTS),
        rng.uniform(5, 32, _N_POINTS),
    ])
    # distinct random texture tile per landmark (stable across frames -> good matching)
    tiles = rng.integers(0, 256, size=(_N_POINTS, _TILE, _TILE), dtype=np.uint8)

    # CONSTANT-SPEED trajectory: forward + gentle lateral sinusoid (constant step length so
    # unit-scale monocular VO recovers the shape up to a single global scale).
    poses = []
    step = 0.30
    for i in range(_N_FRAMES):
        s = i * step
        cx = 2.0 * np.sin(s * 0.15)          # gentle lateral sway
        cz = s                                # forward
        cy = 0.0
        C = np.array([cx, cy, cz])           # camera center in world
        # look roughly forward (+z), with the small lateral motion -> identity-ish rotation
        yaw = 0.30 * np.cos(s * 0.15)        # small yaw to keep parallax
        R_wc = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                         [0, 1, 0],
                         [-np.sin(yaw), 0, np.cos(yaw)]])      # cam-to-world
        R = R_wc.T                            # world-to-cam
        t = -R @ C
        poses.append((R, t, C))
    return K, poses, pts, tiles


def _render_frame(K, R, t, pts, tiles) -> np.ndarray:
    img = np.full((_H, _W), 110, dtype=np.uint8)            # mid-gray background
    cam = (R @ pts.T).T + t                                 # Nx3 in camera frame
    half = _TILE // 2
    # paint far points first so nearer ones overwrite (rough z-ordering)
    order = np.argsort(-cam[:, 2])
    for idx in order:
        z = cam[idx, 2]
        if z <= 0.5:
            continue
        u = K[0, 0] * cam[idx, 0] / z + K[0, 2]
        v = K[1, 1] * cam[idx, 1] / z + K[1, 2]
        ui, vi = int(round(u)), int(round(v))
        if half <= ui < _W - half and half <= vi < _H - half:
            img[vi - half:vi + half + 1, ui - half:ui + half + 1] = tiles[idx]
    return img


# --- dataset provider (satisfies lab.dataset_cache.DatasetProvider) ----------------------

class VODatasetProvider:
    def __init__(self, *, seed: int = _HARNESS_SEED):
        self.seed = seed

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        K, poses, pts, tiles = _make_world(self.seed)
        dest = Path(dest)
        if ref.held_out:
            # evaluator-only: ground-truth camera centers (the secret the solver must not see)
            gt = np.array([C for (_, _, C) in poses])
            np.savetxt(dest / "gt.txt", gt, fmt="%.6f")
        else:
            # solver-visible: rendered frames + intrinsics (no poses)
            np.savetxt(dest / "intrinsics.txt",
                       np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), fmt="%.6f")
            for i, (R, t, _C) in enumerate(poses):
                img = _render_frame(K, R, t, pts, tiles).astype(np.int16)
                # deterministic per-frame sensor noise: makes feature quantity/quality (and
                # thus the ORB params the committee tunes) genuinely matter for accuracy.
                noise = np.random.default_rng(self.seed + 1000 + i).normal(0, _NOISE_STD, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                cv2.imwrite(str(dest / f"frame_{i:04d}.png"), img)


class VOMetricExtractor:
    def extract(self, artifacts_dir: str) -> dict:
        import json
        p = Path(artifacts_dir) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}


# --- contracts ---------------------------------------------------------------------------

def _datasets() -> list[DatasetRef]:
    return [
        DatasetRef(name="vo-synth-frames", source="synthetic://vo"),
        DatasetRef(name="vo-synth-gt", source="synthetic://vo", held_out=True),
    ]


def _contract(*, degenerate: bool, threshold: float = ATE_THRESHOLD,
              datasets: list[DatasetRef] | None = None, framework=None,
              source: str = "orb-mono-vo-synthetic-calibration") -> ExperimentContract:
    cmd = ('VO_DEGENERATE=1 python3 "$LAB_CODE/run.py"' if degenerate
           else 'python3 "$LAB_CODE/run.py"')
    crit = Criterion(metric="ate_rmse", op="<=", value=threshold, tolerance=0.0)
    return ExperimentContract(
        success_definition=f"held-out monocular VO ATE-RMSE (sim3-aligned) <= {threshold}",
        gradable_criteria=[crit],
        framework=framework,                  # None -> CPU/local (no image); _CPU_FW -> docker
        datasets=datasets if datasets is not None else _datasets(),
        command=cmd,
        eval_command='python3 "$LAB_CODE/eval.py"',
        code_dir=VO_CODE_DIR,
        budget=BudgetSpec(max_tokens=50_000, max_wall_s=600, max_retries=1),
        oracle=OracleRef(criterion=crit, source=source),
        seed=0,
    )


# --- Track A: vetted menu recipe (committee may only select + clamp these params) --------

# Equivalent bar to ATE_THRESHOLD, expressed higher-is-better for ver2's Menu/loop.
VO_SCORE_BAR = 1.0 / (1.0 + ATE_THRESHOLD)   # ATE 0.40 <-> vo_score 0.714
# Local mode (Track A) never resolves this to an image (loop swallows NoImageError). Docker
# mode (Track B live) resolves it via images/registry.yaml — a key containing "cpu". version
# is "" so resolve() skips the version check and matches the cpu-opencv row.
_CPU_FW = FrameworkSpec(name="cpu", version="", cuda="")


def vo_recipe() -> Recipe:
    """Classical ORB monocular VO with tunable params. The command template and the oracle
    bar are FIXED; the committee may only set nfeatures / ransac_thresh within range."""
    return Recipe(
        id="orb-mono-vo",
        description="Classical ORB monocular VO; held-out Sim(3)-aligned vo_score.",
        framework=_CPU_FW, code_dir=VO_CODE_DIR, datasets=_datasets(),
        train_template=('LAB_NFEATURES={nfeatures} LAB_RANSAC_THRESH={ransac_thresh} '
                        'LAB_SEED={seed} python3 "$LAB_CODE/run.py"'),
        eval_command='python3 "$LAB_CODE/eval.py"',
        metric="vo_score", threshold=VO_SCORE_BAR,
        params=[
            ParamSpec("nfeatures", "int", low=300, high=3000, default=600),
            ParamSpec("ransac_thresh", "float", low=0.3, high=3.0, default=1.0),
        ],
        max_wall_s=600.0,
    )


def vo_menu() -> Menu:
    return Menu([vo_recipe()])


def vo_calibration_records(*, threshold: float = ATE_THRESHOLD,
                           datasets: list[DatasetRef] | None = None,
                           framework=None) -> tuple[ExperimentRecord, ExperimentRecord]:
    """Reproduction-first gate: the honest ORB-VO must VERIFY (low ATE); the degenerate
    static-trajectory control must be REJECTED (large ATE) — proving the grader is not a
    rubber stamp. Pre-contracted, so the planner is skipped (fixed reference scripts).
    threshold/datasets/framework are parameterized so the SAME gate runs on real data
    (TUM) with a baseline-derived bar and the docker image."""
    pos = ExperimentRecord(id="cal-pos", hypothesis="ORB monocular VO (positive control)",
                           status=Status.PROPOSED, priority=100,
                           contract=_contract(degenerate=False, threshold=threshold,
                                              datasets=datasets, framework=framework))
    neg = ExperimentRecord(id="cal-neg", hypothesis="degenerate static VO (negative control)",
                           status=Status.PROPOSED, priority=100,
                           contract=_contract(degenerate=True, threshold=threshold,
                                              datasets=datasets, framework=framework))
    return pos, neg
