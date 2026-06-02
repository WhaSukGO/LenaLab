"""Track B — the VO Implementer: the solver AUTHORS a visual-odometry algorithm, then the
unchanged independent evaluator grades it on the held-out split against a fixed oracle.

This is where "experts implement algorithms" stops being parameter-tuning. The agent writes
only the implementation (main.py); the harness owns the grader (eval.py) and the oracle
(ATE-RMSE <= bar, from the TASK), so the implementer cannot grade or game its own work.
Nothing is accepted until the independent ScriptEvaluator measures the authored code on the
held-out GT — which the agent never sees.

Live authoring runs in a sandbox (container-only, no host shell, no network, writes confined
to the code dir, eval.py off-limits) via ver2's sdk_author. Offline tests inject a fake
author_fn that writes a known-good (or deliberately bad) main.py — the authored code really
runs; only the LLM that would write it is faked."""
from __future__ import annotations

from pathlib import Path

from lab.agents.implementer import ImplementationTask
from lab.models import Usage

from ..plugins.vo import ATE_THRESHOLD, VO_CODE_DIR, _CPU_FW, _datasets

_EVAL_CODE = (Path(VO_CODE_DIR) / "eval.py").read_text()   # single source of truth (harness-owned)
_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run.py").read_text()

VO_TASK_DESCRIPTION = (
    "Implement a MONOCULAR visual-odometry algorithm. Read a grayscale image sequence "
    "(frame_0000.png ... ) and camera intrinsics (intrinsics.txt: fx fy cx cy) from "
    "$LAB_DATA, estimate the per-frame camera trajectory, and write it to "
    "$LAB_ARTIFACTS/traj.txt as one `tx ty tz` per frame (camera centers, any consistent "
    "world frame — absolute scale is unobservable and will be Sim(3)-aligned when scored). "
    "Classic pipeline: feature detection + matching across frames, essential-matrix + "
    "recoverPose, accumulate poses. You are scored by held-out ATE-RMSE you cannot see; do "
    "not attempt to read ground truth — there is none in $LAB_DATA.")


def vo_impl_task() -> ImplementationTask:
    """What to implement + how it is independently judged (oracle fixed here, not by the agent)."""
    return ImplementationTask(
        description=VO_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_EVAL_CODE,                 # harness-owned grader; restored before judging
        metric="ate_rmse", op="<=", threshold=ATE_THRESHOLD,
        datasets=_datasets(),
        entry_filename="main.py",
    )


def vo_impl_task_real(threshold: float, *, datasets=None) -> ImplementationTask:
    """Track B on REAL data (TUM RGB-D). Same task + harness-owned grader; only the data
    and the oracle bar change. The bar is derived from the reference VO's held-out ATE on
    the same sequence (see run_vo_tum_calibration), i.e. 'roughly match/beat the classical
    baseline' — the right oracle when no absolute published number fits this simple VO."""
    from ..plugins.vo_real import tum_datasets

    return ImplementationTask(
        description=VO_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_EVAL_CODE,
        metric="ate_rmse", op="<=", threshold=threshold,
        datasets=datasets or tum_datasets(),
        entry_filename="main.py",
    )


def reference_author():
    """A non-LLM 'author' that writes the reference ORB-VO as main.py. Used as a known-good
    baseline and in offline tests — proves the verification handoff without an API key."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_REFERENCE_MAIN)
        return Usage(tokens_in=0, tokens_out=0)
    return author


# --- RGB-D + generalization experiment ---------------------------------------------------

_RGBD_EVAL_CODE = (Path(VO_CODE_DIR) / "eval_rgbd.py").read_text()      # harness-owned grader
_RGBD_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_rgbd.py").read_text()  # known-good baseline

RGBD_TASK_DESCRIPTION = (
    "Implement an RGB-D visual-odometry algorithm. From $LAB_DATA read grayscale frames "
    "(frame_%04d.png), aligned 16-bit depth (depth_%04d.png), and intrinsics.txt "
    "(fx fy cx cy depth_scale; metric depth in metres = depth_png / depth_scale). USE THE "
    "DEPTH to recover ABSOLUTE (metric) scale — back-project features to 3-D and estimate "
    "pose (e.g. 3D-2D PnP), so the trajectory is metric (no scale ambiguity). Write "
    "$LAB_ARTIFACTS/traj.txt with one `tx ty tz` (camera centre) per frame, in order. You "
    "will be graded on MULTIPLE held-out sequences you never see, with SE(3) (metric) "
    "alignment — so it must generalize and get scale right. Do not read any ground truth.")


def vo_impl_task_rgbd(threshold: float, *, dev: str = "fr1_xyz",
                      heldout: tuple[str, ...] = ("fr1_desk",)):
    """RGB-D Track B task: graded by the generalization grader (runs the authored code on
    unseen sequences, SE(3)-metric ATE/RPE). The bar is derived from the reference RGB-D VO."""
    from ..plugins.vo_rgbd import rgbd_datasets

    return ImplementationTask(
        description=RGBD_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',   # ScriptEvaluator restores eval.py = grader
        eval_code=_RGBD_EVAL_CODE,
        metric="ate_rmse", op="<=", threshold=threshold,
        datasets=rgbd_datasets(dev, heldout),
        entry_filename="main.py",
    )


def rgbd_reference_author():
    """Writes the reference RGB-D VO as main.py (baseline + offline pipeline proof, no API)."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_RGBD_REFERENCE_MAIN)
        return Usage()
    return author


def degenerate_author():
    """Writes a main.py that emits a static (origin) trajectory — the negative control."""
    src = ("import os, glob, numpy as np\n"
           "d=os.environ['LAB_DATA']; a=os.environ['LAB_ARTIFACTS']; os.makedirs(a,exist_ok=True)\n"
           "n=len(glob.glob(os.path.join(d,'frame_*.png')))\n"
           "np.savetxt(os.path.join(a,'traj.txt'), np.zeros((n,3)), fmt='%.6f')\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
        return Usage()
    return author


# --- SLAM with loop closure -------------------------------------------------------------

_SLAM_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_slam.py").read_text()

SLAM_TASK_DESCRIPTION = (
    "Implement RGB-D SLAM **with LOOP CLOSURE** for a long sequence that revisits places. "
    "From $LAB_DATA read frames (frame_%04d.png), depth (depth_%04d.png), intrinsics.txt "
    "(fx fy cx cy depth_scale). Frame-to-frame RGB-D odometry alone DRIFTS badly on this "
    "sequence — you must add global consistency: (1) select keyframes, (2) DETECT LOOP "
    "CLOSURES (recognize when a keyframe revisits an earlier place, e.g. by matching its "
    "features to temporally-distant keyframes + geometric verification), (3) optimize the "
    "POSE GRAPH (sequential + loop constraints) for a globally-consistent trajectory. Write "
    "$LAB_ARTIFACTS/traj.txt with one `tx ty tz` (camera centre) per frame, in order. You "
    "are graded on held-out ATE (SE(3) metric) — VO-only will NOT pass the bar, so loop "
    "closure is required. Do not read any ground truth. (numpy, opencv, scipy available.)")


def vo_impl_task_slam(threshold: float, *, dev: str = "fr1_room",
                      heldout: tuple[str, ...] = ("fr1_room",)):
    """SLAM Track B task. Same generalization grader (runs the authored code on the held-out
    loop sequence, SE(3) ATE), but the bar requires loop closure (VO-only drifts past it)."""
    from ..plugins.vo_rgbd import rgbd_datasets

    return ImplementationTask(
        description=SLAM_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_RGBD_EVAL_CODE,
        metric="ate_rmse", op="<=", threshold=threshold,
        datasets=rgbd_datasets(dev, heldout),
        entry_filename="main.py",
    )


def slam_reference_author():
    """Writes the reference RGB-D SLAM (loop closure + pose graph) as main.py."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_SLAM_REFERENCE_MAIN)
        return Usage()
    return author


def resilient_sdk_author(job_runner, image_registry, dataset_cache, *,
                         model: str = "claude-sonnet-4-6", max_turns: int = 80):
    """A live sandboxed author that does NOT discard work when the session ends early.

    The SDK raises if the agent hits its turn limit (or otherwise errors). The original
    behavior propagated that and the whole experiment was marked FAILED — throwing away a
    perfectly gradable main.py the agent had already written. Here, if the entry file exists,
    we let the independent evaluator grade it anyway (a turn limit is not an algorithm
    failure). Token usage may under-report on this path since the SDK error loses it."""
    from lab.agents.implementer import sdk_author

    inner = sdk_author(job_runner, image_registry, dataset_cache, model=model,
                       max_turns=max_turns)

    def author(task, code_dir: Path, rec) -> Usage:
        try:
            return inner(task, code_dir, rec)
        except Exception as e:  # noqa: BLE001 - any SDK/session error
            entry = Path(code_dir) / task.entry_filename
            if entry.exists() and entry.stat().st_size > 0:
                print(f"NOTE: authoring session ended early ({e}); grading the artifact "
                      f"it left ({entry.name}).")
                return Usage()
            raise

    return author
