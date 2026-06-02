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
