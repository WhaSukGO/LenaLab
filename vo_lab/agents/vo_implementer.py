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
_INFER_CODE = (Path(VO_CODE_DIR) / "infer_heldout.py").read_text()      # GT-free held-out runner
_INFER_CMD = 'python3 "$LAB_CODE/infer.py"'
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
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,  # GT-free inference
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
                      heldout: tuple[str, ...] = ("fr2_desk",),
                      allow_in_sample: bool = False):
    """SLAM Track B task. Graded on a HELD-OUT loop sequence with SE(3) ATE; the bar requires
    loop closure (VO-only drifts past it).

    NOTE (2026-06-04): the original run used dev=held-out=fr1_room (train-on-test) — its
    0.185 m was in-sample, not generalization. The default held-out is now a DISJOINT sequence
    and same-sequence dev/held-out is refused unless explicitly allowed (for legacy repro)."""
    if not allow_in_sample and dev in heldout:
        raise ValueError(
            f"SLAM train-on-test: dev={dev!r} is also in held-out={heldout!r}. Use a disjoint "
            "held-out (e.g. heldout=('fr2_desk',)) or pass allow_in_sample=True to reproduce "
            "the original in-sample run.")
    from ..plugins.vo_rgbd import rgbd_datasets

    return ImplementationTask(
        description=SLAM_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_RGBD_EVAL_CODE,
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,  # GT-free inference
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


# --- KITTI stereo (cross-domain generalization: outdoor driving) ------------------------

_KITTI_STEREO_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_kitti_stereo.py").read_text()

KITTI_TASK_DESCRIPTION = (
    "Implement a STEREO visual-odometry algorithm for outdoor driving (KITTI). From $LAB_DATA "
    "read rectified grayscale stereo pairs (left_%06d.png, right_%06d.png) and intrinsics.txt "
    "(fx fy cx cy baseline_m — the horizontal stereo baseline in metres). USE THE STEREO PAIR "
    "to recover ABSOLUTE (metric) scale: a disparity gives metric depth (Z = fx * baseline / "
    "disparity), so back-project features to 3-D and estimate pose (e.g. 3D-2D PnP) — the "
    "trajectory must be metric (no scale ambiguity). Motion is long forward driving with turns. "
    "Write $LAB_ARTIFACTS/traj.txt with one `tx ty tz` (camera centre) per frame, in order. ALSO "
    "write $LAB_ARTIFACTS/poses.txt with the full per-frame camera pose as a row-major 3x4 "
    "[R|t] cam->world matrix (12 numbers per line, KITTI format) — this lets the grader score the "
    "OFFICIAL KITTI metric (translational t_err % AND rotational r_err deg/m). You are graded on "
    "MULTIPLE held-out sequences you never see. Do not read any ground truth. "
    "TO REDUCE DRIFT below basic frame-to-frame VO: maintain a sliding WINDOW of keyframes and run "
    "LOCAL BUNDLE ADJUSTMENT — jointly refine the recent keyframe poses AND the 3D landmark "
    "positions to minimise reprojection error across the window (scipy.optimize.least_squares is "
    "available; use a sparse/windowed problem to stay fast). Frame-to-frame PnP alone will NOT "
    "clear the bar. (numpy, opencv, scipy available; cv2.StereoSGBM_create computes disparity.)")


_KITTI_EVAL_CODE = (Path(VO_CODE_DIR) / "eval_kitti.py").read_text()   # leaderboard-form t_err grader

# Published KITTI odometry translational-error anchors (t_err %, lower=better), for an EXTERNAL
# bar instead of a self-referential one. Basic frame-to-frame stereo VO ~2-3%; ORB-SLAM2 stereo
# (adds bundle adjustment + loop closure) ~1.15%; learned SOTA (DROID-SLAM) ~0.4%.
KITTI_SOTA_TERR = {"basic_stereo_vo": 3.0, "orbslam2_stereo": 1.15, "droid_slam": 0.40}


def vo_impl_task_kitti(threshold: float = 1.5, *, dev: str = "00",
                       heldout: tuple[str, ...] = ("05", "07")):
    """KITTI stereo Track B task, graded on the leaderboard-form KITTI metric: length-normalized
    translational drift t_err (%) over 100-800 m sub-sequences (eval_kitti.py), GT-isolated. The
    bar is EXTERNAL (published) — default 1.5% sits between basic stereo VO (~2-3%) and ORB-SLAM2
    (~1.15%), so clearing it means beating frame-to-frame VO and approaching classical SLAM."""
    from ..plugins.vo_kitti import kitti_datasets

    return ImplementationTask(
        description=KITTI_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_KITTI_EVAL_CODE,                   # leaderboard-form segment t_err grader
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,  # GT-free inference
        metric="t_err_pct", op="<=", threshold=threshold,
        datasets=kitti_datasets(dev, heldout),
        entry_filename="main.py",
        label_filename="gt*.txt",                     # strip gt.txt AND gt_poses.txt when staging
    )


# --- M2: full SLAM (loop closure + pose-graph) ------------------------------------------------
# The held-out sequences are LONGER and CONTAIN LOOPS (the vehicle returns near places it saw
# earlier). Local BA fixes local drift but not the GLOBAL drift accumulated over a loop — only
# loop closure does. Offline de-risk (scripts/m2_derisk_loopclosure.py) proved an ideal closure
# takes full-strided seq_07 from 2.41% -> 1.32% t_err, approaching ORB-SLAM2's 1.15%.
KITTI_SLAM_TASK_DESCRIPTION = (
    "Implement a STEREO visual-SLAM algorithm for outdoor driving (KITTI). From $LAB_DATA read "
    "rectified grayscale stereo pairs (left_%06d.png, right_%06d.png) and intrinsics.txt "
    "(fx fy cx cy baseline_m). USE THE STEREO PAIR for ABSOLUTE metric scale (Z = fx*baseline/"
    "disparity; cv2.StereoSGBM_create). Motion is long forward driving with turns. Write "
    "$LAB_ARTIFACTS/traj.txt (one `tx ty tz` camera centre per frame, in order) AND "
    "$LAB_ARTIFACTS/poses.txt (full per-frame pose as a row-major 3x4 [R|t] cam->world matrix, "
    "12 numbers/line, KITTI format) — the grader scores the OFFICIAL KITTI metric (t_err % AND "
    "r_err deg/m) on MULTIPLE held-out sequences you never see. Do not read any ground truth.\n"
    "These held-out sequences are LONG and CONTAIN LOOPS: the vehicle RE-VISITS places it saw "
    "earlier (it returns near its start). Frame-to-frame VO and even local bundle adjustment "
    "accumulate GLOBAL drift over such a loop that local methods CANNOT remove — clearing the bar "
    "REQUIRES LOOP CLOSURE. Build a keyframe SLAM:\n"
    "  (1) FRONT-END: stereo VO with a sliding-window LOCAL BUNDLE ADJUSTMENT (jointly refine "
    "recent keyframe poses + 3D landmarks by reprojection error). Make BA ROBUST — never let a BA "
    "update increase window reprojection error vs the PnP pose (revert if it does); a Huber loss "
    "and a conservative trust region help. (An unstable BA is worse than no BA.)\n"
    "  (2) KEYFRAMES: keep keyframes with their pose, ORB keypoints/descriptors, and 3D points.\n"
    "  (3) LOOP DETECTION (appearance, NOT position — your VO position has DRIFTED, so it can't be "
    "trusted to find revisits): for each new keyframe, match its ORB descriptors against PAST "
    "keyframes (a bag-of-visual-words signature or descriptor voting), then GEOMETRICALLY VERIFY a "
    "candidate (enough inlier matches under a fundamental/PnP RANSAC) before accepting it as a loop.\n"
    "  (4) LOOP CONSTRAINT: for a confirmed loop, estimate the relative pose between the current and "
    "matched keyframe (PnP or 3D-3D from the stereo points).\n"
    "  (5) GLOBAL POSE-GRAPH OPTIMISATION: optimise ALL keyframe poses to satisfy the sequential "
    "odometry edges AND the loop-closure edge(s) (e.g. Gauss-Newton / least_squares on SE(3); this "
    "distributes the accumulated drift around the loop), then propagate the corrected keyframe poses "
    "to every frame and rewrite traj.txt + poses.txt. Pose-graph optimisation runs once at the end "
    "over keyframes only, so it is cheap. (numpy, opencv, scipy available.)")


# GPU guidance — appended for the torch-accelerated variant. The CPU finding (M1): aggressive
# bundle adjustment on scipy was ~10s/frame and timed out the grader. A CUDA GPU removes that wall.
KITTI_SLAM_GPU_PARAGRAPH = (
    "\nA CUDA GPU IS AVAILABLE (torch 2.4, torch.cuda.is_available() is True, RTX 3080 16GB). Keep "
    "the classical front-end (cv2 ORB/StereoSGBM/PnP) on CPU, but implement the EXPENSIVE "
    "OPTIMISATION on the GPU with torch: (a) local BUNDLE ADJUSTMENT and (b) the global POSE-GRAPH "
    "optimisation. Represent keyframe poses (as se(3) tangent / quaternion+translation tensors) and "
    "landmarks as torch tensors on 'cuda'; build the reprojection and pose-graph residuals as "
    "differentiable torch ops and minimise on the GPU (torch autograd — Levenberg-Marquardt with "
    "torch.linalg.solve on the GPU, or LBFGS/Adam). Because the GPU is fast, you can afford a LARGER "
    "keyframe window, MORE iterations, and DENSER landmarks than CPU scipy allowed (CPU BA was "
    "~10s/frame and TIMED OUT) — but still keep the per-sequence run well under the time budget. "
    "Move tensors to cuda once and keep the optimisation loop tight; .item()/.cpu() only at the end.")


# --- M2 SCAFFOLD: locked front-end, agent adds ONLY loop closure ------------------------------
# After 4 from-scratch M2 attempts all destabilised the VO front-end (worse than basic VO), this
# isolates the loop-closure skill: the agent is GIVEN the proven front-end (frontend.py, seeded
# into its workspace) and may only build the loop-closure + pose-graph layer on top.
_FRONTEND_KITTI_CODE = (Path(VO_CODE_DIR) / "frontend_kitti.py").read_text()

KITTI_SLAM_SCAFFOLD_DESCRIPTION = (
    "Implement LOOP CLOSURE + POSE-GRAPH optimisation for stereo visual SLAM on outdoor driving "
    "(KITTI). A PROVEN, LOCKED stereo-VO front-end is already provided as `frontend.py` in your "
    "working directory — you MUST import and use it, and you MUST NOT modify, rewrite, delete, or "
    "reimplement it (it scores ~2.8% drift on its own; reproducing or 'improving' it is forbidden "
    "and defeats the task).\n"
    "`from frontend import run_frontend` gives you `fe = run_frontend(os.environ['LAB_DATA'])`, a "
    "dict with: fe['n'] frames; fe['K'] intrinsics; fe['traj'] (n,3) camera centres; fe['poses'] a "
    "list of n 4x4 cam->world VO matrices; and fe['frames'], a list of per-frame dicts each with "
    "'idx', 'kps' (Nx2 pixels), 'des' (Nx32 uint8 ORB descriptors), 'pts3d' (Kx3 metric 3-D points "
    "in that frame's camera coords), 'pidx' (which kps have a 3-D point), and 'Twc' (4x4 VO pose).\n"
    "YOUR JOB — build ONLY the loop-closure layer:\n"
    "  (1) LOOP DETECTION (appearance, NOT VO position — position has drifted): for each frame, "
    "match its 'des' against EARLIER frames' 'des' (a bag-of-words / descriptor-voting signature), "
    "shortlist top candidates with a sufficient temporal gap.\n"
    "  (2) GEOMETRIC VERIFICATION: confirm a candidate with enough inlier matches; reject weak ones "
    "(a FALSE loop is catastrophic). Estimate the relative pose between the two frames using their "
    "'pts3d' (3D-3D / Kabsch or 3D-2D PnP).\n"
    "  (3) GLOBAL POSE-GRAPH OPTIMISATION: build a graph whose nodes are the frame poses fe['poses'] "
    "with sequential odometry edges (from consecutive VO poses) plus your verified loop edges; "
    "optimise all poses on SE(3) (e.g. Gauss-Newton / scipy.least_squares with Lie-algebra "
    "increments), anchored at frame 0, to distribute the loop drift.\n"
    "  (4) Write $LAB_ARTIFACTS/traj.txt (one `tx ty tz` per frame) AND $LAB_ARTIFACTS/poses.txt "
    "(row-major 3x4 [R|t] cam->world per frame, 12 numbers/line) from the OPTIMISED poses.\n"
    "GUARD: if you detect no trustworthy loops, just write the front-end's own fe['traj']/fe['poses'] "
    "unchanged (never do WORSE than the front-end). numpy, opencv, scipy available. Do not read GT.")


# --- M2 DECOMPOSITION: pose-graph optimisation GIVEN an oracle loop-detection ----------------
# Isolates detection vs optimisation: the agent is handed the LOCKED front-end AND correct loop
# constraints (loops.txt oracle), and authors ONLY the pose-graph optimisation. Clears the bar ->
# the wall was loop DETECTION; fails even with perfect loops -> optimisation itself is the wall.
KITTI_SLAM_ORACLE_DESCRIPTION = (
    "Implement ONLY the global POSE-GRAPH OPTIMISATION for stereo visual SLAM — loop detection is "
    "GIVEN to you (an oracle), and the front-end is LOCKED. Two provided inputs you MUST use and MUST "
    "NOT reimplement:\n"
    "  • `frontend.py` (in your working dir): `from frontend import run_frontend`; "
    "`fe = run_frontend(os.environ['LAB_DATA'])` returns the proven VO — fe['n'], fe['poses'] (list "
    "of n 4x4 cam->world matrices), fe['frames'] (per-frame 'Twc'), etc. Do NOT modify it.\n"
    "  • `$LAB_DATA/loops.txt` — CORRECT loop closures (an oracle; you do NOT need to detect loops). "
    "Each line is: `i j  r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz` where (R|t) is the 3x4 relative "
    "pose T_ij between frame i and a later frame j that revisits it, such that Twc_j = Twc_i @ T_ij.\n"
    "YOUR JOB — author the optimisation only:\n"
    "  (1) Build a pose graph: nodes = the frontend frame poses fe['poses']; SEQUENTIAL ODOMETRY "
    "edges between consecutive frames (relative pose from fe['poses'][k], fe['poses'][k+1]); and a "
    "LOOP edge for every line of loops.txt (the given T_ij constraint between i and j).\n"
    "  (2) Optimise ALL poses on SE(3) to satisfy odometry + loop edges (e.g. Gauss-Newton / "
    "scipy.least_squares with Lie-algebra increments), anchored with frame 0 fixed at identity — this "
    "distributes the accumulated drift so the loops close.\n"
    "  (3) Write $LAB_ARTIFACTS/traj.txt (one `tx ty tz` per frame) AND $LAB_ARTIFACTS/poses.txt "
    "(row-major 3x4 [R|t] cam->world per frame, 12 numbers/line) from the OPTIMISED poses.\n"
    "The loops are CORRECT — a standard pose-graph optimisation using them should noticeably reduce "
    "drift versus the raw front-end. numpy, opencv, scipy available. Do not read any ground truth "
    "trajectory (loops.txt is the only provided oracle).")


def vo_impl_task_kitti_slam_oracle(threshold: float = 1.8, *, dev: str = "06",
                                   heldout: tuple[str, ...] = ("07", "09")):
    """M2 decomposition: agent authors ONLY the pose-graph optimisation, given the locked front-end
    + a correct loop-detection oracle (loops.txt). Isolates whether the wall is detection or
    optimisation. Use with LoopOracleKITTIProvider (writes loops.txt) + seed frontend.py."""
    from ..plugins.vo_kitti import kitti_datasets

    return ImplementationTask(
        description=KITTI_SLAM_ORACLE_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_KITTI_EVAL_CODE,
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,
        metric="t_err_pct", op="<=", threshold=threshold,
        datasets=kitti_datasets(dev, heldout),
        entry_filename="main.py",
        label_filename="gt*.txt",   # strips gt.txt/gt_poses.txt; loops.txt (the oracle) survives
    )


def vo_impl_task_kitti_slam_scaffold(threshold: float = 1.8, *, dev: str = "06",
                                     heldout: tuple[str, ...] = ("07", "09")):
    """M2 SCAFFOLD: the agent is given the LOCKED proven front-end (frontend.py, seeded into its
    workspace by the run script) and authors ONLY the loop-closure + pose-graph layer. Same official
    KITTI t_err grading on loopy held-out 07/09. Isolates the loop-closure skill from the front-end
    the agent kept breaking in the 4 from-scratch attempts."""
    from ..plugins.vo_kitti import kitti_datasets

    return ImplementationTask(
        description=KITTI_SLAM_SCAFFOLD_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_KITTI_EVAL_CODE,
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,
        metric="t_err_pct", op="<=", threshold=threshold,
        datasets=kitti_datasets(dev, heldout),
        entry_filename="main.py",
        label_filename="gt*.txt",
    )


def vo_impl_task_kitti_slam(threshold: float = 1.8, *, dev: str = "06",
                            heldout: tuple[str, ...] = ("07", "09"), gpu: bool = False):
    """M2: KITTI stereo SLAM (loop closure + pose graph), graded on the official KITTI t_err over
    LOOPY held-out sequences (07/09 return near their start). Default bar 1.8% sits between basic
    stereo VO (~2.4% on these seqs) and the ideal-closure 1.32% proven offline / ORB-SLAM2 1.15%.

    gpu=True runs in vo-gpu-torch:1 (torch.cuda + cv2) and tells the agent to put the bundle
    adjustment + pose-graph optimisation on the GPU — removing the CPU BA speed wall from M1."""
    from ..plugins.vo_kitti import kitti_datasets

    desc = KITTI_SLAM_TASK_DESCRIPTION + (KITTI_SLAM_GPU_PARAGRAPH if gpu else "")
    return ImplementationTask(
        description=desc,
        framework=_GPU_FW if gpu else _CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_KITTI_EVAL_CODE,
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,
        metric="t_err_pct", op="<=", threshold=threshold,
        datasets=kitti_datasets(dev, heldout),
        entry_filename="main.py",
        label_filename="gt*.txt",
    )


# --- CONTAMINATION PROBE: neutral stereo VO on synthetic (provably-unseen) data --------------
# Deliberately KITTI-free wording so the agent approaches a novel domain fresh — the test is
# whether its authoring is real capability or carries memorised KITTI priors. From-scratch is the
# justified exception to the incremental-build default (we are measuring authoring, not climbing).
SYNTH_TASK_DESCRIPTION = (
    "Implement a STEREO visual-odometry algorithm. From $LAB_DATA read rectified grayscale stereo "
    "pairs (left_%06d.png, right_%06d.png) and intrinsics.txt (one line: fx fy cx cy baseline_m — "
    "the horizontal stereo baseline in metres). USE THE STEREO PAIR to recover ABSOLUTE (metric) "
    "scale: a disparity gives metric depth (Z = fx * baseline / disparity), so back-project "
    "features to 3-D and estimate the camera motion (e.g. 3D-2D PnP) — the trajectory must be "
    "metric. Motion is smooth forward translation with gentle turns. Write $LAB_ARTIFACTS/traj.txt "
    "with one `tx ty tz` (camera centre) per frame, in order. ALSO write $LAB_ARTIFACTS/poses.txt "
    "with the full per-frame camera pose as a row-major 3x4 [R|t] cam->world matrix (12 numbers per "
    "line) — this lets the grader score the official length-normalized translational error (t_err "
    "%) AND rotational error (r_err deg/m). You are graded on MULTIPLE held-out sequences you never "
    "see. Do not read any ground truth. (numpy, opencv, scipy available; cv2.StereoSGBM_create "
    "computes disparity.)")


def vo_impl_task_synth(threshold: float = 4.0, *, dev=None, heldout=None):
    """Contamination-probe task: neutral stereo VO graded on PROCEDURALLY SYNTHESIZED held-out
    sequences (no model has seen them). Bar default 4% — generous; we care WHETHER the agent
    authors a working metric VO on novel data and how close to the reference's ~1.7%, not a tight
    SOTA gate. dev/heldout are ignored (the SyntheticStereoProvider owns the splits)."""
    from ..plugins.vo_synth import synth_datasets

    return ImplementationTask(
        description=SYNTH_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_KITTI_EVAL_CODE,
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,
        metric="t_err_pct", op="<=", threshold=threshold,
        datasets=synth_datasets(),
        entry_filename="main.py",
        label_filename="gt*.txt",
    )


# --- M3: VISUAL-INERTIAL ODOMETRY (fuse IMU) -------------------------------------------------
# The agent is given the LOCKED proven stereo VO front-end + an IMU stream, and must FUSE them to
# bridge vision blackouts. Incremental-build: it does not re-author the VO, only the fusion.
SYNTH_VIO_TASK_DESCRIPTION = (
    "Implement VISUAL-INERTIAL ODOMETRY (VIO) — fuse a stereo camera with an IMU. Two provided "
    "inputs you MUST use and MUST NOT reimplement:\n"
    "  • `frontend.py` (in your working dir): `from frontend import run_frontend`; "
    "`fe = run_frontend(os.environ['LAB_DATA'])` returns the proven stereo VO — fe['poses'] (list of "
    "n 4x4 cam->world matrices) and fe['frames'] (per frame: 'pts3d' metric 3-D points, 'kps', 'des'). "
    "Do NOT modify it.\n"
    "  • `$LAB_DATA/imu.txt`: n lines, each `wx wy wz ax ay az` = body(camera)-frame gyroscope "
    "(rad/s) and accelerometer specific force (m/s^2, GRAVITY INCLUDED, magnitude ~9.8 at rest), the "
    "measurement over interval [i, i+1] at dt = 0.1 s.\n"
    "THE CATCH — vision sometimes FAILS: there are stretches of textureless frames where the stereo "
    "VO finds no features and HOLDS its pose (fe['poses'] stops moving; fe['frames'][i]['pts3d'] is "
    "empty/tiny). Across those gaps the VO loses the real motion. The IMU keeps measuring — but the "
    "IMU ALONE drifts badly (bias + noise), so you cannot just integrate it.\n"
    "YOUR JOB — author the FUSION:\n"
    "  (1) Use the VO motion where vision works.\n"
    "  (2) DETECT vision failure (few/zero pts3d, or the VO step is ~stationary while the IMU shows "
    "motion).\n"
    "  (3) BRIDGE each blackout by integrating the IMU: rotate by the gyro (R <- R*exp(w*dt)); for "
    "translation, carry the world VELOCITY estimated from the last good VO steps, rotate the "
    "accelerometer to world and SUBTRACT GRAVITY (a_world = R*accel - g, g=(0,-9.81,0) in the "
    "convention where +y is down means g=(0,+9.81,0) — verify the sign so a static body integrates "
    "to no motion), then p += v*dt + 0.5*a*dt^2; v += a*dt. Resync velocity to the VO when vision "
    "returns.\n"
    "  (4) Write $LAB_ARTIFACTS/traj.txt (`tx ty tz` per frame) AND $LAB_ARTIFACTS/poses.txt (3x4 "
    "[R|t] cam->world per frame). Graded on held-out sequences by official t_err. Fusing beats both "
    "VO-alone (fails on blackouts) and IMU-alone (drifts). numpy/opencv/scipy available; no GT.")


def vo_impl_task_synth_vio(threshold: float = 4.0, *, dev=None, heldout=None):
    """M3: agent authors IMU-VO FUSION on synthetic visual-inertial sequences with vision blackouts.
    Locked front-end + IMU provided; bar default 4% (beat VO-alone, which fails the blackouts). Use
    with SyntheticVIOProvider + seed frontend.py."""
    from ..plugins.vo_synth import vio_datasets

    return ImplementationTask(
        description=SYNTH_VIO_TASK_DESCRIPTION,
        framework=_CPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_KITTI_EVAL_CODE,
        heldout_infer_command=_INFER_CMD, heldout_infer_code=_INFER_CODE,
        metric="t_err_pct", op="<=", threshold=threshold,
        datasets=vio_datasets(),
        entry_filename="main.py",
        label_filename="gt*.txt",   # strips gt.txt/gt_poses.txt; imu.txt (provided sensor) survives
    )


def kitti_stereo_reference_author():
    """Writes the reference classical stereo VO as main.py (baseline + offline proof, no API)."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_KITTI_STEREO_REFERENCE_MAIN)
        return Usage()
    return author


def kitti_degenerate_author():
    """Static (origin) trajectory for KITTI stereo — the negative control (counts left_*.png)."""
    src = ("import os, glob, numpy as np\n"
           "d=os.environ['LAB_DATA']; a=os.environ['LAB_ARTIFACTS']; os.makedirs(a,exist_ok=True)\n"
           "n=len(glob.glob(os.path.join(d,'left_*.png')))\n"
           "np.savetxt(os.path.join(a,'traj.txt'), np.zeros((n,3)), fmt='%.6f')\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
        return Usage()
    return author


# --- KITTI LEARNED VO (the GPU / learned-research track) --------------------------------

from ..plugins.vo import _GPU_FW                                              # noqa: E402

_KITTI_LEARNED_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_kitti_learned.py").read_text()
_LEARNED_EVAL_CODE = (Path(VO_CODE_DIR) / "eval_learned.py").read_text()      # harness-owned grader

KITTI_LEARNED_TASK_DESCRIPTION = (
    "Implement a LEARNED monocular visual-odometry algorithm with PyTorch that TRAINS ON THE "
    "GPU. From $LAB_DATA/train/seq_*/ read left_%06d.png + poses.txt (KITTI 3x4 cam->world per "
    "frame, the supervision) and train a network to predict the 6-DoF relative pose between "
    "consecutive frames. Then, for each $LAB_DATA/test_input/seq_<s>/ (left frames, NO labels), "
    "run inference and ACCUMULATE the predicted relative poses into a camera trajectory, writing "
    "$LAB_ARTIFACTS/traj_<s>.txt (one `tx ty tz` per frame). You are graded by held-out Sim(3)-"
    "aligned ATE on test sequences you never see the labels for. torch (CUDA), numpy, opencv are "
    "available; use torch.cuda. Do not read any test ground truth.")


def vo_impl_task_kitti_learned(threshold: float,
                               train: tuple[str, ...] = ("00", "02", "06", "08", "09"),
                               test: tuple[str, ...] = ("05", "07")):
    """Learned-VO Track B task: the agent authors a torch training pipeline that runs on the
    GPU (framework=torch -> vo-gpu-torch:1, --gpus all). Graded by eval_learned.py (Sim(3) ATE
    on held-out sequences). The training is a harness JOB — wall-clock, not tokens."""
    from ..plugins.vo_kitti_learned import kitti_learned_datasets

    return ImplementationTask(
        description=KITTI_LEARNED_TASK_DESCRIPTION,
        framework=_GPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_LEARNED_EVAL_CODE,
        metric="ate_rmse", op="<=", threshold=threshold,
        datasets=kitti_learned_datasets(train, test),
        entry_filename="main.py",
    )


# --- RUNG 3: LEARNED VO on the CONTAMINATION-CLEAN synthetic domain --------------------------
# Learned methods are the most memorisation-prone; training + testing on procedural synthetic data
# the model cannot have seen makes "did it LEARN or MEMORISE" cleanly separable (unlike leaky KITTI).
SYNTH_LEARNED_TASK_DESCRIPTION = (
    "Implement a LEARNED monocular visual-odometry algorithm with PyTorch that TRAINS ON THE GPU. "
    "From $LAB_DATA/train/seq_*/ read left_%06d.png + poses.txt (a row-major 3x4 cam->world matrix "
    "per frame, the supervision) and train a network to predict the 6-DoF relative pose between "
    "consecutive frames. Then, for each $LAB_DATA/test_input/seq_<s>/ (left frames, NO labels), run "
    "inference and ACCUMULATE the predicted relative poses into a camera trajectory, writing "
    "$LAB_ARTIFACTS/traj_<s>.txt (one `tx ty tz` per frame). You are graded by held-out Sim(3)-"
    "aligned ATE on test sequences whose labels you never see. torch (CUDA), numpy, opencv are "
    "available; use torch.cuda. Do not read any test ground truth.")


def vo_impl_task_synth_learned(threshold: float = 5.0, *, train=None, test=None):
    """Rung 3: learned monocular VO trained on the GPU, on PROVABLY-UNSEEN synthetic sequences
    (SyntheticLearnedProvider). Graded by eval_learned (Sim(3) ATE). Bar in metres; default 5 m is a
    'the net learned something + generalises' gate (learned VO was sub-classical in prior work)."""
    from ..plugins.vo_synth import synth_learn_datasets

    return ImplementationTask(
        description=SYNTH_LEARNED_TASK_DESCRIPTION,
        framework=_GPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_LEARNED_EVAL_CODE,
        metric="ate_rmse", op="<=", threshold=threshold,
        datasets=synth_learn_datasets(),
        entry_filename="main.py",
    )


def kitti_learned_reference_author():
    """Writes the reference pure-torch learned VO as main.py (trains on GPU; baseline)."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_KITTI_LEARNED_REFERENCE_MAIN)
        return Usage()
    return author


def kitti_learned_degenerate_author():
    """Static (origin) trajectory per test sequence — the negative control for learned VO."""
    src = ("import os, glob, numpy as np\n"
           "d=os.environ['LAB_DATA']; a=os.environ['LAB_ARTIFACTS']; os.makedirs(a,exist_ok=True)\n"
           "for sd in sorted(glob.glob(os.path.join(d,'test_input','seq_*'))):\n"
           "    s=os.path.basename(sd).replace('seq_','')\n"
           "    n=len(glob.glob(os.path.join(sd,'left_*.png')))\n"
           "    np.savetxt(os.path.join(a,f'traj_{s}.txt'), np.zeros((n,3)), fmt='%.6f')\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
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
