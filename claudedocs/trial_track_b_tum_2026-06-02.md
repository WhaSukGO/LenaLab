# Trial Report — Track B (agent-authored VO) on real TUM data

- Date: 2026-06-02
- Lab: `vo_lab` (ver3 solver) on the Touchstone (ver2) verification spine
- Trial: **live Track B** — a sandboxed Claude agent authors a monocular Visual Odometry
  algorithm from scratch, graded by an independent evaluator on a held-out real trajectory.
- Command: `ANTHROPIC_API_KEY=… python -m vo_lab.run_vo_tum_implement 0.1337`

---

## 1. Dataset

| | |
|---|---|
| Source | **TUM RGB-D**, sequence `freiburg1_xyz` |
| URL | https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz (~0.47 GB) |
| Frames used | first **200** RGB frames (640×480, grayscale), `max_frames=200` |
| Intrinsics | fr1 calibrated: fx=517.31, fy=516.47, cx=318.64, cy=255.31 |
| Ground truth | motion-capture trajectory, pre-associated to frames by timestamp (≤20 ms) |
| Split | solver sees **frames only**; the GT trajectory is **held-out** (evaluator-only) |
| Download | once, cached at `~/.cache/vo_lab/tum` (shared across runs) |

Chosen over KITTI because KITTI odometry images are a single ~22 GB download (no
per-sequence option); TUM `fr1/xyz` is the standard small monocular benchmark with
ground truth and simple translational motion.

## 2. Implementation (what the agent wrote)

The agent authored **`main.py` (360 lines)** entirely on its own, in a container sandbox
(no host shell, no network, `eval.py` off-limits). Archived: `artifacts/agent_authored_vo_tum_v1.py`.

Its approach (from its own docstring + code) — notably more sophisticated than the ORB
reference baseline:
- **PnP-centric** pose estimation (`solvePnPRansac`) against a maintained 3-D landmark map.
- **Deferred initialization**: waits until baseline/median-depth ratio is good enough for a
  reliable two-view triangulation, then **back-fills** the earlier frames with PnP.
- **Reprojection-pruned landmark map** (REPROJ_THR=4 px, capped at 3000 landmarks).
- **LK optical-flow tracking** between frames + re-detection when tracked count drops.
- Guards: rotation-magnitude limit, minimum-landmark / minimum-PnP-point checks, graceful
  "hold last pose" when PnP fails on a frame.

I/O contract (same as synthetic): reads `$LAB_DATA` frames + `intrinsics.txt`, writes
`$LAB_ARTIFACTS/traj.txt` (one `tx ty tz` per frame).

## 3. How it was verified

1. Sandboxed Claude session writes + tests `main.py` in the `vo-cpu-opencv:1` container.
2. Independent `ScriptEvaluator` re-instantiates the harness-owned grader (`eval.py`),
   runs the authored code, and measures **ATE-RMSE after Sim(3) Umeyama alignment**
   (scale-corrected — monocular scale is unobservable) against the **held-out** GT.
3. Oracle bar = **baseline-beating**: reference classical ORB-VO scored **0.0891 m** on the
   same held-out split, bar set to ×1.5 = **0.1337 m**.

## 4. Elapsed time

| Phase | Time |
|---|---|
| Dataset download | once, ~minutes (cached thereafter; 0 s on this run) |
| **Agent authoring session** | **≈ 27 min** wall-clock (03:17:36 → 03:44:47 UTC), hit the 40-turn limit |
| Independent grading (run + eval in Docker) | ≈ 15 s |
| Token spend | ~300 k (order-of-magnitude; see note in §6) |

## 5. Result

| Metric | Value |
|---|---|
| Verdict (graded) | **VERIFIED** ✅ |
| Held-out ATE-RMSE (Sim3) | **0.1242 m** (≤ 0.1337 m bar) |
| Frames scored | 200 |
| Recovered global scale | 0.038 (monocular; absorbed by Sim(3) alignment) |
| Reference baseline (classical ORB-VO) | 0.0891 m |

**Interpretation:** the agent's VO clears the bar on real footage but is *less accurate
than the simple classical reference* (0.124 vs 0.089 m). The trajectory plot shows good
early/mid tracking in X-Z, then **drift** toward the end (typical monocular behavior) and
under-capture of the lateral X-Y back-and-forth.

## 6. The failure, and the fix

The live run reported `RESULT: FAILED` — **not a hang**, and not an algorithm failure. The
agent kept refining to chase the tight bar and hit its authoring **`max_turns=40`** limit;
the SDK raised, and the harness discarded 27 min of *working* code as FAILED.

- **Salvage**: grading the agent's exact `main.py` on held-out → **VERIFIED, 0.124 m**.
- **Fix shipped**: `resilient_sdk_author` (in `vo_lab/agents/vo_implementer.py`) — if an
  authoring session ends early but left a valid entry file, the evaluator grades it anyway
  (a turn limit must not discard a working artifact). Turn budgets raised (TUM 80,
  synthetic 40). Token under-reporting on the early-exit path is a known minor gap.
- **Still open (genuine hangs)**: `job_runner` runs containers with **no execution
  timeout**; a real infinite loop in authored code could still hang. A job-level watchdog
  is the recommended next reliability fix.

## 7. Demo artifacts

- **Trajectory plot:** `artifacts/demo/trajectory.png` — estimated (red, Sim3-aligned) vs
  ground truth (black), X-Z and X-Y projections.
- **Demo video:** `artifacts/demo/vo_demo.mp4` (200 frames, 15 fps) — input frame on the
  left, trajectory traced so far on the right.
- **Authored algorithm:** `artifacts/agent_authored_vo_tum_v1.py`.

Regenerate (no API, runs the VO locally on the cached frames):
```bash
PYTHONPATH=. python -m vo_lab.visualize \
  artifacts/agent_authored_vo_tum_v1.py \
  _vo_tum_impl_run/cache/data/vo-tum-frames \
  _vo_tum_impl_run/cache/heldout/vo-tum-gt/gt.txt \
  artifacts/demo
```

## 8. Reproduce the trial

```bash
# 1) calibrate on real data (local, non-billed): downloads once, sets the bar
PYTHONPATH=. python -m vo_lab.run_vo_tum_calibration            # -> bar ≈ 0.1337 m
# 2) live Track B (billed + Docker): agent authors VO, graded on held-out
ANTHROPIC_API_KEY=… python -m vo_lab.run_vo_tum_implement 0.1337
```
Prereqs: Docker + image `docker build -f docker/Dockerfile.cpu-opencv -t vo-cpu-opencv:1 .`;
`ANTHROPIC_API_KEY` in `.env`.
