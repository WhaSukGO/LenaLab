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

---

## 9. Update — clean re-run (v2), after the resilient-author fix

The run above hit the 40-turn limit and was salvaged manually. After shipping
`resilient_sdk_author` (grades the artifact even if the session ends early) and raising the
budget to 80 turns, the live run was repeated and **completed end-to-end, VERIFIED**:

| Metric | v1 (salvaged) | **v2 (clean re-run)** |
|---|---|---|
| Verdict | VERIFIED (manual salvage) | **VERIFIED (automatic)** |
| Held-out ATE-RMSE (Sim3) | 0.124 m | **0.052 m** |
| vs reference baseline (0.089 m) | worse | **better** |
| Authored approach | PnP-centric, deferred init | `goodFeaturesToTrack` + optical-flow tracking, **wider keyframe baseline (skip=2)**, SIFT fallback, keyframe interpolation |
| Algorithm file | `artifacts/agent_authored_vo_tum_v1.py` | `artifacts/agent_authored_vo_tum_v2.py` |

The session again ended via the SDK (`error result: success`), and the resilient author
graded the left artifact automatically — no manual salvage needed. A background watchdog
(kill any sandbox container running >360 s) guarded the unattended run and fired zero kills
(the run was healthy throughout). Demo regenerated from v2:
`artifacts/demo/{trajectory.png, vo_demo.mp4, vo_demo.gif}` (ATE 0.052 m).

> **⚠️ Correction (2026-06-04).** 0.052 m is monocular **Sim(3) shape** accuracy: the grader
> rescales for free (the raw trajectory was ~40× the GT size, `recovered_scale ≈ 0.025`), so
> "better than the 0.089 m reference" is a shape comparison, not metric, and is n=1 (no
> variance). Graders now emit a `scale_implausible`/`caveats` field with every number.

---

## 10. Improvement experiment — RGB-D + generalization + RPE

Motivated by the analysis that the monocular setup discards depth, scores on one sequence,
and reports only ATE. New modules (all in `vo_lab/plugins/vo_rgbd.py` + `vo_ref/run_rgbd.py`
+ `vo_ref/eval_rgbd.py`):

1. **Depth exposed (RGB-D)** — the provider materializes the TUM depth channel
   (`depth_%04d.png`, metres = pixel / 5000) so the solver can recover **metric** scale.
2. **Generalization grader** — runs the solver's code on **held-out *sequences* it never
   authored against** (`seq_*/input`), with GT isolated at `seq_*/gt.txt`. Scored with
   **SE(3)** alignment (no scale freebie) so depth must actually be used; reports ATE, **RPE**
   (drift), and a scale-error diagnostic.

**Reference RGB-D VO (3D-2D PnP) validation on real data (local, non-billed):** mean
held-out ATE on the *unseen* `fr1_desk` = **0.057 m**, RPE 0.011, **scale_err 0.077**
(near-metric — depth works). Degenerate control = 0.70 m → REJECTED; gate **OPEN**, bar
0.086 m. The discrimination margin (~12×) is far wider than the monocular gate's.

Run: `python -m vo_lab.run_vo_tum_rgbd_calibration` (local) →
`python -m vo_lab.run_vo_tum_rgbd_implement 0.086` (live, billed). Offline coverage:
`tests/test_vo_rgbd_provider.py` (depth exposure + GT isolation).

**Live RGB-D agent run — VERIFIED.** A sandboxed Claude agent authored a multi-strategy
RGB-D VO (SIFT→3D-2D PnP RANSAC primary, KLT optical-flow fallback, keyframe recovery; depth
for metric scale). Graded on the **unseen** `fr1_desk` with SE(3)-metric alignment:
**ATE-RMSE = 0.033 m**, RPE 0.010, **scale_err 0.032** (near-perfect absolute scale) — under
the 0.086 m bar and better than the classical RGB-D reference (0.057 m). This is the strongest
and most honest result in the project: metric (no scale freebie) and on a scene the solver
never authored against. Algorithm archived `artifacts/agent_authored_vo_rgbd_v1.py`; demo
`artifacts/demo_rgbd/`.

| | monocular v2 | reference RGB-D | **agent RGB-D** |
|---|---|---|---|
| held-out ATE | 0.052 m (Sim3) | 0.057 m (SE3) | **0.033 m (SE3)** |
| alignment | scale-corrected | metric | **metric** |
| scored on | same sequence | unseen sequence | **unseen sequence** |

**Process note (a bug + fix).** The first live RGB-D attempt FAILED after ~1.17M tokens: the
RGB-D `DatasetRef` names contained a `:` which broke Docker `-v host:container` mounts, so
every sandbox run errored (the agent authored blind). Local-mode calibration had passed
because it uses no `-v` mounts. Fix: mount-safe dataset names + a **Docker-mode** reference
dry-run (not just local) to validate the live path. The relaunch then succeeded.

---

## 11. SLAM with loop closure — reference works; live agent attempt FAILED (honest)

**Reference SLAM works (validated, committed).** `vo_ref/run_slam.py`: RGB-D VO front-end →
keyframes → geometrically-verified loop detection → self-contained SE(3) pose-graph
optimization (scipy). On the `fr1_room` loop sequence: VO-only ATE **0.86 m** vs reference
SLAM **0.23 m** (~73% drift reduction; 12–42 loop closures). Docker calibration gate OPEN
(bar 0.347 m); VO-only and a degenerate control both fail it → loop closure is *necessary*.

**Live agent SLAM attempt — REJECTED (a genuine negative result).** A sandboxed Claude agent
authored a 352-line SLAM (SIFT matching + loop detection + pose-graph). Outcome:
- Graded fairly (run to completion), its trajectory ATE = **412 m** — the **pose-graph
  optimization diverged** (catastrophic blow-up on a 3 m-scale trajectory). The verifier
  correctly **REJECTED** it — which is the lab working as intended (it caught a broken result).
- Compounded by **my harness misconfiguration**: the hang watchdog killed at 480 s, but the
  agent's SLAM took ~507 s *inside the container* (slower than its 153 s host run — fewer
  cores), and the grader's own timeout is 600 s. So every in-container test was killed
  *before finishing* → the agent never saw its own output and couldn't debug the divergence.
  Watchdog threshold corrected to 900 s (> grader's 600 s).

**Takeaways.** (1) The verifier did its job — a broken SLAM scored 412 m and was rejected;
nothing false was accepted. (2) SLAM-from-scratch in one session is materially harder than
VO/RGB-D for the agent (it got the *structure* right but the optimizer diverged). (3) The
runtime/safety budget must let the agent actually *see* its results (watchdog > grader
timeout; per-run fast enough to iterate). (4) Loop closure itself is demonstrated in the repo
via the working reference. The live agent SLAM is left as an open frontier, not a success.

---

## 12. SLAM re-run — VERIFIED, after the watchdog fix + failure-memory loop (2026-06-03)

The §11 attempt was left an open frontier. Two fixes shipped between sessions: the watchdog
threshold was raised to **900 s** (> the 600 s grader, so in-container tests finish and the
agent can debug), and a **cross-run failure-memory loop** (`vo_lab/memory.py`) now injects
prior failures into the author's prompt — the 412 m divergence was backfilled into the ledger
with a fix hint (`vo_lab/backfill_slam.py`).

Re-running the **same task** (`fr1_room`, bar 0.347 m, root `_vo_slam_impl_run2`):

| Metric | §11 (v1) | **§12 re-run (v2)** |
|---|---|---|
| Verdict | REJECTED | **VERIFIED** ✅ |
| Held-out ATE (SE3 metric) | 412 m (diverged) | **0.185 m** |
| RPE / scale_err | — | 0.024 / 0.127 |
| vs reference SLAM (0.23 m) | — | **better** |
| vs VO-only (0.86 m) | — | **4.6× better** |
| Tokens / wall | (killed) | ~4.89 M / ~52 min |
| Algorithm | `agent_authored_vo_slam_v1.py` | `agent_authored_vo_slam_v2.py` |

**The failure memory was causally used.** The agent's own docstring: *"Pose graph:
TRANSLATION-ONLY linear system (rotations fixed from VO) → sparse LSQR, guaranteed no
divergence … Fallback: VO-only if … ill-conditioned."* It abandoned the non-linear pose graph
that diverged in v1 for a provably-stable linear solve, added a VO-only fallback, and
sanity-checks each loop constraint against VO before trusting it — directly answering the
injected lesson ("DIVERGED / verify it beats VO-only"). The agent reached for a stable
formulation *because it had been told what failed*. Demo figure:
`artifacts/blog/ep5_slam_verified.png`.

**Net:** the failure→memory→recovery loop works (the agent fixed its own divergence).

> **⚠️ Correction (2026-06-04, external review).** This SLAM run graded on its TRAINING
> sequence: `dev=fr1_room, heldout=(fr1_room,)` (identical frames/params). So **0.185 m is
> in-sample, NOT a generalization result** — it does not show the SLAM transfers to an unseen
> scene (unlike RGB-D fr1_xyz→fr1_desk and KITTI 00→05,07, which are disjoint). The recovery
> narrative stands; the generalization claim does not. The SLAM config now refuses
> same-sequence dev/held-out; a disjoint-sequence re-run (e.g. held-out fr2_desk, needs a
> download) is the honest follow-up.

---

## 13. KITTI stereo — cross-domain generalization, VERIFIED first try (2026-06-03)

Every prior trial was TUM (indoor, hand-held). The open question: did the agent learn *visual
odometry*, or *TUM*? Test: **KITTI** outdoor driving — a new domain (large-scale forward
motion, different image statistics) AND a new modality (**stereo**, never attempted before;
prior tracks were mono/RGB-D). Stereo's calibrated baseline keeps scale observable, so the bar
stayed honest: **SE(3) metric**, graded on unseen sequences. New code is solver-side only
(`vo_kitti.py` provider + `run_kitti_stereo.py` reference); the grader `eval_rgbd.py` was
**reused unchanged** — it is modality-agnostic (runs the authored code on each held-out seq,
scores traj.txt vs gt.txt; never reads the input frames).

Calibration (local, non-billed): reference stereo VO mean held-out ATE **3.48 m** (seq 05/07),
scale_err 0.010 (metric works), degenerate control 90.9 m REJECTED, gate OPEN, bar 5.22 m.

Live agent run (billed, Docker):

| Metric | Value |
|---|---|
| Verdict | **VERIFIED** ✅ (first attempt) |
| Mean held-out ATE (SE3 metric) | **3.53 m** (seq 05: 3.17, seq 07: 3.89) |
| scale_err | **0.007** (0.7% — metric scale nailed from stereo) |
| vs reference stereo VO (3.48 m) | **matches** |
| Tokens / wall | ~4.03 M / ~run |
| Algorithm | `artifacts/agent_authored_vo_kitti_v1.py` |

The agent composed a correct stereo pipeline from scratch (its docstring: SGBM disparity →
metric depth → ORB BF matching → PnP+RANSAC → inlier refinement, constant-velocity fallback) —
no prior KITTI or stereo experience. Demo: `artifacts/blog/ep6_kitti_verified.png` (the metric
trajectory tracks hundreds of metres of driving on both unseen sequences).

**Net:** the agent's VO transfers across BOTH a new domain and a new modality, matching the
classical reference on the first attempt. The strongest evidence yet that the lab has been
certifying real capability, not dataset overfitting — and the verifier earned the conclusion by
scoring on data and a sensor the agent never touched. The KITTI success was auto-recorded to the
cross-run memory (domain `kitti`), so a future stereo/driving agent starts from this approach.

---

## 14. Track A — autonomous committee lineage on REAL data (the autonomy pillar, 2026-06-03)

Every prior section is Track B (single-shot authoring). Track A is the OTHER half: a committee
(PI + Geometry/SLAM + Data experts) running an autonomous *lineage* of menu-constrained
experiments, each independently verified. Previously it ran only on a synthetic world where the
ORB params barely moved the metric (loop machinery, not discovery). This run points it at real
TUM fr1/xyz, where the knobs matter, with cross-run within-lineage memory (ver2 ResearchHistory).

Live run (`run_vo_tum_committee`, billed): calibration gate OPEN (reference VERIFIED, degenerate
REJECTED), then a 6-experiment lineage:

| Experiment | nfeatures | ransac | ratio | held-out ATE (Sim3) |
|---|---|---|---|---|
| vo-real-001 | 1500 | 0.9 | 0.8 | 0.155 m |
| vo-real-002 | 1000 | 0.7 | 0.75 | 0.141 m |
| vo-real-003 | — | — | — | **FAILED** (MenuError: proposal mis-named the recipe; loop survived) |
| vo-real-004 | 1200 | 1.0 | 0.0 | 0.154 m |
| **vo-real-005** | **400** | **1.0** | **0.0** | **0.114 m** (best) |
| vo-real-006 | 600 | 0.7 | 0.75 | 0.155 m |

**Result:** the committee autonomously improved its held-out ATE **26%** (0.155 → 0.114 m),
discovering a non-obvious optimum (fewer features + no ratio test). Deliberation was genuine —
hundreds of words of geometric reasoning per expert (Lowe ratio test on near-planar matches,
RANSAC angular tolerance at fr1/xyz, monocular scale drift under Sim(3)); it even flagged a
hypothesis/params contradiction in one proposal. A mid-lineage FAILURE was recorded and skipped
(crash-resumable). ~440k tokens. Demo: `artifacts/blog/ep7_committee.png`.

**Honest bound:** it did NOT beat the hand-tuned reference default (0.089 m) — it found the best
option within the menu's reach, which is the job of a menu-constrained panel. The point of Track A
is the demonstration that the lab can **self-direct a verified research program** (propose → verify
→ learn → propose), the half of "an AI research lab" that single-shot Track-B authoring never shows.
The run was stopped after experiment 6 (it kept proposing follow-ups; the demonstration was
complete). Known robustness gap surfaced: the PI sometimes uses the experiment id as the recipe id
(the vo-real-003 MenuError) — handled gracefully but worth hardening.

---

## 15. Learned VO on GPU — agent authors ML, beats the reference, VERIFIED (2026-06-03)

The last frontier: a different KIND of research — machine learning, not classical geometry.
Can the agent author code that TRAINS a neural network on the GPU? Infra: a CUDA PyTorch image
(`vo-gpu-torch:1`), training as a harness GPU job (gpu_lease + `--gpus all`) — wall-clock, not
tokens. Data: KITTI (train seqs 00/02/06/08/09 with GT poses visible — supervision is legit;
test seqs 05/07 labels isolated). Graded Sim(3) (monocular scale unobservable) + RPE.

Reference learned VO (pure-torch pose CNN): held-out ATE 31.5 m — honestly ~9× worse than
classical (monocular learned VO is drift-dominated). Bar set generously (×1.3 = 40.96 m): the
test was whether the agent can do ML research, not beat classical.

Live agent run (`run_vo_kitti_learned_implement`, billed, GPU):

| Metric | Value |
|---|---|
| Verdict | **VERIFIED** ✅ |
| Held-out ATE (Sim3) | **19.8 m** (seq_05 23.8, seq_07 15.8) — **beats the 31.5 m reference by 37%** |
| RPE/frame | **0.62 m** (reference 1.54 m — less than half the drift) |
| Tokens / GPU wall | ~2.79 M / ~1530 s (25 min of GPU jobs) |
| Algorithm | `artifacts/agent_authored_vo_learned_v1.py` (313 lines) |

The agent first checked `torch.cuda.is_available()` in the sandbox (confirming GPU before
authoring), then innovated beyond the reference: an **8-channel input stacking both RGB frames
with dense optical flow**, a compact CNN trained from scratch with a cosine LR schedule, images
preloaded to RAM. The optical flow (explicit motion cues) is a genuine ML design choice and
halved the per-frame drift. Demo: `artifacts/blog/ep8_learned.png`.

**Honest bound:** at 19.8 m it is still ~5–6× worse than classical VO (3.5 m) — monocular learned
VO drifts, and the verifier never let it hide. But the agent took a learned baseline and pushed it
meaningfully forward with a real ML idea. The capability demonstrated — authoring AND training a
GPU neural network, with innovation — is the point; the sub-classical number is honest. Recorded
to cross-run memory (domain `learned-vo`).
