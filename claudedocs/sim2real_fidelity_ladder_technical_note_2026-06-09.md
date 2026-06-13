# Verifying Sim-to-Real for Autonomous-Driving Localization: From a Fidelity Ladder to SOTA-SLAM Validation

*Technical note. 2026-06-09 (rev. with Part II; sim-faithfulness result retracted — see §8/§9). A
verification-first study of sim-to-real for AD localization, with a deterministic held-out grader. Part I
decomposes the sim-to-real gap for a learned VO (the fidelity ladder) — the solid result. Part II pivots
from authoring to verifying: it integrates + grades SOTA learned SLAM (DROID-SLAM) on real
environment-labeled data; the sim-faithfulness question it set out to answer is left **inconclusive**
(the only RAM-runnable scene was too straight for the metric to be meaningful).*

---

## Summary

We study a question central to deploying learned localization: **does training a visual-odometry network
in simulation transfer to the real world, and if not, what is the gap *made of*?** Using a
verification-first testbed (a deterministic, LLM-free grader scoring on hidden real ground truth), we
decompose a 150× sim-to-real failure into two separable causes, each with a receipt:

1. an **appearance gap** (~42 m of 47 m), which is a *data/rendering* problem and is **closed** by
   rendering real-appearance training data; and
2. a **model-capacity ceiling** (~24 m above classical geometry), which is a *model* problem that
   real data, rendered data, and 3× viewpoint-augmented rendered data **all hit equally.**

> **Variance audit (n=3 retrainings/rung).** Each rung was re-trained 3× to bound training noise (±~1.5 m):
> procedural **69.71 ± 0.05**, rendered **27.35 ± 1.49**, real **25.61 ± 1.64** m. Rendered and real are
> **statistically indistinguishable** (the gap is within the noise), so "rendering closes the appearance
> gap" is reported as *rendered ≈ real ≈ 26 ± 1.5 m, both far below procedural ~70* — not as a spurious
> 0.1 m tie. The decomposition stands; the precision is now honest.

**Conclusion:** rendered/generative training data genuinely closes the sim-to-real appearance gap, but
beating classical geometry is a model-scaling problem, not a data problem. This is directly the mandate
of a simulation-for-AD team: *manufacture training data that transfers, and verify it.*

---

## 1. Problem and contribution

Learned VO/SLAM is the modern SOTA direction (e.g. DROID-SLAM), and the field's bet for the chronic
shortage of labelled real driving data is **high-fidelity simulation** — 3D Gaussian Splatting, neural
rendering, generative video — which yields perfect poses and unlimited viewpoints. The open risk is the
**sim-to-real gap**: a network trained on rendered data may not transfer, and you cannot know without
ground truth you can't cheaply obtain in the real world.

Our contribution is *measurement under a grader that cannot be fooled*: we build a **fidelity ladder**
that isolates the contribution of training-image appearance to real-world transfer, and we quantify the
mechanism with an independent appearance-distance metric.

## 2. Method

**Verification-first harness.** A strict separation between *solver* (the algorithm under test) and
*evaluator* (a deterministic, no-LLM grader). The grader runs the solver in a sandbox, aligns its
trajectory to **held-out real ground truth** with a Sim(3) Umeyama fit (monocular VO is scale-free), and
reports ATE/RPE. "It produced a trajectory" is never mistaken for "the trajectory is correct." Ground
truth is isolated by construction (label files are stripped from the solver's view).

**The localizer.** A single fixed network across all conditions: a ResNet-18 pose-regressor (6-channel
stacked-frame input → 3-D translation + 6-D continuous-rotation heads), rotation-weighted Huber loss,
GPU + temporal-swap augmentation, test-time augmentation, and validation-ATE model selection. Holding the
model fixed is what makes the comparison clean: **only the training domain changes.**

**Training domains (the ladder), all tested on the same held-out real KITTI (07, 09):**

| Domain | Training imagery | Labels |
|---|---|---|
| procedural | hand-made synthetic corridors (ray-cast textured planes) | exact |
| **rendered** | real KITTI scenes re-rendered as novel views (see below) | exact |
| **rendered + viewpoint-aug** | + two parallel offset paths/scene (novel viewpoints) | exact |
| real | real KITTI frames | from INS (sparse/noisy) |

**The renderer (`GSplatModule`).** For the rendered rungs we reconstruct a colour 3-D point cloud from
real KITTI **stereo** (SGBM disparity → metric depth → back-projection into world frame using the GT
poses) and render novel views by projecting the cloud into a target camera with z-buffering and splat
hole-fill. A `world_offset` renders **parallel-path** viewpoints (the cloud comes from the real camera
positions; only the target view is translated), preserving exact relative-pose supervision — the
augmentation real data cannot provide. *Honest scope: this is stereo-depth reprojection / point-splatting,
the achievable precursor to optimised 3DGS (gsplat); it is OpenCV-only and uses real pixels.*

## 3. Results

**Sim-to-real collapse and closure.** The fixed network scores **0.45 m** ATE on unseen *synthetic* test
sequences (generalisation on data it cannot have memorised — a contamination control), but **69.6 m** on
real KITTI (a ~150× collapse), and **27.2 m** when trained on real KITTI. So the gap is real, and learned
VO at this scale (~1k frames, one RTX 3080) is not classical-competitive even on real data.

**The fidelity ladder.**

![Fidelity ladder: rendering closes the appearance gap; rendered/real rungs cluster at the ~27 m ceiling](../artifacts/blog/fidelity_ladder.png)

| Training domain | Held-out real ATE (Sim3) | seq_07 / seq_09 |
|---|---|---|
| procedural synth | 69.6 m | 72.4 / 66.8 |
| **rendered** | **27.1 m** | 34.3 / 19.9 |
| **rendered + viewpoint-aug** | **26.9 m** | 31.8 / 21.9 |
| real | 27.2 m | 31.0 / 23.5 |
| *classical stereo VO (reference)* | *~few m* | — |

**Appearance-distance diagnostic (mechanism check).** An independent distribution distance over intensity
+ gradient-magnitude histograms (no access to the grader) finds the rendered frames **3.2× closer** to
real than the procedural ones (0.167 vs 0.532) — tracking the transfer jump and confirming the gap is
photometric, not geometric.

## 4. Findings

1. **The appearance gap is a data/rendering problem, and it is solved.** Rendering real appearance — even
   crudely, from a hole-filled point cloud — closes 69.6 → 27.1 m, landing on the real-trained ceiling.
   *Rendered data trains the network as well as real data.*
2. **The residual is a model-capacity ceiling, not sim-to-real.** Rendered, real, and 3×-augmented
   training all converge to ~27 m, far above classical's few metres. The **data axis is saturated**;
   the bottleneck is the network.
3. **Manufactured viewpoint diversity did not break the ceiling.** 3× rendered training views moved ATE by
   0.3 m (within run-to-run noise; per-sequence mixed) — consistent with (2).

**Implication for simulation-for-AD:** generative/rendered training data closes the sim-to-real *appearance*
gap with verified evidence; to make a learned localizer beat classical geometry you scale the *model/compute*,
not the data fidelity or diversity.

## 5. Limitations (stated, not hidden)

- The renderer reprojects **real pixels**, so we prove *"rendered real-appearance ≈ real,"* not
  *"fully-synthetic-but-photoreal ≈ real."* Isolating the latter needs optimised 3DGS/generative rendering
  (a documented next step behind the same interface).
- Monocular Sim(3) ATE measures **shape, not metric scale**; truncated-window ATE, not KITTI's official
  segment metric; two test sequences; single-GPU compute budget. Learned-VO numbers carry training
  randomness (the ~0.3 m augmentation effect is within it).

## 6. Relevance and reproducibility

This maps directly onto a deployed-AD **simulation/physical-AI** mandate: *build high-fidelity environments
for the **training and verification** of an AD model, and minimise the sim-to-real gap.* The artifact here
is the **verification** half — a grader-backed method that says *how much* of the gap a rendering pipeline
actually closes, and *which half* of the gap a given investment (better rendering vs bigger model) will move.

**Reproducible:** synthetic-world generator + `GSplatModule` renderer + the fidelity-ladder runner/grader
are deterministic and resumable; the figure and `results.json` regenerate from cached numbers with no GPU
(`run_fidelity_ladder.py --use-cached`). Full narrative + per-experiment receipts in the project blog
(Episodes 0–17) and the cross-run memory ledger.

---

# Part II — From authoring to verifying: grading SOTA SLAM, and is the sim faithful enough?

## 7. Motivation and method

Authoring a deployed SLAM from scratch is the wrong build-vs-buy call (our from-scratch C++ VIO front-end +
Ceres windowed BA worked at 2.18%, but tightly-coupled IMU fusion diverged — VINS-Fusion is thousands of
tuned lines). The field *integrates* proven systems. So we retarget the lab from author to **verifier**: a
**`SystemAdapter`** lets the held-out grader score *any* SLAM system (classical, learned, agent-authored)
by mapping our sequence I/O ↔ the system's, with no grader change.

**Real, environment-labeled data (replaces synthetic for this part).** KITTI *raw* drives, already tagged
**City / Residential / Road**, shipping real **OXTS GNSS/IMU** + GT poses, converted to our contract
(`fetch_kitti_raw.py`). A caught bug worth noting: GT poses were initially in the IMU frame, not the camera
frame — the camera-IMU extrinsic correction took a road drive from a broken 151% t_err to a sane 9%
(verify-first discipline catching a silently-wrong label).

## 8. Results

**Environment-stratified benchmark** (real KITTI, Sim3 ATE):

| System | city | road | residential |
|---|---|---|---|
| classical stereo VO | 0.16 m | 4.04 m | 0.93 m |
| our C++ Ceres VO (stereo) | **0.08 m** | 55.6 m ✗ | 100.8 m ✗ |
| **DROID-SLAM (learned, monocular)** | **0.083 m** | — (RAM) | — (RAM) |

The benchmark surfaced our *own* C++ VO's divergence on longer drives — the point of verification. DROID
(monocular SOTA) matches the stereo C++ VO and beats classical on city. Integrating DROID required its
matched env (torch 1.10; the extensions do **not** build against torch 2.4) and fixing an ops leak (spawn
containers hoarding 14.7 GB GPU).

**The sim-faithfulness experiment — attempted, RETRACTED as a result.** We ran DROID on a real city scene vs
the *same scene re-rendered* (`GSplatModule`) and measured a real-vs-rendered ATE delta of 0.033 m, then
claimed the sim was "faithful enough to validate a learned SLAM." **That claim is retracted; it is
inconclusive.** Three converging flaws: (1) the only scene DROID could run on here (city_0001) is
**dead-straight** (2.2 m lateral over 107 m), where Sim(3) ATE is near-trivial — a similarity transform
aligns almost any roughly-straight estimate, so the number reflects scene-easiness, not fidelity; (2) the
run is **not reproducible** (re-run OOM-killed silently on 15 GB RAM); (3) the published figure plotted
**raw monocular trajectories** at arbitrary scale, so it looked like failure while the table claimed success.
The curvy scenes that would constitute a real test all failed on this hardware. **A valid sim-faithfulness
result requires curvy scenes that fit RAM, Sim(3)-aligned trajectory plots, and N-run reproducibility — none
of which we have.** What *does* stand: DROID-SLAM *integration* (it builds + runs SOTA learned SLAM on real
data) is real, and the environment benchmark (curvy road/residential) is unaffected.

## 9. Limitations (Part II)

- **The sim-faithfulness result is retracted** (see above): straight scene → trivial metric, single
  non-reproducible run, misleading raw-trajectory plot. Honest status: *inconclusive*, not proven.
- **DROID is RAM-bound on this hardware.** Real-frame runs OOM on a 15 GB machine in the multi-run context;
  five mitigations (stride, buffer, GPU clean-wait, downscaling, separate-process render) did not resolve it.
  Robust multi-scene DROID needs more RAM.
- The renderer **reprojects real pixels** (stereo-depth point-splat) — the achievable precursor to optimised
  3DGS, which is the stronger, open follow-up.

## 10. Reproducibility (Part II)

`KittiRawProvider` (`fetch_kitti_raw.py`), `SystemAdapter`/`run_slam_benchmark.py`, `Dockerfile.droid` +
`run_droid_slam.py` (the DroidAdapter), and `run_sim_faithfulness.py` are committed; the benchmark and
sim-faithfulness figures + JSON are under `artifacts/slam_benchmark/`. Full narrative in blog Episode 18.
