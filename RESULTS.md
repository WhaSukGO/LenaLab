# LenaLab — Results & Highlights

*An AI agent doing real computer-vision research — analyzing the problem, designing an approach,
implementing and training the algorithm, and confirming it generalizes — across
**autonomous-driving localization + perception** (SLAM, simulation-for-training, surround-camera BEV,
3D occupancy). This page is the evidence; the full chronicle is
[`claudedocs/blog_agent_in_a_lab_2026-06-03.md`](claudedocs/blog_agent_in_a_lab_2026-06-03.md).*

What makes the numbers below worth reading: every one is measured on **data the agent never trained
on**, so it counts only when the work genuinely generalizes — reproducible, variance-bounded, and
honestly caveated (one early result didn't hold up and was retracted; the rest were checked across
repeated runs). The agent does the research; held-out measurement keeps the scoreboard honest.

---

## 1. Real SLAM on real km-scale driving — SOTA, working, visualized

DROID-SLAM (learned SOTA) run on the canonical **KITTI odometry** loops — real driving, kilometre-scale,
with loop closure. Implemented **stereo** DROID (metric scale) and benchmarked vs monocular and classical.

| sequence | loop length | **stereo DROID** | monocular | classical VO | our C++ VO |
|---|---|---|---|---|---|
| seq_07 | 0.7 km | **0.39 m (0.06%)** | 24.2 m | 3.07 m | diverged |
| seq_09 | 1.7 km | **3.43 m (0.20%)** | 93.2 m | — | — |
| seq_05 | 2.2 km | **0.60 m (0.03%)** | 55.1 m | — | — |

t_err for stereo seq_07 ≈ **0.28%** (better than typical published DROID). Stereo kills the monocular
scale-drift; SOTA-learned SLAM beats classical ~8×; our hand-built C++ VO can't survive a kilometre.

**Watch SLAM work, not just read ATE:**
- `artifacts/slam_benchmark/droid_stereo_seq07.png` — the estimate overlaying a real 695 m loop
- `artifacts/slam_benchmark/droid_map_seq07.png` — the **dense 3D map** (400k points) DROID reconstructs
- `artifacts/slam_benchmark/tracking_seq07.gif` — **feature tracking** (flow trails) on the driving video
- `artifacts/slam_benchmark/loop_build_seq07.gif` — the **loop closing** frame-by-frame
- `artifacts/slam_benchmark/leaderboard_seq07.png` — multi-system overlay

→ *Autonomous-driving SLAM.* (Blog Episode 19.)

---

## 2. Sim-to-real for learned localization — the fidelity ladder (variance-audited)

Does training a VO network in simulation transfer to the real world, and if not, what is the gap *made of*?
We decompose it with a deterministic held-out grader, **n=3 retrainings per rung** (mean ± std):

| training domain → real | held-out ATE |
|---|---|
| procedural synthetic | 69.71 ± 0.05 m |
| **rendered** (real-appearance) | **27.35 ± 1.49 m** |
| rendered + viewpoint-aug | 26.49 ± 0.93 m |
| **real** | **25.61 ± 1.64 m** |

**Rendering closes the appearance gap** — rendered ≈ real (within noise); both far below procedural. The
residual ~26 m is a model-capacity ceiling that real, rendered, and augmented data all hit equally. So:
*generative/rendered data genuinely closes sim-to-real; beating classical geometry is a model-scaling
problem, not a data problem.* See [`claudedocs/sim2real_fidelity_ladder_technical_note_2026-06-09.md`](claudedocs/sim2real_fidelity_ladder_technical_note_2026-06-09.md).

→ *Simulation for AD perception/training.* (Blog Episode 17.)

---

## 3. Can a rendered sim *validate* a SLAM? — scene-dependent (honest)

Ran DROID on real scenes vs the same scenes **re-rendered**, across 7 scenes:

- **Short, close-range scenes** (≤170 m): rendered ≈ real (delta ≤ 0.14 m) → sim is faithful.
- **Long / far-field scenes** (>350 m): rendered diverges (delta 2–9 m) → the stereo-reprojection renderer
  has no far structure, so DROID drifts on it.

**Verdict: scene-dependent — not a blanket yes.** A first version of this overclaimed "proven" and was
**retracted**; the corrected, evidence-based answer is above. Optimized 3DGS (Episode 20, pipeline built,
quality open) is the route that might extend faithfulness to long range.

---

## 4. The discipline — a trustworthiness audit of every number

Triggered by the retraction, every blog headline was re-audited: scene-degeneracy checks, n-run variance,
reproductions. Outcome: **1 retraction, 2 corrections, the rest verified.** Two of the auditor's *own*
diagnoses were corrected on the record (a docker relative-path bug mis-blamed on RAM; a WSL `.wslconfig`
memory cap mistaken for hardware reboots). Per-claim verdict table:
[`claudedocs/accurate_results_report_2026-06-11.md`](claudedocs/accurate_results_report_2026-06-11.md).

*This is the differentiator: a lab that audits and retracts its own results is one whose surviving numbers
you can trust.*

---

## 5. Agent-authored CV — the lab's thesis

An LLM agent authoring vision algorithms from scratch, graded by the verifier:
- **Learned VO** (ResNet pose-CNN, 6-D rotation): 0.55 ± 0.05 m on unseen synthetic, ~6× the reference.
- **Contamination probe**: agent-authored stereo VO **1.20%** beats the reference on provably-unseen data
  (capability, not dataset recall).
- **VIO** (IMU fusion): agent **3.83%** vs VO-alone 18%.
- Honest negatives kept: from-scratch loop closure and C++ tightly-coupled IMU fusion **failed** and are
  recorded as such.

---

## 6. A second problem class — the lab generalizes (multi-camera BEV perception)

The strongest test of a *harness* is whether it works on a task it wasn't shaped around. Every
result above is **ego-motion** (*where did the camera go?*). So the harness was rebuilt end-to-end
for a fundamentally different problem — **Bird's-Eye-View perception**: fuse **6 surround cameras**
into a top-down vehicle-occupancy map (nuScenes), scored by **IoU**, a metric with nothing in
common with trajectory error.

A sandboxed agent, given only the data contract + grid spec (never *how*), authored a Lift-Splat
network from scratch — and we ran it **three times**. The honest result is two-sided and, we think,
more valuable than a single lucky pass:

| | n=3 held-out IoU | mean ± std | vs bar 0.08 |
|---|---|---|---|
| **agent** (re-authors each run) | 0.108 / 0.038 / 0.111 | **0.085 ± 0.034** | **2 / 3 VERIFIED** |
| **fixed-recipe reference** (3 seeds) | 0.138 / 0.142 / 0.143 | 0.141 ± 0.002 | always passes |

The agent **can** author real multi-view BEV perception — two of three runs cleared the held-out
bar with genuine Lift-Splat networks (one even implemented *correct* flip augmentation: swap
left/right cameras **and** update extrinsics). But it's **not robust**: one run in three
self-sabotaged (over-aggressive validation holdout on tiny data) and *failed*. The diagnostic
settles why — a **fixed-architecture reference is rock-stable (std 0.002)**, so the variance is the
agent's redesign-every-run latitude, **not** the task (`artifacts/bev/bev_variance_n3.png`).

A single run would have over-claimed "VERIFIED 0.1075"; the harness caught it wasn't reproducible.
**Then we closed the loop.** The diagnosis predicted the fix: lock the fragile parts (geometry +
the correct flip augmentation + training) into a seeded core the agent can't edit, and have it
author **only the network** — its variance should collapse. We built that scaffold and ran it n=3:

| condition | n=3 mean ± std | pass |
|---|---|---|
| fixed-recipe reference | 0.141 ± 0.002 | 3/3 |
| agent **free-form** (authors everything) | 0.085 ± 0.034 | 2/3 |
| agent **scaffold** (authors only the network) | **0.136 ± 0.005** | **3/3** |

**Confirmed** (`artifacts/bev/bev_scaffold_compare.png`): locking the fragile parts collapsed the
agent's variance **7.3×** and lifted it to near-reference quality — 3/3 clearing the bar, all three
leaving the locked core byte-for-byte untouched. This is the lab's full cycle on one problem:
**build → find it's non-robust → diagnose → prescribe a fix → validate it.** The agent's freedom is
both its power (it invents real architectures) and its risk (it can self-sabotage the fragile glue);
the harness tells the difference, and the scaffold keeps the freedom where it helps.

- Figures: `bev_scaffold_compare.png` (the payoff), `bev_variance_n3.png`, `bev_before_after.png`,
  sweep `bev_sweep_scene0103.gif` · full report + diagnosis + scaffold:
  [`claudedocs/bev_track_b_report_2026-06-15.md`](claudedocs/bev_track_b_report_2026-06-15.md)
- The whole verification spine transferred: harness-owned GT + IoU grader (anti-tamper, 2 passing
  tests), held-out scene split, calibrated oracle (degenerate → 0.000, REJECTED).

*Honest scope:* nuScenes **mini** (10 scenes), vehicle-class only, from-scratch backbones — so
absolute IoU is below full-nuScenes LSS. The claim is **the harness generalizes, keeps results
honest, and turns a non-robust finding into a validated fix** — not a SOTA BEV number.

→ *Perception for autonomous driving.* **Five domains (monocular VO, RGB-D VO, SLAM, KITTI stereo,
BEV) — and on BEV the lab did its hardest job twice: it caught a non-robust result, then fixed it.**

---

## 7. Into 3D — camera→occupancy, and a finding that replicates

The sixth domain pushes from 2D to **3D semantic occupancy** (the current AD-perception frontier):
6 surround cameras → a **200×200×12 voxel grid** of vehicle occupancy (nuScenes), scored by per-voxel
IoU. It also answers a science question left open by BEV: *was the "agent free-form is high-variance,
a scaffold fixes it" finding BEV-specific, or does it generalize?*

| condition | n | held-out voxel IoU | std | pass |
|---|---|---|---|---|
| fixed-recipe reference | 3 | 0.099 | 0.003 | — |
| agent **free-form** | 3 | 0.086 (0.054/0.113/0.092) | **0.024** | 2/3 |
| agent **scaffold** (authors only the net) | 3 | 0.079 (0.076/0.084/0.076) | **0.004** | 3/3 |

**It replicates.** Free-form is again high-variance (one self-sabotaging run); the scaffold collapses
the variance **~6×** at a clean n=3 (`artifacts/occ/occ_scaffold_compare.png`), with all 3 runs passing
and each leaving the locked 3D core byte-for-byte unmodified. So the cross-domain rule — *the agent's
freedom is its variance source; scaffolding scopes it* — now holds in **both 2D and 3D**. Honest
nuance: on 3D the scaffold buys *reliability*, not a higher peak (the locked core's capacity caps it
below free-form's best 0.113).

Prediction viz on held-out scenes: `artifacts/occ/occ_pred_heldout.png`. Full report (incl. an
honestly-recorded 3-hour training hang + the watchdog fix):
[`claudedocs/occ_domain_report_2026-06-19.md`](claudedocs/occ_domain_report_2026-06-19.md).

→ *3D perception for autonomous driving.* **Six agent-authored domains: monocular VO, RGB-D VO, SLAM,
KITTI stereo, BEV, and 3D occupancy.**

---

## How to navigate
- **Full chronicle (Episodes 0–20):** [`claudedocs/blog_agent_in_a_lab_2026-06-03.md`](claudedocs/blog_agent_in_a_lab_2026-06-03.md)
- **BEV Track-B report (the second problem class):** [`claudedocs/bev_track_b_report_2026-06-15.md`](claudedocs/bev_track_b_report_2026-06-15.md)
- **Occupancy report (3D, the sixth domain):** [`claudedocs/occ_domain_report_2026-06-19.md`](claudedocs/occ_domain_report_2026-06-19.md)
- **Accurate-results report (per-claim verdicts):** [`claudedocs/accurate_results_report_2026-06-11.md`](claudedocs/accurate_results_report_2026-06-11.md)
- **Sim-to-real technical note:** [`claudedocs/sim2real_fidelity_ladder_technical_note_2026-06-09.md`](claudedocs/sim2real_fidelity_ladder_technical_note_2026-06-09.md)
- **How the harness works:** [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)
- **Figures:** `artifacts/slam_benchmark/`, `artifacts/blog/`, `artifacts/fidelity_ladder/`

## Honest edges (stated, not hidden)
- Learned numbers carry ~±1.5 m training noise; only comparative claims surviving it are kept.
- Sim-faithfulness established only for short close-range scenes; long-range is an open gap.
- 3DGS pipeline works but the render is basic (no adaptive densification); crisp render is future work.
- Single consumer GPU (16 GB VRAM caps DROID's keyframe buffer on the longest sequence).
