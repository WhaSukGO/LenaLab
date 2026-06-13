# LenaLab — Results & Highlights

*A verification-first computer-vision research lab: solutions count only when an **independent,
deterministic verifier** measures them on **held-out data the solver never saw**. Built toward
**autonomous-driving localization** — simulation-for-training and SLAM. This page is the evidence; the full
chronicle is [`claudedocs/blog_agent_in_a_lab_2026-06-03.md`](claudedocs/blog_agent_in_a_lab_2026-06-03.md).*

The single thing that makes this lab different: **"it ran" is never success.** Every number below is
verified-reproducible, variance-bounded, honestly caveated — or retracted. (One flagship result *was*
retracted; see the audit.)

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

## How to navigate
- **Full chronicle (Episodes 0–20):** [`claudedocs/blog_agent_in_a_lab_2026-06-03.md`](claudedocs/blog_agent_in_a_lab_2026-06-03.md)
- **Accurate-results report (per-claim verdicts):** [`claudedocs/accurate_results_report_2026-06-11.md`](claudedocs/accurate_results_report_2026-06-11.md)
- **Sim-to-real technical note:** [`claudedocs/sim2real_fidelity_ladder_technical_note_2026-06-09.md`](claudedocs/sim2real_fidelity_ladder_technical_note_2026-06-09.md)
- **How the harness works:** [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)
- **Figures:** `artifacts/slam_benchmark/`, `artifacts/blog/`, `artifacts/fidelity_ladder/`

## Honest edges (stated, not hidden)
- Learned numbers carry ~±1.5 m training noise; only comparative claims surviving it are kept.
- Sim-faithfulness established only for short close-range scenes; long-range is an open gap.
- 3DGS pipeline works but the render is basic (no adaptive densification); crisp render is future work.
- Single consumer GPU (16 GB VRAM caps DROID's keyframe buffer on the longest sequence).
