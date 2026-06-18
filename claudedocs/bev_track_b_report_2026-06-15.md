# Bringing a Second Problem Class into the Lab: Multi-Camera BEV Perception

*Project report · 2026-06-15 · LenaLab (verification-first CV research lab)*

---

## 1. Why this exists

LenaLab's thesis is narrow and testable: an AI agent's work counts **only** when an independent,
deterministic verifier measures it on **held-out data the agent never saw and cannot game** —
"it ran" is never success. So far that thesis was proven on four **ego-motion** domains
(monocular VO, RGB-D VO, SLAM, KITTI stereo): all variations on *where did the camera go?*

A fair skeptic asks: **is the harness just a VO-shaped trick, or does the verification-first
discipline actually generalize to a different kind of computer vision?**

This report answers that by standing up a genuinely different problem class — **multi-camera
Bird's-Eye-View (BEV) perception** — inside the same harness, end-to-end, and showing every
piece of the verification spine transfers: held-out split, harness-owned ground truth, an
independent metric, an anti-tamper grader, a calibrated oracle, and a sandboxed agent author.

---

## 2. The task

Given a nuScenes sample's **6 surround cameras** + their intrinsics + camera→ego extrinsics,
predict a top-down **vehicle-occupancy grid** in the ego frame
(100 m × 100 m at 0.5 m/cell → 200 × 200). This is the canonical Lift-Splat vehicle-segmentation
task. It exercises geometry the VO domains never touched:

- **cross-view fusion** — six cameras pointing in different directions, fused into one frame,
- **per-pixel depth lifting** — each image pixel becomes a ray of 3-D hypotheses,
- **a metric ego-frame raster** — output lives in world metres, not pixel space,

and it's scored by **Intersection-over-Union**, an area metric with nothing in common with the
trajectory-error metrics (ATE / t_err) the lab used before. Nothing about the VO grader could be
reused; the whole evaluation contract had to be rebuilt — which is exactly the point.

---

## 3. Architecture — how it plugs into the verification-first harness

A LenaLab domain is the tuple `{dataset adapter, harness-owned GT + metric, held-out split,
reference bar}`, behind which a sandboxed agent authors the algorithm. All five were built:

```
 nuScenes mini ──prep──▶  ~/.cache/vo_lab/bev/{train,val}/*.npz       (one-time, scripts/prep_nuscenes_bev.py)
                              │
                              ▼
   Provider (NuScenesBEVProvider) lays out, per run:
     LAB_DATA/train/<tok>.npz       6 cams + calib + bev GT   ← agent may train on these
     LAB_DATA/test_input/<tok>.npz  6 cams + calib (NO bev)   ← agent predicts on these
     [held-out] <tok>_bev.npy       secret BEV GT             ← only the grader sees these
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
   AGENT authors main.py             HARNESS owns eval_bev.py
   (trains on GPU, writes            (loads secret GT, computes IoU,
    pred_<tok>.npy per sample)        restored before judging — un-tamperable)
              └───────────────┬───────────────┘
                              ▼
                    Oracle:  miou ≥ bar  →  VERIFIED / REJECTED
```

**What the solver controls:** only `main.py` — the network, the training, the occupancy
threshold. **What the harness controls and the solver can never touch:** the held-out scenes,
the ground-truth rasterization, the IoU metric, and the pass/fail bar. The grader is restored
from the task spec immediately before judging, so even a malicious `eval.py` earns nothing.

This mirrors the proven learned-VO track (GPU training + held-out generalization), so the BEV
domain reuses the *same* `build_vo_implementer_harness`, sandbox image (`vo-gpu-torch:1`), and
`resilient_sdk_author` — only the task spec, provider, grader, and reference are new.

---

## 4. Harness-owned ground truth (and why it's trustworthy)

`scripts/prep_nuscenes_bev.py` rasterizes the BEV GT: for each sample it takes every
`vehicle.*` 3-D box, transforms its footprint from global into the sample's **ego frame**
(translate then rotate by the inverse ego pose), and fills the polygon into the 200 × 200 grid.
Held-out = the official nuScenes `mini_val` scenes (`scene-0103`, `scene-0916`), disjoint from
`mini_train`. Result: **323 train / 81 val** samples.

The GT was validated two independent ways before any metric was trusted:

1. **By eye** (`artifacts/bev/bev_gt_check.png`): on a parking-lot scene the footprints cluster
   in coherent parking-row structure around the ego marker — geometry, not noise.
2. **By learnability**: a model *fits* it (IoU climbs from 0 to ~0.17). A geometrically broken
   GT cannot be learned — so the fact that training works is itself a correctness proof.

---

## 5. The reference baseline & the numbers

`vo_lab/plugins/vo_ref/run_bev_learned.py` is a self-contained **Lift-Splat-Shoot** model:
ResNet-18 → per-pixel (softmax depth distribution × context) → lift to a camera frustum of 3-D
points using the real intrinsics + cam→ego extrinsics → voxel-pool (scatter-add) into the ego
grid → conv head → occupancy logits.

| setting | held-out vehicle-IoU | notes |
|---|---|---|
| reference, **ImageNet-pretrained** backbone, n=3 seeds | **0.169 ± 0.002** | tight variance → a stable bar |
| reference, **from-scratch** backbone (sandbox: no network) | **0.1042** | the honest sandbox bar |
| all-zero **degenerate** control | **0.0000** | the grader is no rubber stamp |
| busy held-out samples (pretrained) | ~0.22 | `artifacts/bev/bev_pred_heldout.png` |

The two reference numbers differ because the sandbox has **no network access**, so neither the
reference nor the agent can download pretrained weights — both train from scratch. That lowers
IoU (0.169 → 0.104) but it's the *honest* sandboxed setting, and it's the number that sets the
agent's bar.

**Oracle bar = 0.08** (from-scratch reference ÷ 1.3, the same margin policy as the learned-VO
track). **Calibration gate: OPEN** (reference 0.104 ≫ bar 0.08 ≫ degenerate 0.000).

---

## 6. Validation done *before* spending a token

- **End-to-end calibration** (`python -m vo_lab.run_bev_calibration`, non-billed): trains the
  reference in the real Docker sandbox, grades it with the harness-owned IoU grader, runs the
  degenerate control — gate OPEN. This exercised provider → sandbox train+infer → grader → oracle
  on real hardware with zero API cost.
- **Anti-tamper** (`tests/test_bev_implementer.py`, 2 tests, passing): an all-zero author is
  REJECTED; a fake `eval.py` reporting `miou=1.0` is overridden by the restored grader and still
  REJECTED. Same gaming-resistance the VO domain has.
- **Variance**: the pretrained reference's 0.169 ± 0.002 over 3 seeds confirms the bar is a stable
  property, not a lucky run.

---

## 7. The live agent runs (n=3) — capability, but **not robust**

A sandboxed Claude agent (`claude-sonnet-4-6`), given only the data contract and the grid spec,
authored a Lift-Splat network from scratch — three independent times. The honest picture:

| run | held-out mean IoU | verdict |
|---|---|---|
| run 1 | 0.1075 | ✅ VERIFIED |
| run 2 | **0.0376** | ❌ **REJECTED** |
| run 3 | 0.1107 | ✅ VERIFIED |
| **n=3** | **0.085 ± 0.034** (range 0.038–0.111) | **2 / 3 cleared the 0.08 bar** |

A *single* run would have reported a clean "VERIFIED at 0.1075" — and it would have been partly
luck. Running it three times reveals the truth: **the agent can author a passing BEV network, but
not reliably.** One run in three (run 2) landed at less than half the IoU and *failed*. This is
exactly what variance testing exists to catch, and the lab's norm is to report it, not bury it.

### Is the variance the *task* or the *agent*? (the diagnostic)

Decisive cheap experiment: train a **fixed-architecture** from-scratch reference at three seeds and
measure its spread. If the task were inherently noisy at this data scale, the reference would swing
too. It doesn't:

| | n=3 held-out IoU | mean | **std** |
|---|---|---|---|
| **fixed-recipe reference** (seeds 0/1/2) | 0.138 / 0.142 / 0.143 | 0.141 | **0.002** |
| **agent** (re-authors each run) | 0.108 / 0.038 / 0.111 | 0.085 | **0.034** |

The fixed recipe is **rock-stable (std 0.002)** and sits *above* the agent's best run. So the task
is stably learnable and the regime is **not** the problem — the agent's variance (≈17× the
reference's) comes from it **redesigning the algorithm every run**. Figure:
`artifacts/bev/bev_variance_n3.png`. (The diagnostic uses best-epoch val-IoU, so its *absolute*
0.141 is a touch optimistic vs the final-epoch sandbox reference 0.104 that set the bar; the point
is the **variance**, which is unambiguous.)

The failure is concrete, not mysterious. Comparing the authored code:
- **run 2 (failed)** carved out **15 % of the already-tiny 323-sample training set** for threshold
  calibration (≈275 left to train on) and used a simpler flip augmentation (`flip_K` adjusts only
  `cx`) — if its flipped BEV *target* wasn't ego-y-flipped to match the flipped cameras, the
  augmentation injects label noise. Either choice plausibly costs the thin margin.
- **runs 1 & 3 (passed)** calibrated on the full training set and (run 3) implemented flip
  augmentation *correctly* (explicit ego-frame + extrinsic + intrinsic update).

So the agent's freedom to redesign — the very thing that makes Track B "authoring, not tuning" — is
also the variance source: good designs clear the bar, self-sabotaging ones (over-aggressive
holdout, subtly-wrong augmentation) don't. The best run's predictions
(`artifacts/bev/bev_agent_heldout.png`, before/after `bev_before_after.png`, sweep
`bev_sweep_scene0103.gif`) are real multi-view perception; the *reliability* is the open question.

**What this does and doesn't show.** It **does** show the harness generalizes to a brand-new problem
class and grades it reliably (it caught the non-robustness a single run would have hidden), and that
a capable agent *can* author real BEV perception (2/3). It does **not** show the agent reliably
matches a fixed reference — it doesn't, yet, at this data scale. The legitimate paths to a *robust*
agent result are (a) **more data** (a larger nuScenes subset stabilizes both reference and agent) or
(b) a **scaffold** that fixes the architecture and has the agent author only the uncertain piece —
the same "isolate the skill" move the SLAM track used. Re-rolling live runs until one passes would
be p-hacking and is explicitly *not* done here. Archived algorithms:
`artifacts/agent_authored_bev_v1.py` (run 1).

---

## 8. Honest scope (stated, not hidden)

- **Small-data regime.** nuScenes **mini** = 10 scenes total. The from-scratch 0.10 / pretrained
  0.17 IoU are well below full-nuScenes LSS (~0.32 over 28 k samples). The gap is data quantity,
  not a harness flaw — the claim here is *the harness generalizes*, demonstrated end-to-end, **not**
  a SOTA BEV number.
- **Vehicle class only** — the standard LSS sub-task; other classes are future work.
- **From-scratch backbones** in the sandbox (no pretrained download) — a real constraint that
  lowers absolute IoU but keeps the comparison honest and self-contained.

---

## 9. Files & reproduction

```
scripts/prep_nuscenes_bev.py            harness-owned nuScenes→BEV adapter (held-out split + GT)
scripts/bev_lss.py                      pretrained reference + variance trainer (bar 0.169±0.002)
scripts/grade_bev.py                    standalone IoU grader
scripts/viz_bev_pred.py                 surround→BEV pred-vs-GT (model-based)
scripts/viz_bev_from_preds.py           same, from saved agent masks (architecture-agnostic)
vo_lab/plugins/bev_nuscenes.py          Track-B provider (train / test_input / held-out GT)
vo_lab/plugins/vo_ref/eval_bev.py       harness-owned IoU grader (anti-tamper)
vo_lab/plugins/vo_ref/run_bev_learned.py from-scratch Lift-Splat reference (sandbox bar)
vo_lab/agents/bev_implementer.py        task spec + reference/degenerate authors
vo_lab/run_bev_calibration.py           non-billed gate (sets the bar)
vo_lab/run_bev_implement.py             live billed Track B
tests/test_bev_implementer.py           2 anti-tamper tests (passing)
artifacts/bev/                          GT check, held-out predictions, reference checkpoints
```

```bash
# one-time: prep data (in vo-bev:1)
python scripts/prep_nuscenes_bev.py <nuscenes_root> ~/.cache/vo_lab/bev
# non-billed: confirm the gate + derive the bar
python -m vo_lab.run_bev_calibration
# live (billed + Docker + GPU): the agent authors a BEV network
ANTHROPIC_API_KEY=... python -m vo_lab.run_bev_implement 0.08
```

**Verdict:** the verification-first harness transfers cleanly from ego-motion to multi-view
perception. Every part of the discipline — held-out data, harness-owned GT + metric, anti-tamper
grading, a calibrated oracle — was rebuilt for an unrelated task and works. The agent-authoring
result is honest and two-sided: a sandboxed agent **can** author real BEV perception (2 of 3 runs
cleared the held-out bar), but **not reliably** (n=3 IoU 0.085 ± 0.034; one run failed), and the
diagnostic shows the variance is the agent's redesign latitude, not the task (a fixed recipe is
stable at 0.141 ± 0.002). The most valuable thing here is that **the harness caught that** — a
single run would have over-claimed a clean win. LenaLab spans five domains (monocular VO, RGB-D VO,
SLAM, KITTI stereo, BEV perception); the thesis holds across all five, and on BEV it did its
hardest job: it kept an honest result honest. "It ran" — and even "it passed once" — is never the
same as "it's robust."
