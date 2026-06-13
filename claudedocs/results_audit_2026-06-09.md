# Results Audit — what needs re-running before the blog is trustworthy

*2026-06-09. Triggered by the retracted sim-faithfulness "flagship." Goal: a list of experiments to
re-run (or re-examine) so every headline number is one we've actually stress-tested.*

---

## The failure modes the retraction taught us (what we audit for)

1. **Degenerate scene** — a near-straight trajectory makes Sim(3) ATE near-trivial (any roughly-straight
   estimate aligns). → *Checked: only the sim-faithfulness city scene was degenerate. All other test
   scenes (synthetic lte/synth/vio ~12–28 m lateral; real KITTI 07/09 full loops) are genuinely curvy —
   their metrics are sound on this axis.*
2. **Single-run / unmeasured variance** — especially **learned** results (random init). This is the **main
   remaining risk.**
3. **Scale-free metric hiding a scale failure** — Sim(3)/shape-only numbers can look good while metric
   scale is wrong (Ep2 was a 40× rescale).
4. **Misleading plots** — raw (un-aligned) trajectories plotted on metric GT look like failure (or success)
   regardless of the number. Every trajectory figure must be **Sim(3)-aligned** before plotting.

---

## Priority 1 — LEARNED results: re-train N times to get variance (the real risk)

These are single training runs with random init; their headline numbers — and especially *small differences
between them* — may be within retraining noise. **Re-train each ≥5× and report mean ± std.**

| # | Trial | Number(s) at risk | Why it matters |
|---|---|---|---|
| **17** | **Fidelity ladder** | rendered **27.1** vs real **27.2** vs +aug **26.9** | **The headline conclusion ("rendered ≈ real; gap is appearance not capacity") rests on these being within noise of each other.** If retraining std is ±several m, the ladder's middle claim is unsupported. *This is the single most important re-run.* |
| **16** | Rung 3 learned VO | synth **0.45**, real-trained **27.2**, sim→real **69.6** | the "beats ref ~7×" and "~150× collapse" need error bars; ordering is probably safe (gaps are large) but the magnitudes are n=1 |
| **8** | Learned VO on GPU | **19.8 m** | already flagged in the blog's own stat note as unmeasured-variance; close the loop |

**Action:** a `--repeat 5` mode on the learned-VO + fidelity-ladder runners; report mean±std; only keep
comparative claims (A<B) that survive the variance. Cost: GPU-heavy (each retrain ~30–60 min) — the
fidelity ladder alone is ~4 rungs × 5 = 20 trainings.

---

## Priority 2 — reproduce the classical / C++ / agent numbers (likely deterministic, but unverified)

The blog's stat note verified **RGB-D (Ep3)** and **KITTI stereo (Ep6)** are deterministic (std=0 over 5×/2×).
The newer classical/C++ results were **not** variance-checked:

| # | Trial | Number | Check |
|---|---|---|---|
| **18b** | SLAM benchmark | classical 4.0/0.9; **C++ VO 55/100 (diverges)** | reproduce ≥3×; C++ VO uses RANSAC-PnP → confirm deterministic or report std; confirm the divergence is real, not a one-off |
| **9** | KITTI BA (M1) | **2.03%** t_err | reproduce; confirm deterministic + the dev seq is long/curvy enough for the segment metric |
| **12** | Contamination probe | **1.20%** (key "capability not recall" claim) | reproduce; this is a load-bearing claim, must be solid |
| **15** | M3 VIO | **3.83%** vs VO-alone 18% / ref 4.2% | reproduce the agent's main.py run; confirm deterministic; note it uses *synthetic* IMU |

---

## Priority 3 — presentation/honesty fixes (re-examine, not necessarily re-run)

| # | Trial | Issue | Fix |
|---|---|---|---|
| 2 | Mono VO v2 | "0.052 m" is **shape-only after a 40× rescale** — scale totally failed | present the scale failure as the headline, not a footnote |
| 1 | Mono VO v1 | Sim(3) mono = shape-only | label shape-only explicitly |
| 5 | SLAM re-run | **in-sample** (dev = held-out = fr1_room) | present as in-sample only, or re-run held-out |
| 18b | DROID-SLAM | city **0.083 m** is on the **degenerate straight scene** + not reproducible | downgrade to "integration works"; drop the accuracy number until a curvy reproducible run exists |
| — | all figures | any **raw-trajectory** plot is misleading | regenerate every trajectory figure **Sim(3)-aligned** + state scale-free where mono |

---

## Already solid (low priority)

- **Ep3 RGB-D (0.033 m SE3)** — metric (SE3, scale observable), deterministic (re-run 5× std=0), curvy scene. Trust: high.
- **Ep6 KITTI stereo (3.53 m SE3)** — metric, deterministic (2× std=0), curvy. Trust: high.
- **Negatives (Ep4, 10, 11, 13, 14, 18a-IMU, sim-faithfulness)** — honest failures; verify only the *baselines* they're measured against (e.g. the 2.81% "floor", the 1.32% "reachable" de-risk) reproduce.

---

## Recommended order

1. **Fidelity-ladder variance (Ep17)** — highest-leverage; decides whether the rendered≈real conclusion holds.
2. **Rung 3 + Ep8 learned variance** — same retraining infra; finishes the learned-results audit.
3. **Reproduce the classical/C++/agent numbers (Ep9, 12, 15, 18b)** — fast (deterministic), high confidence.
4. **Presentation fixes + Sim(3)-aligned figures** — no compute, just correctness.

After 1–4, the blog's every surviving number is either *verified reproducible* or *honestly caveated*, and
the retraction is the only failure of integrity in the record — which is itself the point of the lab.
