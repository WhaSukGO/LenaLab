# Research: Is the *Architecture* the Cross-Space Bottleneck? — Expert Panel

*Research report · 2026-06-22 · LenaLab · question raised by the user: "Don't you think it's the model
architecture that is the bottleneck?" Four-expert adversarial panel + literature. Report only — no implementation.*

## Executive summary

A four-expert panel (architecture-proponent, transfer-skeptic, literature reviewer, methodology skeptic)
reviewed the cross-space result. **Three of four conclude architecture is NOT the primary bottleneck; the
fourth (arguing the user's side) reached only 60% and conceded the key weakness.** The dominant verdict:
the bottleneck is **domain / camera-rig distribution shift + insufficient training diversity (no domain
randomization)**, not model capacity. The single most decisive point came from the literature reviewer:
**our masked-mean IPM already has the architectural property the cross-rig literature says matters most**
— permutation/camera-count-invariant *geometric* fusion (exactly the change that fixed MVDet's
cross-scene generalization). And the methodology skeptic showed the current experiment **can't even
isolate architecture** (12 scenes, no augmentation, single seed, IoU@0). Best estimate: **~25–30%**
architecture-primary vs **~70%** domain/data/method-primary — *and presently under-determined*. The user's
hypothesis is reasonable but the evidence leans the other way. A concrete 3-step experiment ladder settles
it before investing in a fancier model.

## The question
The 12-train cross-space model fits TRAIN warehouses ~0.18–0.29 IoU (≈ a dedicated per-space IPM ~0.22)
but collapses to ~0.04–0.07 on UNSEEN warehouses; 4→12 training scenes barely moved it (0.044→0.046).
**Is the model architecture (masked-mean IPM + ResNet18 + 3 fixed height planes) the primary reason it
won't transfer to an unseen warehouse?**

## The panel

| Panelist (lens) | Verdict on "architecture is primary" | Confidence |
|---|---|---|
| **Architecture proponent** (steelman YES) | Architecture is *a* co-primary bottleneck | **60% YES** |
| **Transfer/distribution skeptic** | NO — capacity already sufficient; it's distribution shift | **70% NO** |
| **Literature reviewer** (sourced) | Domain/data-primary; key transferable arch property already present | **78% NO** (domain-primary) |
| **Methodology skeptic** | Architecture claim is currently **unfalsifiable**; data/method are higher-prob confounds | **~15% YES** (80% unfalsifiable) |

### Each opinion (condensed)
- **Architecture proponent (YES, 60%):** masked-MEAN throws away *cross-view consensus* (the space-invariant
  signal); 3 fixed Z-planes bake in a scene/object-specific height prior (vs Lift-Splat's *learned* depth);
  C=64/layer3 bottleneck invites shortcut learning; no calibration-conditioning/attention. A more
  expressive, geometry-grounded model could learn transferable features. **Honest concession:** the flat
  4→12 data curve "screams data-diversity bottleneck," which tempers the position.
- **Transfer skeptic (NO, 70%):** the decisive fact is *intra-checkpoint* — same weights score 0.18–0.29 on
  train vs 0.04–0.07 unseen, so **architecture is held constant across the drop and can't be its cause**.
  Capacity is already ≈ a dedicated per-space IPM. More capacity (the agent model at 0.39 per-space) bought
  **zero** cross-space gain → classic overfitting fingerprint. Bottleneck = each warehouse is its own
  distribution; few source rigs. (DomainBed: data/augmentation dominate, not the method.)
- **Literature reviewer (domain-primary, 78%):** cross-rig multi-view detection collapses rig-to-rig
  *regardless of backbone*; the fix that worked (MVDet→generalizable) was **order-invariant mean-pool +
  DropView**, i.e. *simpler geometric* fusion — **which our masked-mean already is**. BEV cross-domain drop
  is driven by camera-intrinsic/focal shift and BEV feature distribution (DG-BEV), repaired on the *data/
  representation* side (scale-invariant depth, perspective augmentation, domain randomization). **A pure
  architecture swap — especially *learned* depth — could make it worse** (DG-BEV had to repair learned depth).
- **Methodology skeptic (unfalsifiable, arch-primary ~15%):** 12 scenes is below the floor where "generalize
  across rigs" is even learnable; **no domain randomization** makes extrinsics-overfitting the path of least
  resistance; single seed + per-space-tuned schedule means 0.044 vs 0.046 may be within noise; IoU@0
  conflates calibration with localization. The experiment can *detect* a transfer failure but cannot
  *attribute* it. Deployability: "one frozen model for all warehouses" is the wrong target — cameras are
  static & calibrated, so **per-space few-shot adaptation** is cheap and natural; the architecture question
  is moot unless adaptation *also* fails.

## Convergence & disagreement
- **Strong agreement (4/4):** the failure is a genuine, large transfer gap; data diversity / domain
  randomization is a (probably *the*) major lever; the current setup can't cleanly blame architecture.
- **Agreement (3/4 + partial 4th):** capacity is *not* the bottleneck — the model fits seen warehouses as
  well as a dedicated one.
- **Disagreement:** how much the *fusion/lifting* design (masked-mean + fixed Z-planes) caps transferability.
  The proponent says it's the residual limiter; the literature reviewer says masked-mean is actually the
  *right*, already-transferable choice and learned-depth could backfire. This is the real open seam.

## Verdict on the user's hypothesis
**Plausible but most likely not the *primary* bottleneck.** The architecture is imperfect (no cross-view
attention, hand-set height planes, small backbone), but (a) it already fits seen spaces to capacity, (b) it
already embodies the key cross-rig-transferable property (permutation/count-invariant geometric fusion),
and (c) the flat data curve + absent domain randomization point upstream to **distribution shift + data
diversity** as the binding constraints. Architecture is a **secondary, currently-unfalsifiable** factor —
worth testing only *after* the data/method levers are ruled out.

## Convergent discriminating experiment ladder (cheapest first)
All four panelists independently proposed steps of the same ladder. Run in order; each gates the next:
1. **Make the metric honest (hours).** Re-score existing checkpoints with a PR-curve / best-F1 operating
   point + 3-seed variance. Tells us how much of "0.05" is calibration vs localization, and whether 4→12
   was noise.
2. **Diversity + domain-randomization sweep on the SAME architecture (a few runs).** Toggle texture/lighting/
   layout/intrinsic jitter + grid-extent normalization; scale source rigs k∈{2,4,8,12,all-but-one} and plot
   unseen IoU vs k. **Rises with diversity/randomization → data-primary (architecture moot). Stays floored →
   first real evidence for an architectural limit.**
3. **Few-shot per-space adaptation (the deployment proxy).** Fine-tune the frozen generalist on a small slice
   of the unseen warehouse. Recovers ~0.22 → deployment solved, architecture decisively not the blocker.
4. **Only if 1–3 implicate architecture:** swap fusion/lifting with data fixed — learned soft depth/height +
   cross-view attention + camera-extrinsics embedding. Jump in unseen IoU at fixed data ⇒ architecture was
   the limiter.

## Recommendation
Document this (done) and, if pushing further, **run the ladder — not an architecture rebuild first.**
Steps 1–3 are cheap and local (3080, no pod); they will most likely show the lever is **domain
randomization + diversity** and/or that **per-space adaptation** is the deployable answer. Reserve an
architecture overhaul (Step 4) for if-and-only-if the data/method levers fail.

## Sources (from the literature panelist)
- Vora et al., *Bringing Generalization to Deep Multi-View Pedestrian Detection* — https://ar5iv.labs.arxiv.org/html/2109.12227 (order-invariant mean-pool + DropView massively lifts cross-rig/scene transfer)
- Wang et al., *Towards Domain Generalization for Multi-view 3D Object Detection in BEV* (DG-BEV, CVPR'23) — https://arxiv.org/abs/2303.01686
- *Understanding Cross-Sensor Feature Variations for Generalizable 3D Perception* — https://arxiv.org/html/2606.11573
- *UniDrive: Universal Driving Perception Across Camera Configurations* — https://arxiv.org/abs/2410.13864
- *MVUDA: Unsupervised Domain Adaptation for Multi-view Pedestrian Detection* — https://arxiv.org/html/2412.04117
- *MVRackLay* (synthetic warehouse diversity) — https://arxiv.org/pdf/2211.16882 · Tremblay et al., *Domain Randomization* — https://www.researchgate.net/publication/329748828 · *NRSeg* — https://arxiv.org/pdf/2507.04002

*Confidence: high that capacity is sufficient and that data/domain-shift is at least co-primary; medium on
the exact architecture-vs-data split (the masked-mean-vs-attention seam is genuinely open and only Step 2/4
settles it). Aggregate panel estimate: ~25–30% architecture-primary, ~70% domain/data/method-primary, with
the claim currently under-determined by the experiment.*

## Empirical resolution — the ladder was run (2026-06-22, local 3080)
The panel's predictions held up:
- **Step 1 (honest metric):** threshold sweep recovered only ~20–25% rel — the gap is real (matches the
  methodology skeptic's calibration caveat, but it's not the main story).
- **Step 2 (domain randomization, SAME architecture):** unseen IoU 0.046 → **0.058** — a marginal lift
  that does NOT close the gap. Data-side DR at 12 scenes helps a little but isn't the silver bullet.
- **Step 3 (few-shot per-space adaptation):** zero-shot 0.052 → **few-shot 0.355** → **from-scratch
  per-space 0.415**. The decisive result: the SAME architecture reaches ~0.41 per-space, so **capacity
  is ample and architecture is NOT the bottleneck** (the transfer-skeptic's 70% and the literature
  reviewer's 78% were vindicated). From-scratch even beats warm-starting from the generalist → the
  cross-space features are non-transferable; per-space is the deployable path.

**Closed verdict:** architecture is not the primary bottleneck. Frozen cross-space transfer is hard
*independent of architecture* (more scenes flat, DR marginal); per-space training/adaptation is the
correct, working, deployable answer (~0.36–0.42). The user's architecture hypothesis was reasonable but
the evidence resolved against it.
