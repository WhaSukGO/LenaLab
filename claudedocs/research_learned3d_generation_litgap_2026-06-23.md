# Research: Learned-Queried-3D for Camera-Consistent Generation — Literature-Gap Analysis

*Research report · 2026-06-23 · serious-research direction-finding for the hypothesis: "a learned 3D
representation, **queried via attention (QKV)**, geometry **learned end-to-end** (not handed in), driving
**camera-consistent generation."** Deep reads of the nearest priors + a landscape map. Report only — no implementation.*

## Executive summary

The hypothesis sits in **genuinely thin space**, and there is now a **concrete baseline to beat with a
precise delta**. The closest prior — **Latent Spatial Memory / "Mirage"** (Microsoft, arXiv 2606.09828,
June 2026) — builds a *persistent 3D **latent** point memory* and camera-consistent generation, **but**
(a) its geometry is **handed in** (frozen DepthAnything-3, explicitly swappable), and (b) it **queries by
render-then-condition** (project the cache to the target view → ControlNet side-branch), **not by QKV**.
So Mirage delivers the *memory substrate and the goal* (~35–45% of the hypothesis) while leaving the two
**load-bearing claims wide open**: *geometry learned end-to-end by analysis-by-synthesis* and *a transformer
querying the 3D store via attention*. No surveyed work occupies the intersection {learned-by-analysis-by-
synthesis geometry × QKV-into-a-held-3D-volume × measured by a geometric-consistency metric} — **that is
the contribution zone.** It's feasible for an independent researcher (Mirage's code is open; a single-GPU
proof-of-concept is viable with a smaller backbone), and driving data makes the cleanest *measurement*
setting (held-out LiDAR/pose as the verifier — which also ties to the user's AD career + verification-first ethos).

## Landscape map — who does what (5 axes)

| Work | Geometry source | 3D representation | Query mechanism | Domain |
|---|---|---|---|---|
| **Mirage / Latent Spatial Memory** (2606.09828) | **given** (DepthAnything-3, frozen) | persistent **latent** point cloud (f∈ℝ⁴⁸) | **render-then-condition** (warp→ControlNet) | indoor/static (RE10K) |
| **Gen3R** (2601.04090) | learned **feed-forward** (adapter on VGGT tokens) | geometric *latents* + point clouds | latent **alignment/concat** | open scene |
| **GEN3C** (CVPR'25) | **given** (DepthAnythingV2 + DROID) | explicit **point cache** | **render-then-condition** | open scene + Waymo |
| **Captain Safari** (2511.22815) | pose/depth-**assisted** | pose-aligned **feature memory** | **attention / QKV** ← rare | open + driving + object |
| **Geometry Forcing** (2507.07982) | learned (alignment to geo-foundation) | implicit/latent (no volume) | **none** (regularizer only) | world-model |
| **MagicDrive / -V2** | **given** (boxes/BEV/pose) | none held | cross-attn over **given conditions** | driving |
| **WoVoGen / DiST-4D / WorldSplat** | given / predicted (occ / metric depth / Gaussians) | explicit (voxel/depth/Gaussians) | **render-then-condition** | driving |
| **GAIA-2/3, OccWorld** | given (ego/agents/occ-GT) | latent / occ tokens | token conditioning / autoregressive | driving |

**Eval primitives that barely exist yet:** **GeCo** (2512.22274) and VGGRPO (2603.26599) — differentiable
geometric-consistency metrics (flow/depth/pose) usable as metric *and* training signal. Most generation
papers still report only FID/FVD + a downstream task, not direct 3D-query consistency.

## The white space (the empty cell)
Three observations pin the gap:
1. **Learned-geometry × QKV-into-a-held-3D-volume is nearly empty.** Everyone who *holds* an explicit 3D
   store **renders** from it (WoVoGen, DiST-4D, WorldSplat, GEN3C, Mirage); everyone who *attends* does so
   over **given** conditions with **no held 3D** (MagicDrive, GAIA). **Captain Safari** is the lone occupant
   of "attention-into-3D" — but its geometry is pose/depth-assisted, not end-to-end.
2. **End-to-end-learned geometry via analysis-by-synthesis** lives only in open-scene/object work; in
   driving everyone hands in at least one geometric signal (LiDAR-depth, occupancy GT, boxes).
3. **The diagonal "learned-3D × attention-queried × scored by a geometric-consistency metric" is
   essentially unoccupied.**

## Recommended research question (#1)
> **Can a persistent 3D *latent* memory whose geometry is learned end-to-end by an analysis-by-synthesis
> consistency objective — and which is *queried by cross-attention (QKV)* rather than rendered-and-
> conditioned — produce camera-consistent generation that is *more robust to geometry error* than the
> render-then-condition + frozen-depth approach (Mirage)?**

- **Why it's open:** Mirage leaves exactly these two gaps (its own stated weakness is that *inherited depth
  errors propagate uncorrected*); Captain Safari has QKV but not end-to-end geometry; Gen3R learns geometry
  but feed-forward + latent-alignment, not queried. The intersection is unoccupied.
- **The precise delta vs Mirage (the baseline to beat):** keep Mirage's setup (open code, Wan-class
  backbone, RE10K), change **two things** — (1) replace the render-then-ControlNet query with a **learned
  QKV attention into the latent point set**; (2) add a **self-rendering / analysis-by-synthesis consistency
  loss** so geometry is *corrected by* the generation objective instead of inherited frozen. **Falsifiable
  claim:** this recovers on depth-error cases that break Mirage's frozen-prior pipeline (measurable).
- **Evaluation:** a **GeCo/VGGRPO-style geometric-consistency metric** as the verifier (multi-view + temporal
  consistency, camera-control fidelity), plus Mirage's WorldScore/RE10K-NVS for direct comparison. *(This is
  a natural fit for the user's verification-first, deterministic-grading instinct.)*
- **Feasibility (honest):** Mirage used **32 A100** for the *easier* (geometry-given) problem. A single
  H100 + the 1000-hr grant forces a **smaller backbone (1–2B / LoRA), a tight domain, and an overfit-then-
  generalize proof-of-concept** — not a full system. The cleanest minimal wedge is the two-change ablation
  above against the open Mirage baseline, which is a real, publishable result even at small scale.

## Alternative questions (ranked)
2. **Attention-query vs render-then-condition at equal compute, on geometric consistency (GeCo).** No
   head-to-head exists — the two camps never share a metric. A clean controlled A/B (both heads on one
   backbone). *High feasibility; ideal deterministic-grading study.*
3. **Geometry-supervision ablation curve:** how much geometric GT can be *removed* (given→learned) before
   learned-queried-3D consistency collapses, in driving? Nobody has mapped this frontier. *High feasibility
   (pure ablation sweep).*
4. **GeCo-as-training-signal:** close the loop — use differentiable geometric consistency as a *loss*, not
   just eval, for the learned-query path (watch for reward-hacking). *Medium-high.*

## Driving as the measurement setting (ties to the career)
Driving is a **cleaner *measurement* setting but a contaminated *purity* setting** — and that tension is
itself the design discipline. Pros: ground-truth poses/LiDAR/ego-motion give an **oracle to *score* learned
geometry against**, and a constrained motion manifold makes "did the 3D stay consistent?" cheaply
verifiable (perfect for a verification-first harness). The discipline: to honestly test "geometry *learned*
end-to-end," you **withhold LiDAR/pose at training and use it only at eval** — abundant GT becomes your
*verifier, not your crutch*. This is a sharper experimental contract than open-scene (no GT) or object (too
easy), and it doubles as on-domain AD-perception signal.

## Concrete first moves (cheap → before any big compute)
1. **Read the three closest priors at code level:** Mirage (`github.com/microsoft/LatentSpatialMemory`),
   Captain Safari (2511.22815), Geometry Forcing (2507.07982). Confirm the QKV + analysis-by-synthesis
   intersection is truly open and write the one-paragraph delta.
2. **Reproduce the Mirage baseline** on RE10K (open code/weights) — your control arm and your harness.
3. **Stand up the GeCo/VGGRPO consistency metric** as the deterministic verifier (your lab's pattern).
4. **Then** the two-change wedge (QKV query + self-rendering loss) as the minimal experiment.

## Sources
- Latent Spatial Memory / Mirage: https://arxiv.org/abs/2606.09828 · code https://github.com/microsoft/LatentSpatialMemory · project https://microsoft.github.io/LatentSpatialMemory/
- Gen3R https://arxiv.org/abs/2601.04090 (https://xdimlab.github.io/Gen3R/) · GEN3C https://research.nvidia.com/labs/toronto-ai/GEN3C/
- Captain Safari (pose-aligned 3D memory, QKV) https://arxiv.org/pdf/2511.22815 · Geometry Forcing https://arxiv.org/abs/2507.07982
- MagicDrive-V2 https://arxiv.org/abs/2411.13807 · WoVoGen https://arxiv.org/abs/2312.02934 · DiST-4D https://arxiv.org/abs/2503.15208 · WorldSplat https://arxiv.org/html/2509.23402v1
- GAIA-2 https://arxiv.org/abs/2503.20523 · OccWorld https://arxiv.org/abs/2311.16038 · Wonderland https://arxiv.org/abs/2412.12091 · Diff4Splat https://arxiv.org/abs/2511.00503
- GeCo (geometric-consistency eval) https://arxiv.org/html/2512.22274v2 · VGGRPO https://arxiv.org/pdf/2603.26599

*Confidence: HIGH that the {learned-by-analysis-by-synthesis × QKV-queried × consistency-scored} cell is
open and that Mirage is the right baseline (deep-read + landscape both confirm). MEDIUM on whether the
"QKV vs render-condition" delta alone is large enough — the robustness-to-depth-error framing (Mirage's
stated weakness) is what makes it sharp. Compute is the main practical risk: scope to small-backbone PoC.*
