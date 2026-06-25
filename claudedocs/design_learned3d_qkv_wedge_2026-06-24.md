# Design: Learned-3D-Store + QKV — the Open-Seam Research Wedge

*Design spec · 2026-06-24 · `/sc:design` (spec only — no implementation). Targets the empty cell from the
B2 analysis: **a 3D feature store whose geometry is learned end-to-end by analysis-by-synthesis, queried via
QKV, producing images.** Companion to `research_b2_internal3d_mechanisms_2026-06-24.md`.*

## 0. Honest scope (read first)
This is the **research path (path 2)**, not the stylized tool. It produces a *mechanism result*, not a
camera-change app for your animations. It's designed in a domain where **3D ground-truth exists to verify
the claim** (RealEstate10K / driving) — **stylized 2D is explicitly out of scope here** (no GT to verify
learned geometry; it's a downstream *application* only after the mechanism works). Expect a **months-long,
single-GPU proof-of-concept** with real risk it doesn't beat the baseline. If you want the tool, that's the
other spec.

## 1. Requirements & the contribution
- **Claim to test:** a store whose geometry **emerges from the generation objective** (no off-the-shelf
  depth/pose/VGGT) and is **queried by attention** will be **more robust to geometry error / OOD inputs**
  than the field's default (given-geometry + render/warp-then-condition), at **equal compute**.
- **Why it's novel (from the B2 map):** every existing B2-ii method either hands geometry in (GEN3C, Mirage,
  Gen3R, Captain Safari) or doesn't query by attention (all but Captain Safari). **The {learned × QKV} cell
  is empty.** Landing it — even at PoC scale — is the contribution.
- **Base to extend:** **Captain Safari** (open, Apache-2.0; already has QKV-into-a-3D-store) — swap its
  *frozen* StreamVGGT store-builder for a *trainable* one. *(Fallback: Gen3R, or a smaller DiT from scratch.)*

## 2. Architecture

```
 seed/prev frames ──▶ ┌──────────────────────────────┐
 + their cameras      │  STORE-BUILDER (TRAINABLE) ◀── the change
                      │  small encoder → pose-tagged  │   geometry (where tokens sit in 3D) predicted by a
                      │  3D-located LATENT tokens      │   light head; NO external depth/pose supervision
                      └───────────────┬──────────────┘
                                      ▼   write
                      ┌──────────────────────────────┐
                      │  3D LATENT STORE  M={(p_i,f_i)}│  persistent, world-coord latent tokens
                      └───────────────┬──────────────┘
 target camera ──▶ build queries ─────┼── QKV cross-attention (RETRIEVAL, kept from Captain Safari)
 (rays/pose)                          ▼
                      ┌──────────────────────────────┐
                      │  DiT video/image generator    │──▶ generated target view
                      └───────────────┬──────────────┘
                                      ▼
                     analysis-by-synthesis loss (denoise target vs GT frame)
                     ── gradient flows back THROUGH retrieval INTO the store-builder ──▲
                        (so the store's 3D is shaped by "must explain the target view")
```

**Components:**
- **Store-builder (new, trainable):** maps input frame(s)+camera → a set of **pose-tagged latent tokens
  placed in 3D** by a small predicted-geometry head. Replaces Captain Safari's frozen StreamVGGT extraction.
- **Store:** persistent world-coordinate latent tokens (keep Captain Safari's form).
- **Retrieval (kept):** target-camera queries → **QKV cross-attention** into the store → conditioning tokens.
- **Generator (kept, LoRA-tuned):** the DiT, conditioned on retrieved tokens.

## 3. Training signal — analysis-by-synthesis (the heart)
- **Primary:** the **diffusion/denoising loss on the held-out target view** *is* the analysis-by-synthesis
  signal — the model must render the target through the queried store; gradient flows **back through QKV into
  the store-builder**, so geometry is corrected by "explain this view." **No external depth supervision.**
- **Stabilizers (to stop the store collapsing — a real risk):**
  - **multi-view/cycle consistency:** query the same store from ≥2 target cameras, penalize disagreement.
  - **optional weak geometric *init*** (curriculum): warm-start the store-builder from a depth prior, then
    **release it** to be shaped by synthesis. *(Decision §7: pure-emergent vs warm-then-free.)*
  - light entropy/occupancy regularizers on token placement.

## 4. The experiment IS a 2×2 (this isolates the contribution)
| | render/warp-then-condition | **QKV retrieval** |
|---|---|---|
| **geometry GIVEN** (frozen) | GEN3C / Mirage (baselines) | **Captain Safari (baseline)** |
| **geometry LEARNED** (ours) | ablation A (learned store, render-cond) | **★ OURS (learned × QKV)** |
Running all four on one backbone/data/compute **cleanly attributes** any gain to (a) learning geometry,
(b) QKV retrieval, or (c) their interaction. The headline test is **★ vs Captain-Safari** and **★ vs Mirage**.

## 5. Evaluation
- **Robustness stress test (the money metric):** inject **depth error / OOD inputs** (degrade or corrupt the
  geometry signal). Hypothesis: given-geometry baselines degrade; the learned store **recovers** (geometry
  is corrected by synthesis). This is Mirage's stated weakness → the sharp, falsifiable win.
- **Geometric consistency:** **GeCo / VGGRPO** (flow/depth/pose consistency across generated views) — the
  deterministic verifier (fits your verification-first instinct).
- **Standard NVS:** PSNR/SSIM/LPIPS + Mirage's **WorldScore** (camera-control, 3D/photometric consistency)
  for apples-to-apples vs the baseline.

## 6. Data & compute
- **Primary domain:** **RealEstate10K** (matches Mirage → direct baseline comparison; geometry GT for eval).
- **Secondary (optional, AD tie):** **nuScenes/Waymo** with **LiDAR/pose held out at train, used only as the
  eval oracle** ("verifier not crutch").
- **Compute:** single **H100 + ~1000 free hrs**. **LoRA on the DiT + a small trainable store-builder**,
  low-res, RE10K subset, **overfit-then-generalize PoC**. Baselines (Mirage/Captain Safari) reuse open code.

## 7. Phased plan & gates (each gates the next)
- **Phase 0 — baselines + verifier (weeks).** Reproduce **Captain Safari** + **Mirage**; stand up **GeCo**.
  *Gate:* both baselines reproduce within tolerance; GeCo runs. **Also: code-level check that Captain
  Safari's retrieval isn't hard-coupled to StreamVGGT features** (flagged risk — if it is, the swap is bigger).
- **Phase 1 — make the store trainable.** Drop StreamVGGT supervision; train the store-builder under the
  denoising loss (+ stabilizers). *Gate:* learned store **matches** frozen store on in-distribution (doesn't
  collapse).
- **Phase 2 — the 2×2 + stress test.** Run all four cells; inject depth error/OOD. *Gate:* **★ (learned×QKV)
  beats given+render-cond on the robustness stress test** at equal compute. (If not → honest negative; the
  field's "given geometry" default is justified — still a result.)
- **Phase 3 — write up / extend.** Driving setting and/or the stylized *application* only after the mechanism
  is shown.

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| **Store collapses** without geometric grounding (analysis-by-synthesis geometry is hard) | cycle-consistency + weak geometric *init* curriculum, then release; entropy reg |
| Captain Safari retrieval **coupled to StreamVGGT** features | Phase-0 code check; fallback = Gen3R or a small from-scratch DiT |
| Compute (Captain Safari/Mirage are heavy; 32×A100-class baselines) | small backbone + LoRA + low-res + RE10K subset; PoC, not full system |
| **Scooped** (hot area, monthly papers) | keep the delta sharp (robustness-to-bad-geometry framing); move fast on Phase 0–2 |
| Doesn't beat baseline | the 2×2 + stress test yields a publishable *negative* either way |

## 9. Decisions needed (before `/sc:implement`)
1. **Domain:** RE10K only (cleanest baseline match) · + driving (AD tie, held-out oracle) · stylized is *not*
   a research-phase choice. *(Rec: RE10K first.)*
2. **Base:** extend **Captain Safari** (rec — open + already QKV) vs Gen3R vs small-from-scratch DiT.
3. **Geometry grounding:** pure-emergent vs **weak-init-then-free** curriculum. *(Rec: weak-init-then-free —
   lower collapse risk.)*
4. **Ambition:** mechanism PoC (prove the cell is reachable) vs push-to-paper. *(Rec: PoC gate first.)*

## 10. Next step
`/sc:implement` Phase 0: reproduce Captain Safari + Mirage baselines, stand up GeCo, and the code-level
coupling check — *before* building the trainable store. *(Recommended defaults: RE10K · extend Captain Safari
· weak-init-then-free · PoC gate.)*
