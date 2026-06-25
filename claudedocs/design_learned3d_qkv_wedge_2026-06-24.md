# Design: Learned-3D-Store + QKV — the Open-Seam Research Wedge

*Design spec · 2026-06-24 · `/sc:design` (spec only — no implementation). Targets the empty cell from the
B2 analysis: **a 3D feature store whose geometry is learned end-to-end by analysis-by-synthesis, queried via
QKV, producing images.** Companion to `research_b2_internal3d_mechanisms_2026-06-24.md`.*

## 0. Scope (updated — domain locked 2026-06-24)
Research path, now aimed at **your domain via the right data: stylized animation that is *perspective-
correct*** (rendered 3D-CG / toon-shaded / **stop-motion**). That data has **both the stylized look AND
recoverable 3D ground-truth**, so it can **train + verify the learned-geometry claim *and* double as the
application** — dissolving the earlier "stylized has no 3D" blocker (we sidestep *hand-drawn*; we use
stylized data that *does* carry geometry). This is the closure of your original **stop-motion instinct**:
perspective-correct stylized footage is the key. Still a **months-long, single-GPU PoC** with real risk;
**hand-drawn-2D / Ghibli is a deferred transfer stretch**, not the PoC target.

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
*On rendered toon data we have GT depth/pose, so the **given-geometry** arm can use either the GT depth
(upper-bound) or an **off-the-shelf estimator on the toon RGB** — which tends to fail on flat toon shading,
and that failure is exactly the **stress test** where the **learned** arm should win.*

## 5. Evaluation
- **Robustness stress test (the money metric):** inject **depth error / OOD inputs** (degrade or corrupt the
  geometry signal). Hypothesis: given-geometry baselines degrade; the learned store **recovers** (geometry
  is corrected by synthesis). This is Mirage's stated weakness → the sharp, falsifiable win.
- **Geometric consistency:** **GeCo / VGGRPO** (flow/depth/pose consistency across generated views) — the
  deterministic verifier (fits your verification-first instinct).
- **Standard NVS:** PSNR/SSIM/LPIPS + Mirage's **WorldScore** (camera-control, 3D/photometric consistency).
- **Direct geometry check (bonus of rendered data):** because toon data is *rendered*, we have **exact
  depth/pose GT** → measure **learned-geometry accuracy directly**, not just downstream consistency. Cleaner
  verifier than RE10K.

## 6. Data & compute
- **Primary domain — stylized-but-geometric animation:**
  - **Manufactured synthetic toon-3D multi-view (primary):** render 3D toon/anime-shaded characters & scenes
    (MMD model libraries, anime 3D assets, Blender NPR) from **many known cameras** → **perfect camera+depth
    GT**, stylized look, built-in domain randomization, **unlimited quantity**, and **you control coverage
    (incl. extreme top-down)**. Serves as *both* the training set *and* the verifier.
  - **Secondary / transfer tests:** real **3D-CG anime / game-cutscene / VTuber-MMD** (perspective-correct)
    and **stop-motion** (Pat&Mat-style; cameras via SfM) — test realism + the sim→real-stylized gap.
  - **Hand-drawn 2D / Ghibli:** deferred stretch (no 3D → reachable only via transfer from the above).
- **Baselines on the same data:** the 2×2 is self-contained — train all four cells on the toon set (don't
  rely on Mirage's RE10K numbers; its real-video weights won't match toon out of the box).
- **Compute:** single **H100 + ~1000 free hrs**. **LoRA on the DiT (Captain Safari) + a small trainable
  store-builder**, low-res, a curated toon-multiview subset, **overfit-then-generalize PoC**.

## 7. Phased plan & gates (each gates the next)
- **Phase 0 — data + baselines + verifier (weeks).** (a) Build the **toon-multiview data pipeline** (render
  N toon assets × many cameras → frames + GT depth/pose; bake in coverage incl. top-down). (b) Reproduce
  **Captain Safari** + **Mirage**, fine-tuned on the toon set. (c) Stand up **GeCo** + the direct-geometry
  check. (d) **Code-level check that Captain Safari's retrieval isn't hard-coupled to StreamVGGT** (flagged
  risk — if it is, the swap is bigger). *Gate:* toon data renders with GT; baselines train on it; verifiers run.
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

## 9. Decisions — LOCKED (2026-06-24)
1. **Domain:** ✅ **stylized-but-geometric animation** — *synthetic toon-3D multi-view (primary)* + stop-motion
   / 3D-CG (transfer tests). Hand-drawn 2D deferred. *(NOT RE10K — that was a benchmark default; user
   corrected it.)*
2. **Base:** ✅ **extend Captain Safari** (open, already QKV-into-store).
3. **Grounding:** ✅ **weak-init-then-free** — warm-start the store-builder from rendered GT depth, then
   release it to be shaped by synthesis.
4. **Ambition:** ✅ **mechanism PoC first** (prove the empty cell is reachable + beats baselines on the
   stress test), then decide on paper/extension.

## 10. Next step
`/sc:implement` **Phase 0**: (a) the **toon-multiview data pipeline** (render toon 3D from many cameras → GT
depth/pose), (b) reproduce Captain Safari + Mirage on the toon set, (c) GeCo + direct-geometry eval, (d) the
StreamVGGT-coupling code check — *before* building the trainable store.
*Human-in-loop: training runs go on a cloud pod (not the local 3080) per standing preference.*
