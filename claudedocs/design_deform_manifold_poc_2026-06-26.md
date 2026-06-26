# Design: PoC — learned deformation manifold for a stylized character (single-character first)

*`/sc:design` · 2026-06-26 · Turns the learned-deformation-manifold idea + the exaggeration-gap research into
a concrete, cheaply-gated PoC. Architecture/plan only — implementation is the next step (`/sc:implement`).*

## Goal & measurable success
**From one image of a (held-out) anime character → render it at a new viewpoint AND/OR a new (possibly
exaggerated) expression, on-model, in 2D style.** PoC succeeds if, on held-out targets, it **beats the plain
NVS→restyle baseline** (our committed `artifacts/imgpose_poc/`) on: **identity** (a character-feature
similarity score), **multi-view consistency** (MEt3R), **expression fidelity** (does the intended expression
appear), and **perceptual** (LPIPS/SSIM vs held-out real frame). **Never denoising loss** (the mistake we
already paid for, twice).

## Scope decision — single character FIRST (not "millions")
The dream is a *generalizable, identity-disentangled* manifold across many characters. The PoC de-risks the
*load-bearing* claims on **one well-covered character** before any scale: a single popular character (e.g. a
series protagonist) already has **thousands of frames** across expressions/angles to mine. Identity transfer
(the "from one image of a *new* character" part) is the **last** gate, only after the manifold itself works.

## Architecture (chosen to fit anime, per the research)
Key research-driven choice: **NOT rigid 3D** (it fights anime — we saw it fail) and **NOT a linear
blendshape rig** (can't represent exaggeration). Instead a **2.5D canonical + learned nonlinear deformation
field + diffusion completer**:

1. **Identity code `z_id`** — a per-character latent (a canonical character representation: a feature map /
   stylized canonical view). For the single-character stage `z_id` is fixed; later it's *fit from one image*.
2. **Deformation manifold (the novel core)** — a **nonlinear implicit deformation field** (ImFace-style)
   split into two **disentangled** latent axes:
   - `z_expr` — expression/exaggeration (the manifold that knows "how high the eyebrow goes," eye/mouth
     coupling); use a **gradient-domain / intrinsic** warp formulation (Alive-Caricature-style) so large
     non-uniform local deformation is representable.
   - `z_view` — viewpoint as a **learned 2D drawing-convention axis** (not a metric camera).
   The field maps `(z_id, z_expr, z_view)` → a dense warp (+ visibility) of the canonical.
3. **Diffusion completer / restyle head** — warping reveals disocclusions (unseen side/back); a stylized
   anime diffusion decoder (SDXL/FLUX, like the PoC's restyle) **fills + cleans** to a hand-drawn-quality 2D
   frame, conditioned on `z_id` (identity-locked) + the warped guide (droppable geometry context,
   Geometry-as-context-style).
- *Backbone option to stand on:* SEGA / AniArtAvatar (single-image drivable stylized avatar) for the canonical
  + completer, with our deformation manifold replacing their (realistic) expression model.

## Data plan (auto-labeled, per the research)
- **Stage-0 mining:** collect one character's frames; **auto-label** with Anime Face Detector + landmarks +
  an MLLM expression tagger + a coarse viewpoint tag (front/3-4/profile/…) — the comic→blendshape recipe that
  produced 10k+ stylized expression points automatically. No hand rigging.

## De-risking order (each a cheap go/no-go gate — STOP if a gate fails)
1. **Gate 1 — manifold reconstructs + interpolates (cheapest).** Train the deformation field on one
   character; can it reconstruct held-out frames and *interpolate* expression/viewpoint smoothly (incl. an
   exaggerated expression linear models can't)? *Validates the core claim that a learned manifold captures
   anime deformation.* ~1–2 pod sessions.
2. **Gate 2 — drive to held-out targets.** Given canonical + target `(z_expr, z_view)` → generate the frame;
   beat the NVS→restyle baseline on identity + MEt3R + expression fidelity on held-out frames.
3. **Gate 3 — exaggeration specifically.** Does it produce *on-model exaggerated* expressions (mouth-stretch,
   big-eyes) the rigid/linear baselines can't? (The actual novelty.)
4. **Gate 4 — single-image identity transfer (hardest, last).** Add a few characters, disentangle `z_id`;
   from **one held-out image** fit `z_id`, drive with the shared manifold → new angle/expression on-model.

## Evaluation harness (build once, reuse every gate)
- Identity score (character-feature embedding sim), **MEt3R** (multi-view consistency), expression-fidelity
  (tag/landmark match), LPIPS/SSIM vs held-out reals, + side-by-side image panels. **No denoising loss.**

## Compute & cost
- Pod (H100 fine; inference + medium training). Bulk cost = data mining + manifold training. **Gate 1 is
  small** — that's the point: spend little to learn if the core idea holds before scaling.

## Honest risks (carried from research)
- Anime *viewpoint* is a drawing convention → this is a learned drawing manifold, **not** 3D reconstruction
  (set expectations: great for expression + moderate turns; large/top-down still hard).
- Disocclusion completion + identity-lock tension is the core engineering risk.
- Identity disentanglement across stylized characters (Gate 4) is the deepest unknown.
- This is a **multi-week research build**, not one session — but **Gate 1 is a cheap, decisive first test.**

## Next step
`/sc:implement` **Gate 1 only**: mine one character's frames + auto-label, train the disentangled nonlinear
deformation field, and test reconstruction + expression/viewpoint interpolation (incl. one exaggerated
expression). Report on pixels + the metrics above → decide whether to proceed to Gate 2.
