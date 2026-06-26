# Scope: Generation-Quality Test (the loss-to-pixels gate)

*Design/scope · 2026-06-26 · next-session experiment for the learned-3D+QKV wedge. The generation demo
showed the +16% denoising-loss gain is subtle in pixels and neither generation reproduces the held-out
viewpoint. This test measures **actual generation quality** (not loss) to decide whether the signal is real
enough to justify building the full novel mechanism. Spec only — no implementation.*

## The question
**Does the learned store give a *visible, viewpoint-faithful* improvement over frozen — measured by image
quality vs the real target frame — under *proper* generation (CFG on)?** And, prerequisite: **can the model
reproduce a target viewpoint at all** with proper inference?

## What changed since the demo (fix its weaknesses)
- The demo used **CFG off** + a simplified sampler → likely the main reason neither gen followed the camera.
  **This test uses CFG on + the released model's intended inference recipe.**
- The demo eyeballed pixels. **This test computes standard novel-view metrics** vs the real held-out frame.

## Metrics (vs the REAL target frame at each held-out camera)
Standard NVS metrics — they jointly capture image quality *and* viewpoint match (a gen that matches the real
frame at that camera is, by definition, viewpoint-faithful):
- **LPIPS ↓ (primary)** — perceptual distance (pip install `lpips`).
- **PSNR ↑ , SSIM ↑** — pixel/structural.
- *(optional)* DreamSim / CLIP-image similarity ↑.
- Plus **side-by-side PNGs** (real | frozen-gen | learned-gen) for qualitative viewpoint read.

## Generation settings
- **CFG ON** (sweep guidance ~ {4, 6}) — the released model's standard inference; this is the key fix.
- **~50 denoising steps**; match `validate_lora/...py`'s sampler/scheduler exactly (shift, etc.).
- Same noise seed across frozen vs learned (apples-to-apples); a couple seeds for variance.

## Protocol
1. **Sanity control first:** generate the **canonical query (frame 20)** — the model's *trained* task — with
   CFG on. It should reproduce reasonably. *If even this fails → the generation pipeline/CFG is wrong, fix
   before judging anything.*
2. **In-window viewpoints** (e.g., frames 4, 8, 12 — inside the key window): the model *should* reproduce
   these. Tests "can it hit a viewpoint at all."
3. **Held-out outside-window viewpoints** (16, 18, 19, 20): the hard generalization case.
4. For every viewpoint: generate with **frozen store** and **learned store** (cached `learned_store.pt` — no
   retrain), decode, compute LPIPS/PSNR/SSIM vs the real frame. Report per-viewpoint + means.

## Interpretation framework (decide the path)
- **A — learned clearly beats frozen on LPIPS *and* gens are viewpoint-faithful** → strong go → build the
  *real* novelty (from-frames learned store-builder) + multi-scene.
- **B — learned ≈ frozen perceptually (gain was only in loss)** → the loss signal doesn't translate → the
  trainable-store-copy wedge isn't worth scaling as-is; reconsider mechanism or scope.
- **C — even *frozen* (CFG on) can't reproduce viewpoints** → the released base model's camera control is the
  ceiling, not our store → our store can't fix that → reconsider the base model / approach.
This 3-way outcome is the value: it tells us *which* of build / re-design / pivot is right.

## Compute & staging
- **Cached/staged on `/workspace`** (survives stop): DiT (9.8 GB), VAE, `multi_vp_inputs.pt`, **`learned_store.pt`
  (skip the ~50-min training)**, `crossview.py`/`gen_demo.py` to reuse the loader + sampler.
- **Add:** `pip install lpips` (+ env restore from the lock, ~5 min — resume wipes the pip env).
- **Cost:** CFG doubles forward cost; ~50 steps × ~8 viewpoints × 2 stores × CFG ≈ a moderate run (~30–45
  min GPU). No training needed (store cached).

## Deliverables
- A metrics table: **frozen vs learned — LPIPS/PSNR/SSIM**, per-viewpoint + mean, for in-window & held-out.
- CFG-on side-by-side PNGs (real | frozen | learned) at several viewpoints.
- The **3-way verdict (A/B/C)** with the numbers + an honest visual read → the go/redesign/pivot decision.

## What I need from you (next session)
**Restart the stopped pod** (`lo1q77b9ga90zk`) and paste the SSH — everything's staged on `/workspace`
(weights + cached learned store + scripts), so it's a fast resume (env restore + `lpips` install only). I'll
run the whole test and report the A/B/C verdict + the images. (Or a fresh H100 if it won't resume.)
