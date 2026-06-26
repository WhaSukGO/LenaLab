# PoC: single-image → moderate 3/4 view (practical path) — WORKS

*2026-06-26 · proof-of-concept after the research wedge was parked (B+C). Input: a Ghibli-style boy,
front-facing close-up (`image.png`). Target: moderate 3/4, keep style + identity. Top-down was out of scope.*

## Verdict: VIABLE via NVS→restyle (the opposite outcome of the research wedge)
The practical pipeline produced a **believable ~25–40° 3/4 turn of the same character, on-style** — the thing
the learned-store wedge could not do. Images in `artifacts/imgpose_poc/` (`COMPARE_A_main.png`,
`05_restyle_t21clean.png`).

## What worked / what didn't (3 approaches tried)
- **A) Novel-view diffusion → anime restyle — WORKS (the winner).** rembg cutout → **Zero123++**
  (`sudo-ai/zero123plus-v1.2`) 6-view orbit → pick the ~30° tile → Depth-Anything-V2 → **Animagine-XL-3.1
  img2img + ControlNet-depth + IP-Adapter(source)**. Zero123++ was the *only* thing that actually **rotated
  the head** (and surprisingly well for OOD anime); the anime-SDXL restyle recovered crisp lineart/cel-shading
  + removed the NVS halo. The rotation MUST come from a real NVS model.
- **B) Control-guided restyle without NVS — clean but DOESN'T turn.** img2img+IP-Adapter is a *restyler*, not
  a view-changer — stayed ≤~10° even with strong "3/4" prompts. (Best identity preservation, though.)
- **C) Pure depth reprojection — fails.** Depth-Anything reads the flat anime face as planar → 25° yaw shears
  it into streaks + holes. Single-image geometry is a dead end for big-ish anime turns.

## Honest caveats
- Identity is **"same character," not pixel-locked** — face drifts toward generic anime, expression sterner;
  style is anime/cel but **slightly off from true soft Ghibli**; background is dropped (cutout); some shading
  artifacts. **Good for a keyframe/concept turn; needs per-shot cleanup before real animation.**
- Works ~20–40°; larger turns / top-down stay hallucinated (as predicted for a single front portrait).

## To make it production-useful (next steps if pursued)
1. **Lock identity** — train a small character **LoRA** (needs a few consistent views of this boy; could
   bootstrap from Zero123++ orbit frames) so the face stays the *same* across angles.
2. **Match the Ghibli look better** — a Ghibli-specific style LoRA/checkpoint instead of generic Animagine.
3. **Background + compositing**, and an **orbit-consistent** version (multi-frame) for animation keyframes.

## Env gotcha (recorded)
diffusers 0.38 crashes on the pod's torch 2.4.1 (flash-attn-3 custom-op PEP-604 annotations). Fix:
`diffusers==0.31.0` + `transformers==4.49.0`.
