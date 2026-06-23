# Workflow: Change the Camera View of a Consistent Character (mid-2026)

*Practical cheat-sheet for an animation creator who has an image and wants the **same character from a
different camera angle** (e.g., close-up face → top-down). Goal = a working workflow now, not training a
model. Companion to `research_learned3d_generation_litgap_2026-06-23.md`.*

## The honest core truth
A true **top-down from a frontal close-up requires information that isn't in the input** (top of head,
shoulders, back). **No tool retrieves it — every tool invents it.** The question is which approach invents
**consistently and controllably.** That single fact decides the winner.

## The three families (and where each breaks)

| Family | Examples | Camera control | Extreme angle (→top-down) | Ease | Cost |
|---|---|---|---|---|---|
| **A. 2D novel-view diffusion** | Stable Virtual Camera (Seva), **GEN3C**, ViewCrafter | explicit paths/presets | **breaks past ~45–60°** — invents unseen content *per-frame*, identity drifts | self-host (H100) | open; GPU time |
| **B. Image→3D→render** ⭐ | **Hunyuan3D 2.5/3.x**, TRELLIS, **Tripo/Rodin (hosted)** | **exact** (render any camera in Blender) | **only family that nails it** — geometrically correct top-down; invents the unseen top-of-head *once*, then it's fixed/editable | hosted UI or H100 | free tier → ~$0.1–0.6/model |
| **C. Commercial video** | Runway Gen-4 (Orbit/Pan/Dolly), Kling 3, Luma | named moves (orbit) | can **orbit**, but **no clean static top-down**; good for *motion* | hosted | subscription |

## ⭐ Recommended workflow — Image → 3D → render
**Why it wins for you:** it's the *only* route that gives **exact, repeatable arbitrary angles with locked
identity** — because once you have a 3D asset, a top-down is just where you put the camera. Bonus for an
animation creator: you also get a **posable 3D asset you can animate and render any camera path**, not just
one reframed still. Open-weights, commercial-friendly, fits your free H100 hours.

1. **Input a fuller character shot, not a tight face crop.** Image-to-3D loses identity from face crops;
   give it a 3/4 or full-body view of the character.
2. **Generate the 3D asset** — Hunyuan3D 2.5/3.x (best for stylized/anime characters; ~16–24 GB VRAM) or
   TRELLIS (MIT, game-ready). Hosted alternative: Tripo / Rodin (no GPU needed).
3. **Render the exact camera** (top-down, worm's-eye, any angle) in Blender.
4. *(Optional) restyle:* pass the render back through Midjourney / nano-banana (Gemini) to restore your
   exact 2D art style — best of both worlds (exact geometry + your look).

**Caveats:** photoreal *faces* survive the 3D trip worst (artifacts, identity loss); **stylized / anime
characters survive best.** Texture/detail of auto-generated 3D is good-not-perfect — the optional restyle
pass fixes most of it.

## Fallbacks
- **Moderate re-angle (≤~45°), stay 2D:** GEN3C or Seva — faster, keeps texture fidelity, no 3D pipeline.
- **You want motion (orbit/fly-around), not a fixed angle:** Runway Gen-4 "Orbit."

## ✅ Cheapest first test (≈$0, under an hour — do this before any GPU)
1. Take one character image (a fuller shot, not a face crop).
2. Run it through **Tripo's free tier (2,000 free credits)** *and* a **Hunyuan3D HuggingFace Space**.
3. Download the mesh, open in **Blender**, render an **exact top-down** and a **worm's-eye**.
4. **Decision:** does it clear your identity/style bar? → If yes, scale up on the H100 (Hunyuan3D 2.5
   full-res/PBR). → If the 3D looks wrong, your input was too close-up; retry with a wider shot. → If even
   a wider shot fails (e.g., photoreal face), fall back to GEN3C for moderate angles + accept the limit.

## Where the research idea reconnects (later, optional)
Your earlier "learned-queried-3D generation" research is exactly the thing that would fix Family A's
**extreme-angle breakage** (a persistent learned 3D memory that accumulates unseen regions consistently).
So: use the tool workflow now; if the extreme-angle limit keeps biting your real shots, *that* is your
sharp, motivated research problem — pick it up via the lit-gap report then.

## Sources
- Stable Virtual Camera https://github.com/Stability-AI/stable-virtual-camera · https://arxiv.org/abs/2503.14489
- GEN3C https://research.nvidia.com/labs/toronto-ai/GEN3C/ · https://huggingface.co/nvidia/GEN3C-Cosmos-7B
- Hunyuan3D 2.5 https://arxiv.org/pdf/2506.16504 · https://github.com/Tencent-Hunyuan/Hunyuan3D-2 · TRELLIS comparisons https://www.3daistudio.com/blog/trellis-2-vs-hunyuan-3d-differences-explained
- Tripo pricing https://www.tripo3d.ai/pricing · API comparison https://www.3daistudio.com/blog/best-3d-model-generation-apis-2026
- image-to-3D identity limits https://arxiv.org/pdf/2603.01328 · Runway Gen-4 camera control https://help.runwayml.com/hc/en-us/articles/34926468947347-Creating-with-Camera-Control-on-Gen-3-Alpha-Turbo
