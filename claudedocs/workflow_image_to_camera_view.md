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

## Stylized (Ghibli / 2D-anime) addendum — different rules apply

**Why stylized is its own case.** A Ghibli/2D frame was **drawn, not photographed** — there is **no 3D
behind it.** So "new camera angle" is **invention, not reconstruction**, and the 3D-reconstruction tools
above (GEN3C, depth estimators) partly *fight the medium* (depth nets are trained on real photos and
misfire on flat-shaded line art). Two consequences:
- **Data:** *natural* multi-view anime data **does not exist** (anime isn't camera-captured). Stylized
  multi-view data only exists **synthetically** — render 3D toon/NPR models (huge free **MMD** libraries,
  Blender toon shading, anime 3D assets) from many known cameras. A few synthetic anime-4D sets exist
  (CharacterShot/Character4D, PAniC-3D) but they're character-level.
- **Exactness is impossible** for hand-drawn art — you get *plausible in-style* new views, not faithful ones.

### The practical pipeline: proxy → control maps → style generator
Geometry as a *guide* (from a rough 3D proxy), **style from a 2D generator.** This sidesteps "no real 3D"
and keeps the flat look. Character-level is the solid case.

1. **Rough 3D proxy of the subject** — image→3D (Hunyuan3D 2.5 / TRELLIS / Tripo free tier). We mostly need
   the *geometry*, so a rough mesh is fine (texture/identity don't have to be perfect). For humanoids, a
   posed mannequin/base-mesh also works.
2. **Render the new camera angle as CONTROL MAPS** — in Blender, place the camera at the target angle
   (top-down, etc.) and render a **depth map** + **normal map** + (for characters) an **OpenPose skeleton**.
   *This is where camera control is EXACT — you literally place the camera on real 3D.*
3. **Generate the new-angle image in-style, conditioned on those maps** — a Ghibli/2D generator
   (SDXL/Flux + Ghibli LoRA, or EasyControl-Ghibli) with **ControlNet** (depth + pose/normal) set to the
   rendered maps, **plus an IP-Adapter / reference-image** from the *original* image to carry identity,
   palette, and design. Output = a flat Ghibli image of the same character at the new angle.
4. **Polish (optional)** — img2img/inpaint cleanup, or a final pass through Midjourney/nano-banana using the
   original as a character reference to lock style + consistency.

**Concrete tool stack:** Tripo/Hunyuan3D (proxy) → Blender (depth/normal/pose render) → ComfyUI with
SDXL-or-Flux + Ghibli LoRA + ControlNet(depth,openpose) + IP-Adapter (identity from original) →
nano-banana/Midjourney (final style polish). The 2D generator never "understands" 3D — it just follows the
maps the proxy produced.

**Whole-scene variant (harder, frontier):** for a full scene you need a *scene* proxy — either build a
coarse 3D scene, or estimate depth on the stylized image (unreliable) and warp it to the new camera
(CamTrol-style), then let the style generator inpaint the disoccluded regions in-style. Degrades at extreme
angles; character-level is much more reliable.

### Cheapest stylized first test (this week, mostly free)
1. Pick one Ghibli-style character image (fuller shot, not a face crop).
2. **Tripo free tier** → rough 3D mesh.
3. **Blender** → place camera at the new angle → render **depth** + (if humanoid) **OpenPose**.
4. **ComfyUI** (or a HF Space): SDXL/Flux + a Ghibli LoRA + **depth ControlNet** + **IP-Adapter** (original
   as reference) → generate.
5. **Judge:** does it hold *style* AND *identity* at the new angle? That tells you if the proxy→ControlNet
   route clears your bar before any heavier setup.

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
- *(stylized)* EasyControl-Ghibli https://huggingface.co/spaces/jamesliu1217/EasyControl_Ghibli · CamTrol (training-free camera control) https://arxiv.org/pdf/2404.02101 · CharacterShot/Character4D https://www.emergentmind.com/topics/avatar-anime-character-dataset · PAniC-3D https://www.researchgate.net/publication/373317780_PAniC-3D_Stylized_Single-view_3D_Reconstruction_from_Portraits_of_Anime_Characters · ArtNeRF https://arxiv.org/pdf/2404.13711
