# Research: Single-image character → consistent new camera angles (2026)

*`/sc:research` · 2026-06-26 · Question: the goal is a "single shot" — we'll usually have **one image** of a
character. A normal character LoRA needs many images. So how do we get **identity-consistent new camera
angles from a single image** (Ghibli/2D)? Report only — no implementation.*

## Executive summary — the reframe that dissolves the problem
**"Single image" is the *starting point*, not a wall.** The dominant 2026 workflow is:
**one image → bootstrap a *consistent turnaround / multi-view set* → (optionally) train a one-shot character
LoRA on those views → render any angle with locked identity.** You only need one image to *begin*; the first
step manufactures the multi-view data that everything else (LoRA, new angles, animation) needs. This is
exactly what our Zero123++→restyle PoC hinted at — and the research shows there are now **anime-native,
identity-preserving** ways to do that bootstrap far better. *(Confidence: high — multiple independent 2026
sources converge on this pattern.)*

## The four viable routes (by effort, all start from one image)

**1. Fastest — Nano Banana Pro / Gemini 3 Pro Image (your own tool).** Use the **turnaround-first
technique**: prompt it to emit *one image containing front + 45° + 90°* views, which gives the model "a
complete 3D understanding of the head," then reuse that as the multi-view reference for any new angle. Tops
2026 consistency benchmarks (**~93%** character consistency across scenes, ZDNET). *Caveat:* users report
angle changes are still finicky and it "fails to maintain 100% consistency" with the original — expect
retries. *(Confidence: high it's the quickest path; medium on reliability for big angle changes.)*

**2. Open & controllable — FLUX.1 Kontext.** An instruction-edit model that can **"rotate the camera angle,
adjust a face's orientation, generate multiple portrait perspectives from a single reference,"** preserving
identity/style. There's a dedicated **FLUX Kontext Character Turnaround Sheet LoRA**: feed one image → get
front / profile / 3-4 / back in one pass. Slots straight into a ComfyUI pipeline. *(Confidence: high — direct,
documented single-image camera-rotation editing.)*

**3. 3D route — best cross-angle consistency + animation-ready — CharacterGen.** Takes **a single
(anime/stylized) image → a canonical-pose 3D mesh** (trained on 13.7k anime characters), which you can then
render from **any** camera angle consistently *and* rig/animate. This is "internal 3D done right" — a *means*
to produce consistent 2D frames, not a 3D-modeling end-goal. Best when you want **full** angle freedom or
real animation. *Caveat:* output is a mesh with finite fidelity/stylization; generalization beyond its anime
dataset is uncertain. *(Confidence: high it works for anime single-image→3D; medium on final 2D-frame
beauty vs hand-drawn.)*

**4. Production identity-lock — bootstrap a one-shot LoRA.** Single image → generate a consistent turnaround
(via route 1/2/3) → **hand-pick the good views → train a character LoRA on them.** The turnaround poses "make
a good starting dataset for training a character LoRA." For maximum lock, 2026 studios stack **low-strength
LoRA (~0.6) + PuLID adapter (~0.8) + ControlNet (OpenPose)** to keep the face identical frame-to-frame. This
**solves the single-image-LoRA bias** (training on one image overfits that one view) by first bootstrapping a
multi-view set. *(Confidence: high — this is the documented "make it production" recipe.)*

## What does NOT fit our case (honest, saves dead ends)
- **InstantID / PhotoMaker / IP-Adapter-FaceID:** strong *single-image* identity, but **realistic-face only**
  (InsightFace AntelopeV2 face embeddings) → not for anime/full-body/Ghibli. (IP-Adapter we already found is
  a *restyler*, not a view-changer.)
- **Pure depth-warp** (our PoC Approach C): fails on flat anime faces (sheared streaks).
- **Training a LoRA directly on the one image:** "highly biased toward each characteristic of the picture" —
  overfits the single viewpoint. Bootstrap first.

## How this connects to our own PoC
Our Zero123++ → anime-SDXL-restyle result (~25–40° turn, recognizable boy) was a **crude instance of routes
2–3**. The research says the upgrade path is clear: swap the generic NVS+restyle for an **anime-native
bootstrapper** (CharacterGen for 3D, or FLUX Kontext / Nano Banana Pro turnaround for 2D), then **lock
identity with a bootstrapped one-shot LoRA**.

## Honest state of the art (2026)
Consistency is **~85–93%, not 100%**; expect to **generate 10–20 and hand-pick**; identity from one image is
never perfect; "fully consistent characters" is widely expected to be **solved ~2028**. So the realistic
near-term deliverable is *"same character, on-model, with light per-shot cleanup"* — not pixel-locked.

## Recommendations (for your decision — no build yet)
A decision tree by what you value:
- **Want results today, minimal setup →** Nano Banana Pro **turnaround-first** (route 1).
- **Want an open, repeatable pipeline you control →** FLUX.1 Kontext + turnaround LoRA (route 2), then
  optionally bootstrap a one-shot LoRA (route 4).
- **Want full angle freedom / actual animation →** CharacterGen single-image→3D (route 3).
- **The robust general recipe:** *one image → bootstrap turnaround → one-shot character LoRA (PuLID +
  ControlNet) → render any angle.* That's the path that turns "a neat 3/4 demo" into a reusable on-model
  character.

## Sources
- [CharacterGen (project)](https://charactergen.github.io/) · [paper](https://arxiv.org/abs/2402.17214)
- [FLUX.1 Kontext (Black Forest Labs)](https://bfl.ai/models/flux-kontext) · [Replicate: edit images with words](https://replicate.com/blog/flux-kontext) · [FLUX Kontext Character Turnaround Sheet LoRA (RunComfy)](https://www.runcomfy.com/comfyui-workflows/flux-kontext-character-turnaround-sheet-lora)
- [Nano Banana Pro / Gemini 3 Pro Image (Google DeepMind)](https://deepmind.google/models/gemini-image/pro/) · [Consistent character sheets guide](https://selfielab.me/blog/nano-banana-pro-consistent-character-sheets-guide-20260216) · [character consistency guide](https://prompting.systems/blog/nano-banana-pro-character-consistency-guide)
- [AI Character Turnaround Sheet Guide 2026 (Apatero)](https://apatero.com/blog/ai-character-turnaround-sheet-generation-guide-2026) · [Best LoRAs for Consistent Characters 2026 (Thinkpeak)](https://thinkpeak.ai/best-loras-consistent-characters-2026/)
- [Single-Image Character Consistency (Scenario)](https://help.scenario.com/articles/5838320337-single-image-character-consistency-ideogram) · [Few-shot multi-token DreamBooth LoRA (arXiv 2510.09475)](https://arxiv.org/pdf/2510.09475)
- [InstantID (arXiv 2401.07519)](https://arxiv.org/html/2401.07519v1) · [See-through: single-image layer decomposition for anime (arXiv 2602.03749)](https://arxiv.org/abs/2602.03749)
- [Zero123++ / multi-view base], [SyncDreamer (arXiv 2309.03453)](https://arxiv.org/pdf/2309.03453)
