# Recent papers (2026) + a novel mixed approach: one stylized image → consistent new angles

*`/sc:research` · 2026-06-26 · Survey of the freshest CVPR/arXiv work and a proposed **novel synthesis** for
our exact constraint (single Ghibli image → identity-consistent new camera angles). Idea-level — for your
decision, not an implementation.*

## The most relevant recent papers (and the *one reusable idea* from each)
| Paper (date) | Idea worth stealing |
|---|---|
| **Diff-PC** — Identity-preserving & 3D-aware portrait customization, *zero-shot* (arXiv 2602.00639, Jan 2026) | Render an **explicit 3D proxy at the TARGET camera** as a control signal + **decouple identity** into learned tokens (so identity ≠ the proxy's rough texture). Authors show it extends to **anime/cartoon/pixel-art**. *Face-only (FLAME), the main limit.* |
| **Geometry-as-context** (arXiv 2602.21929, Feb 2026) | Interleave geometry + RGB in one model via **camera-gated attention**, with the **geometry context *droppable*** so the final output is clean RGB (not the proxy look). |
| **WorldStereo** — camera-guided video gen ↔ reconstruction via **3D geometric memories** (arXiv 2603.02049, Mar 2026) | A persistent **3D geometric memory** that keeps generation camera-consistent (scene-level). |
| **Virtually Being** (arXiv 2510.14179, Oct 2025) | Repurpose a 3D reconstruction (4DGS) as a **controllable data engine** to synthesize the multi-view training data a personalization step needs. *They need a capture rig — we won't.* |
| **CharacterGen / StdGEN / DrawingSpinUp / PAniC-3D** | Single **(anime) image → canonical 3D character**, animation-ready — a real, anime-native single-image→3D proxy. |
| **TurboPortrait3D** (2510.23929) · **MEt3R** (2501.06336) | Single-step diffusion **refine** of 3D renders to fix artifacts + keep multi-view consistency · a **metric for multi-view consistency** (our eval, instead of denoising loss). |

**The field's consensus mechanism in 2026:** *explicit 3D for the camera move + diffusion for the look +
decoupled identity + droppable-geometry conditioning.* Notably, **everyone uses geometry explicitly** — which
is exactly why our parked "learned-store-via-QKV" wedge underperformed, and why the PoC's explicit-NVS route
worked.

## The novel synthesis — "**Proxy-Anchored Stylized View Synthesis (PASV)**"
**One stylized image → (a) a rough 3D character proxy → (b) render it at the target camera → (c) a
diffusion *restyle head* turns that geometry-correct-but-ugly render into a clean Ghibli frame, with identity
injected from decoupled character tokens and locked by a one-shot personalization bootstrapped from the
proxy's own multi-view renders.**

Concretely, mixing the ideas above:
1. **Single image → anime 3D proxy** (CharacterGen / DrawingSpinUp-style). This does the *actual* camera
   move with correct geometry — the part pure 2D models can't do (our PoC + verdict C both proved this).
2. **Identity-decoupled tokens** (Diff-PC idea, but anime): identity comes from a learned character-feature
   encoder (DINO/CLIP or a small anime-ID encoder), **not** from the proxy's rough texture — so the proxy can
   be ugly and identity still holds.
3. **Geometry-as-context restyle head** (Geometry-as-context + WorldStereo): feed the proxy render at the
   target camera as **droppable geometric context via camera-gated attention** into a strong anime/Ghibli
   diffusion model (FLUX/SDXL). Output = clean hand-drawn-quality 2D at the right angle; the proxy's CG look
   is dropped.
4. **Bootstrapped one-shot identity lock** (Virtually-Being data-engine, but from *one* image not a rig):
   render the proxy from many cameras → hand-pick → train a tiny character LoRA / fit the identity tokens on
   those → identity stays locked across all angles.
5. **Evaluate on the right thing** (MEt3R + LPIPS + an identity score), **never denoising loss** — the exact
   mistake we already paid for.

## Why this is genuinely novel (vs each ingredient)
- **vs Diff-PC:** a *full anime character* proxy (not a FLAME face) + a true 2D-style restyle head + bootstrapped
  identity lock — not face-only, not photoreal-leaning.
- **vs CharacterGen/StdGEN:** the 3D mesh is **not the product** — it's a *droppable geometry context*; a
  diffusion restyle head produces hand-drawn frames (their mesh renders look CG, not Ghibli).
- **vs Virtually Being:** the multi-view "capture" is **synthesized from one image** via the proxy — no 75-camera
  rig, no per-subject volumetric shoot.
- **vs Geometry-as-context / WorldStereo:** applied to a **personalized stylized character with decoupled
  identity**, geometry sourced from a single-image character proxy (not scene SfM).
- **vs our parked wedge:** uses **explicit** geometry (works) + a strong restyle head, **gated on generation
  quality**. It fixes both failures we found (loss≠pixels; learned-store/base-model ceiling).

## Honest risks
- The single-image anime proxy is **rough** (occluded back/top hallucinated) — fine for ≤~45° turns, weak for
  large/top-down (consistent with everything we've seen).
- **Anime identity decoupling** has no ArcFace equivalent; needs a good character-feature encoder (research risk).
- The restyle head must **fix the proxy's CG look without drifting identity** — the core engineering tension.
- It's a **multi-stage system**, not one clean model. (A longer-horizon "elegant" version would distill the
  whole thing into one model — but stage-wise is the pragmatic first build.)

## Recommended minimal validation (cheapest test of the novel idea)
Before any big build, test the **load-bearing new claim**: *does "anime-3D-proxy render → geometry-context
restyle with decoupled identity" beat the plain NVS→restyle PoC on identity + multi-view consistency?* I.e.,
swap our PoC's Zero123++ for a **CharacterGen proxy** + add **identity-token/LoRA decoupling**, and measure
**MEt3R + an identity score** across a few angles on your boy image. ~1 pod session. If it clearly wins → PASV
is worth building; if not → we've learned cheaply, again on *pixels*, not loss.

## Sources
- [Diff-PC (arXiv 2602.00639)](https://arxiv.org/html/2602.00639v1) · [Geometry-as-context (arXiv 2602.21929)](https://arxiv.org/abs/2602.21929) · [WorldStereo (arXiv 2603.02049)](https://huggingface.co/papers/2603.02049)
- [Virtually Being (arXiv 2510.14179)](https://arxiv.org/html/2510.14179v1) · [TurboPortrait3D (arXiv 2510.23929)](https://arxiv.org/pdf/2510.23929) · [MEt3R (arXiv 2501.06336)](https://arxiv.org/pdf/2501.06336)
- [CharacterGen (arXiv 2402.17214)](https://arxiv.org/abs/2402.17214) · [StdGEN (arXiv 2411.05738)](https://arxiv.org/html/2411.05738v1) · [DrawingSpinUp (arXiv 2409.08615)](https://arxiv.org/pdf/2409.08615) · [See-through anime layer decomposition (arXiv 2602.03749)](https://arxiv.org/abs/2602.03749)
