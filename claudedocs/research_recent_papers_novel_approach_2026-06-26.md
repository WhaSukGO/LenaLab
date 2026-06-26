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

## Addendum — the stronger idea: a LEARNED DEFORMATION MANIFOLD (supersedes PASV)
User's idea: don't use rigid 3D (which fights anime's non-physical, exaggerated, non-proportional drawing);
instead **learn a dynamic/deformable *latent* manifold** of a character — the *range* of expression/pose/
viewpoint deformation (how high the eyebrow goes, how the mouth coupling warps the eyes) — disentangled from
identity, learned at scale across frames/characters, then used to drive new-angle/new-expression generation
from one image.

**Verdict: the most animation-correct direction so far, and a better-grounded version of the parked wedge's
instinct** (the wedge learned geometry per-clip from the diffusion loss alone → loss artifact; this grounds
the representation in massive observed deformation data + identity disentanglement). It abandons rigid 3D for
a learned deformation manifold that *embraces* anime conventions. Grounding in current work:
- **Unsupervised learned deformation ("skeleton-free range of motion")** = FOMM → MRAA → **MMFA** (Mar 2026),
  Animate-X.
- **Identity-disentangled motion latents** = Disney disentangled-identity-motion, **X-NeMo** (ICLR 2025),
  DeX-Portrait.
- **One image → drivable/view-controllable (stylized) avatar** = **SEGA** (Mar 2026), **AniArtAvatar** (art
  avatar), AniGS, FastGHA.
- **Exaggerated cartoon deformation** = CaricatureGS, cartoonized blendshapes — *and the key gap:* standard
  morphable models "can hardly converge" on exaggerated expressions (mouth-stretch). **That is the open
  nugget the idea targets.**

**Genuine open novelty:** a *generalizable, identity-disentangled, viewpoint+expression deformation manifold
for **stylized 2D** that handles **non-physical exaggeration**, drivable from one image.* Pieces exist
separately; a unified anime one (esp. the exaggeration, at scale) is not solved.

**Honest hard parts:** (1) DATA dominates — but you do NOT need millions of characters to start; learn the
manifold from one character's frames / existing data and transfer to new identity from one image (millions is
the dream-scale generalization). (2) "Viewpoint" in 2D is a learned *drawing convention*, not metric 3D — so
it's a learned drawing manifold (the right framing, but not 3D reconstruction). (3) The exaggeration handling
is the research risk. (4) Scale/compute is a real project; a **single-character PoC** (build on AniArtAvatar/
SEGA + X-NeMo-style disentangled motion) is the feasible first rung.

**Suggested next step:** single-character PoC — take one anime character with abundant frames, build an
identity-disentangled deformation+viewpoint manifold (standing on SEGA/AniArtAvatar + disentangled motion
latents), and test new-angle + new-expression from a held-out single image; eval on identity + MEt3R +
perceptual, never denoising loss.

## Sources (addendum)
- [X-NeMo (ICLR 2025)](https://arxiv.org/html/2507.23143v1) · [MMFA — unsupervised keypoint face animation (arXiv 2603.04302)](https://arxiv.org/abs/2603.04302) · [FOMM (arXiv 2003.00196)](https://ar5iv.labs.arxiv.org/html/2003.00196) · [MRAA (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Siarohin_Motion_Representations_for_Articulated_Animation_CVPR_2021_paper.pdf)
- [SEGA — drivable 3DGS head from single image (arXiv 2504.14373)](https://arxiv.org/html/2504.14373v3) · [AniArtAvatar (arXiv 2403.17631)](https://arxiv.org/html/2403.17631v1) · [Full-head Gaussian avatar from single image (arXiv 2601.12770)](https://arxiv.org/pdf/2601.12770)
- [CaricatureGS — exaggerated 3DGS faces (arXiv 2601.03319)](https://arxiv.org/html/2601.03319) · [3D Gaussian Blendshapes (arXiv 2404.19398)](https://arxiv.org/pdf/2404.19398)

## Sources
- [Diff-PC (arXiv 2602.00639)](https://arxiv.org/html/2602.00639v1) · [Geometry-as-context (arXiv 2602.21929)](https://arxiv.org/abs/2602.21929) · [WorldStereo (arXiv 2603.02049)](https://huggingface.co/papers/2603.02049)
- [Virtually Being (arXiv 2510.14179)](https://arxiv.org/html/2510.14179v1) · [TurboPortrait3D (arXiv 2510.23929)](https://arxiv.org/pdf/2510.23929) · [MEt3R (arXiv 2501.06336)](https://arxiv.org/pdf/2501.06336)
- [CharacterGen (arXiv 2402.17214)](https://arxiv.org/abs/2402.17214) · [StdGEN (arXiv 2411.05738)](https://arxiv.org/html/2411.05738v1) · [DrawingSpinUp (arXiv 2409.08615)](https://arxiv.org/pdf/2409.08615) · [See-through anime layer decomposition (arXiv 2602.03749)](https://arxiv.org/abs/2602.03749)
