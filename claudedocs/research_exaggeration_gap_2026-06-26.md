# Research: the "exaggeration gap" — deformation models for stylized/anime faces

*`/sc:research` · 2026-06-26 · The load-bearing novelty for the learned-deformation-manifold idea: anime
expressions are *exaggerated, non-proportional, non-physical* (mouth-stretch, eye/mouth coupling, squash &
stretch). Can a learned deformation manifold represent that? What breaks, and what works? Report only.*

## Executive summary
The gap is **real and explicitly acknowledged** in the literature — and there is a **clear, convergent
solution direction**: replace *linear* blendshape models with **nonlinear / implicit deformation fields**,
identity↔expression **disentangled**, with **gradient-domain (intrinsic) deformation** for large local warps,
trained on **auto-labeled exaggerated-expression data**. So the exaggeration is a *known hard problem with
working ingredients*, not a blank wall — which de-risks the idea. *(Confidence: high on the diagnosis +
ingredients; medium that they compose cleanly for *2D anime at scale*, which remains open.)*

## Why standard models fail (the gap, with evidence)
- **Linear 3DMMs (Blanz–Vetter) are linear combinations of example faces** → they cannot represent large,
  non-uniform, *non-physical* local deformation. Models "**can hardly converge**" on exaggerated expressions
  like **mouth-stretch** (stated directly in the ImFace and cartoon-blendshape work).
- **Caricature has no canonical parametric model** — "3D caricature faces do not have parameterized morphable
  models because styles vary by artist." So you can't just fit a fixed rig; the deformation space itself must
  be *learned*.
- **Data scarcity**: exaggerated/non-photoreal expression datasets barely existed (everything was built for
  photoreal humans).

## How the field actually handles exaggeration (the reusable mechanisms)
1. **Implicit, nonlinear morphable models — ImFace / ImFace++** (CVPR'22 / TPAMI): a continuous implicit
   space with **two explicitly disentangled deformation fields (identity vs expression)** + **flexible
   topology** + an extended-expression learning strategy → **handles mouth-stretch-class exaggerations linear
   models can't.** *This is the core mechanism to adopt.*
2. **Intrinsic / gradient-domain deformation — "Alive Caricature 2D→3D"**: represents deformation as **local
   deformation gradients**, purpose-built for **non-uniform, large local** warps (exactly anime's eye/mouth
   coupling). The right *math* for exaggeration.
3. **Curvature- & semantic-driven exaggeration control — CaricatureGS** (Jan 2026, curvature-driven
   deformation on rigged 3DGS, **continuous intensity control**) and **"Learning to Caricature via Semantic
   Shape Transform"** (parsing-map-driven exaggeration that *retains identity*).
4. **Stylized AU control — AU-Blendshape** (2507.12001): fine-grained *stylized* 3D expression manipulation
   via action units → a controllable, style-aware expression axis.
5. **Identity↔expression disentanglement at the deformation level** — geometry-VAE latent spaces +
   disentangled speech/expression blendshapes (2510.25234) + sparsity-constraint disentanglement.

## Data — the scarcity is now solvable (auto-labeling)
The blocker (no exaggerated-expression data) is being broken by **auto-derived datasets**: a comic→blendshape
pipeline used an **Anime Face Detector + landmarks + a multimodal LLM** to convert comic expressions into
**>10,000** stylized blendshape data points automatically, *including exaggerated ones*. Plus
**Parsing-Conditioned Anime Translation** (a dataset+method for anime faces). So the realistic data path:
**auto-label exaggerated anime expressions at scale** rather than hand-rig — and a *single popular character*
already has thousands of frames across expressions/angles to mine.

## Implication for our PoC (what the research dictates)
- **Do NOT use a linear blendshape rig.** Use a **learned nonlinear/implicit deformation field** (ImFace-style)
  for the expression+exaggeration axis, **disentangled from identity**, optionally with **gradient-domain**
  warps for big local deformation.
- **Source the exaggeration manifold from auto-labeled anime expression data** (comic→blendshape-style
  pipeline / one character's frames), not hand authoring.
- Keep **viewpoint as a separate learned axis** (it's a *drawing convention*, not metric 3D — §prior report).
- **Evaluate on identity + perceptual + multi-view consistency (MEt3R)** and an *expression-fidelity* metric,
  never denoising loss.

## Confidence & open risks
- **High confidence:** the diagnosis (linear fails), and that implicit-nonlinear + gradient-domain +
  disentanglement + auto-labeled data are the right ingredients.
- **Open / medium:** these are demonstrated mostly on *realistic or 3D-cartoon* faces; **a unified
  *2D-anime* deformation manifold that (a) handles exaggeration, (b) disentangles identity, (c) adds a learned
  viewpoint axis, (d) drives generation from one image** is **not solved** — that's the genuine contribution
  (and the PoC's risk).

## Sources
- [ImFace (CVPR'22)](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_ImFace_A_Nonlinear_3D_Morphable_Face_Model_With_Implicit_Neural_CVPR_2022_paper.pdf) · [ImFace++ (arXiv 2312.04028)](https://arxiv.org/abs/2312.04028)
- [Alive Caricature 2D→3D (arXiv 1803.06802)](https://ar5iv.labs.arxiv.org/html/1803.06802) · [Modeling Caricature Expressions by 3D Blendshape & Dynamic Texture (arXiv 2008.05714)](https://arxiv.org/pdf/2008.05714)
- [CaricatureGS (arXiv 2601.03319)](https://arxiv.org/html/2601.03319) · [Learning to Caricature via Semantic Shape Transform (IJCV)](https://link.springer.com/article/10.1007/s11263-021-01489-1)
- [AU-Blendshape — stylized 3D expression (arXiv 2507.12001)](https://arxiv.org/pdf/2507.12001) · [Disentangled Speech/Expression Blendshapes (arXiv 2510.25234)](https://arxiv.org/html/2510.25234v1)
- [Co-Speech Gesture+Expression for Non-Photoreal 3D Characters (comic→blendshape data, arXiv 2506.16159)](https://arxiv.org/pdf/2506.16159) · [Parsing-Conditioned Anime Translation (ACM TOG)](https://dl.acm.org/doi/10.1145/3585002)
