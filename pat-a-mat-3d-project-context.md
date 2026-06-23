# Project Context: Implicit 3D / Animatable Character Reconstruction (Pat & Mat)

> **Purpose of this file:** Hand-off context to paste into a new chat session so we can continue
> without losing the thread. It captures the user's background, the idea being explored, the
> research grounding, and the concrete project plan we landed on.

---

## 1. Who the user is / why this matters

- Background: AI researcher specializing in **object detection**, transitioning into a
  **self-driving car engineer** role (mid-2026).
- Learning **BEVFormer** as part of that shift.
- Strong interest in **video generation models**.
- Core problem the user cares about: **"consistent everything, but different camera angle."**
  Prompting with a reference image fails to keep consistency across angles.

## 2. The user's central idea (in their words, paraphrased)

- Start from a single scene, change the camera angle in the next prompt, generate the image.
- The new angle reveals things not visible before → analogous to **building up a 3D space** by
  accumulation.
- If consistency across angles can be reached, it can be used to build consistent video.
- Hypothesis: a transformer could **query a learned 3D space via QKV**, the way BEVFormer /
  occupancy models query a BEV/occupancy representation, and use that to drive generation.

## 3. Key conceptual conclusions reached

- The idea is real and maps onto an active research frontier. Two paradigms:
  - **Paradigm A — explicit geometry handed in, generation learned.** Maintain an explicit 3D
    buffer, reproject it under the new camera, and let the generator only fill unseen regions.
  - **Paradigm B — the geometry/dynamics themselves are learned** (closest to the BEVFormer
    "query a latent 3D volume" intuition).
- **Crucial reframing:** in nearly all these methods, **geometry is *given*, not learned** —
  depth/pose/occupancy/boxes come from off-the-shelf estimators or dataset annotations. What the
  network learns is (a) conditional generation, (b) scene dynamics, and/or (c) cross-view/temporal
  consistency.
- **On "3D is hard to evaluate during training":** you never supervise the 3D directly. You use
  **analysis-by-synthesis / differentiable rendering** — render the candidate 3D back to 2D and
  compare against the frames (color, silhouette, **optical flow**). The 3D is the bottleneck that
  must explain all 2D views. (This is also why a human animator who's seen thousands of 2D frames
  can draw any angle without an explicit 3D model.)
- **Stylized 2D animation breaks the core assumption.** Anime/hand-drawn footage is NOT a faithful
  projection of a stable 3D object (smear frames, off-model drawings, expression-driven proportion
  changes, inconsistent model sheets, characters changing form). So there may be *no* single 3D
  that explains all frames → analysis-by-synthesis has nothing stable to converge to.
- **Driving is the EASY domain** for this program: real synchronized multi-camera rigs + LiDAR +
  ego-motion give abundant, geometrically consistent multi-view evidence. (Ties back to the
  user's career direction.)
- **Pat & Mat is a smart easy starting point** because it is **stop-motion** — real puppets filmed
  by a real camera, so the footage obeys perspective projection and is genuinely multi-view
  consistent (much closer to real video than to a cartoon). The characters are also fairly rigid /
  low-articulation, which neural blend skinning handles cleanly.

## 4. Research grounding (papers surfaced)

### Closest to the user's idea (general scenes)
- **GEN3C** (NVIDIA, CVPR 2025 Highlight) — maintains a spatiotemporal **3D cache** (point clouds
  from predicted depth of seed/previous frames), renders it under the new camera, conditions a
  video diffusion model on those renders. Depth via DepthAnythingV2, poses via DROID-SLAM. Only
  temporal layers of a Cosmos backbone are fine-tuned. Trained on RE10K, DL3DV, Waymo Open
  Dataset, and synthetic Kubric4D. Open weights: `nvidia/GEN3C-Cosmos-7B`; repo `nv-tlabs/GEN3C`.
  Inference ~43GB VRAM with full offloading.
- **Gen3R** (Jan 2026) — recasts a feed-forward reconstruction model (VGGT) as a VAE-like provider
  of **geometric latents**, combined with **appearance latents** from a pretrained video diffusion
  model; aligned-but-disentangled latent space; jointly generates RGB video + consistent point
  clouds. (Geometry prior lives *inside* the latent space — a step toward "learned 3D you can query.")
- **Voyager / WonderWorld** — explorable single-image-to-3D scene generation; joint frame+depth.

### Driving-specific world models (most relevant to the user's career)
- **Vista** (NeurIPS 2024) — generalizable driving world model on SVD; predicts high-fidelity
  futures controllable by multimodal actions (steering, speed, command, trajectory, goal point).
  Two-phase training (prediction; then frozen + learn action control). Dynamic priors:
  position/velocity/acceleration. Trained on OpenDV-YouTube (+ nuScenes). **Single-view.**
- **MagicDrive / MagicDrive3D / MagicDrive-V2** — multi-camera street-view generation conditioned
  on explicit 3D geometry: camera pose, BEV road map, 3D bounding boxes, text. Three condition
  levels (scene/foreground/background) + **cross-view attention** for multi-camera consistency.
  Trained on nuScenes; generated data augments BEV segmentation / 3D detection. MagicDrive3D adds
  deformable 3D Gaussian Splatting for any-view rendering. MagicDrive-V2 = high-res long video.
- **WoVoGen** — builds an explicit **4D world volume** (occupancy-like), forecasts it forward,
  decodes to consistent multi-camera video. Most BEVFormer-like. nuScenes + occupancy.
- **OccWorld** — pure 3D-occupancy world model: VQ-tokenize occupancy, GPT-style autoregressive
  prediction of future occupancy tokens + ego trajectory. Occ3D-nuScenes.
- **GenieDrive** — (1) occupancy world model predicts future 4D occupancy from controls; (2)
  occupancy-guided multi-view video generator (Wan2.1-1.3B base). nuScenes + Occ3D, 12 Hz
  multi-view, trained on 8× L40S 48GB.
- **DiST-4D** — disentangled spatiotemporal diffusion w/ metric depth for 4D driving. nuScenes.
- **WorldSplat** — Gaussian-centric feed-forward 4D scene generation. nuScenes.
- **Cosmos** (NVIDIA) — world foundation model platform (tokenizers + diffusion/AR video models)
  pretrained on massive video; the fine-tunable base under GEN3C.
- Survey anchor: **"The Role of World Models in Shaping Autonomous Driving"** (arXiv 2502.10498);
  plus world-models-for-AD survey (arXiv 2501.11260).

### The "learn implicit 3D from 2D, no 3D labels" line (the user's reinvented idea)
- **3D-aware generation from 2D collections, no 3D supervision:** pi-GAN, GIRAFFE, **EG3D**
  (learns a tri-plane 3D from single-view 2D photos + a camera-pose prior).
- **BANMo** (Meta/CMU, CVPR 2022) — *almost the user's exact proposal.* Builds animatable
  articulated 3D from many **monocular casual videos**, no template, no rig, via differentiable
  rendering. Supervision = color + silhouette + **optical flow** + DINO features; canonical 3D
  space + **neural blend skinning** + cycle-consistency. Predecessors: **LASR**, **ViSER**.
- **Maintained successor codebase: `lab4d-org/lab4d` (Lab4D)** — MIT-licensed, 4D reconstruction
  from monocular videos; folds in BANMo, RAC (Reconstructing Animatable Categories), Total-Recon,
  and ships preprocessing scripts. **Use this, not the archived `facebookresearch/banmo`.**
- Modern alternative: **MoSca** (dynamic Gaussian fusion from casual videos) — faster/trendier,
  less turnkey for the multi-video canonical case.

## 5. Hardware notes (RunPod, ~June 2026)

- **Inference / playing with GEN3C:** single A100 80GB or H100 80GB comfortable (~43GB with
  offloading). RunPod on-demand approx: A100 80GB ~$1.19–1.49/hr, H100 ~$2.69–3.29/hr, H200
  ~$4.39/hr, RTX 4090 ~$0.69/hr. Community/spot cheaper.
- **Fine-tuning a ~7B video DiT:** multi-GPU 80GB (4–8× A100/H100); GenieDrive reference = 8× L40S
  48GB on a Wan2.1-1.3B base. Roughly $10–25/hr.
- **Lab4D / BANMo per-subject optimization (the Pat & Mat project):** single A100 80GB fine; even
  24–48GB works with smaller batch.
- Training a multi-view driving world model from scratch is NOT realistic for an individual.
- RunPod startup program: up to 1,000 free H100 hours (worth applying for). Verify live prices &
  GPU availability before budgeting.

## 6. Public datasets referenced

- Driving (multi-camera): **nuScenes** (6 cams + LiDAR), **Occ3D-nuScenes** (occupancy labels),
  **Waymo Open Dataset**, **nuPlan**, **OpenDV / OpenDV-YouTube**, **Argoverse 2**, **KITTI-360**.
- General multi-view / camera control: **RealEstate10K (RE10K)**, **DL3DV-10K**,
  **Tanks & Temples**, **CO3D / MVImgNet**, **ACID**, synthetic **Kubric4D**.
- Pragmatic starts: nuScenes + Occ3D-nuScenes (driving); RE10K (camera control).

---

## 7. THE PLAN — Pat & Mat reconstruction (where we are now)

**Tooling correction:** build on **Lab4D** (`lab4d-org/lab4d`), not archived BANMo.
**Reality check:** raw episodes are step zero. ~80% of the work is the data-prep pipeline that
turns episodes into per-frame (RGB + mask + flow/tracks + features). Reconstruction itself is
largely a configured training run.

### Open decisions needed from the user (still pending)
1. **One character, one look** — Pat *or* Mat, single consistent appearance.
2. **Success criterion** — (a) learn the pipeline, (b) portfolio artifact, or (c) novel result.
3. **Compute ceiling** — how many GPU-hours willing to spend iterating.
4. **Codebase path** — default Lab4D; alternative MoSca (dynamic Gaussians). Recommend Lab4D first.

(Note: training on copyrighted footage for personal research/learning is standard/fine; just don't
distribute renders commercially.)

### Pipeline TODOs (ordered, with gates)

- **Phase 0 — prove the pipeline first.** Install Lab4D; run its provided cat/dog demo end to end.
  *Gate:* produce a novel-view render of the demo subject before touching Pat & Mat.
- **Phase 1 — episodes → clips.**
  - Shot/cut detection (PySceneDetect); operate within single-camera shots only.
  - Curate ~10–30 short clips where the target character is large, mostly unoccluded, and
    collectively span many viewpoints/poses (coverage > count).
  - Deinterlace/upscale old low-res footage if needed.
- **Phase 2 — per-frame signals (the real work).**
  - Segment the character per frame with **SAM2** (video/tracking mode) → clean mask sequences.
  - Generate motion/correspondence: optical flow and/or **CoTracker3** point tracks + DINO
    features. Lab4D preprocessing scripts orchestrate most of this.
  - Let Lab4D initialize rough camera/root-body poses.
  - *Gate:* one clip with aligned, sane RGB + mask + flow + features.
- **Phase 3 — reconstruct.** Run Lab4D optimization on ONE clip, then scale to the full set
  (canonical shape + blend-skinning weights + per-frame articulation + appearance via
  differentiable rendering against color + silhouette + flow).
  - *Gate:* recognizable canonical mesh + passable novel-view render from one clip → then full set.
- **Phase 4 — evaluate & iterate.** Render novel views, re-pose. Mush/floaters/collapsed limbs →
  usually fixed by adding clips covering the missing viewpoint. Diagnose by worst angles → back to
  Phase 1. **This loop IS the project.**
- **Phase 5 (stretch) — close the loop.** Drive the model with new bone transforms, OR use the
  recovered canonical+deformation model as the **learned geometry prior fed into a generator** —
  the "geometry learned from 2D, not handed in" bridge back to the original goal.

### Pat-&-Mat-specific gotchas
- **Stop-motion "on twos"** → choppy, large inter-frame jumps break dense optical flow. Lean on
  point tracking; **deduplicate doubled frames** so motion isn't artificially zero-then-huge.
- **Viewpoint coverage is the #1 risk** — cameras sit still within a shot, so **pool across many
  shots/episodes** (this is exactly why grabbing the whole series is the right instinct).
- **Occlusion/props** — start with clips where the character is clean and isolated; defer the
  cluttered workshop-chaos shots.

### Immediate next step
User to answer the 4 open decisions (esp. which character + success criterion). Then: produce a
concrete clip-count target, a Lab4D config starting point, and a rough GPU-hour/cost estimate for
the first reconstruction.

---

## 8. Suggested first prompt for the new session

> "Continuing a project from a previous session — context attached as markdown. I'm building an
> implicit/animatable 3D model of a Pat & Mat character from the series footage, BANMo/Lab4D style,
> as a stepping stone toward learning camera-consistent generation (career shift to self-driving).
> My decisions: character = [Pat/Mat]; goal = [learn pipeline / portfolio / novel result];
> compute = [budget]; codebase = Lab4D. Let's start at Phase 0/1. Give me the concrete clip-count
> target, a Lab4D config starting point, and a rough GPU-hour estimate."
