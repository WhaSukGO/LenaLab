# Design: ComfyUI "proxy → ControlNet → Ghibli" Camera-Change Pipeline

*Design spec · 2026-06-24 · `/sc:design` (spec only — no implementation). Goal: given a stylized (Ghibli/
2D) image, produce the **same subject at a new camera angle**, in-style. Companion to
`workflow_image_to_camera_view.md`.*

## 1. Requirements & scope
- **In:** one stylized source image + a desired new camera angle (e.g., top-down).
- **Out:** a flat, in-style image of the same subject/identity at that angle.
- **Principle:** geometry is a *guide* from a rough 3D proxy (exact camera); **style + identity come from a
  2D generator** (ControlNet for geometry, IP-Adapter for identity/palette, LoRA for the Ghibli look).
- **v1 scope: character-level** (the solid case). Whole-scene is a documented stretch (§8).
- **Non-goal:** geometric faithfulness — 2D art has no true 3D, so unseen regions (top of head) are
  *invented in-style*. Acceptable for art.

## 2. Architecture — three stages

```
 STAGE A: PROXY (rough 3D)            STAGE B: CONTROL MAPS (Blender)        STAGE C: GENERATE (ComfyUI)
 ┌───────────────────────┐           ┌──────────────────────────────┐      ┌────────────────────────────┐
 source image ──image→3D──▶ rough     place camera at NEW angle ─────▶      depth map  ─▶ ControlNet(depth)
 (Tripo / Hunyuan3D /     │  mesh ───▶  render:  • depth (Z-pass)    │      pose map   ─▶ ControlNet(pose)
  TRELLIS)                │           │          • proxy RGB (→pose) │      source img ─▶ IP-Adapter (id)
 └───────────────────────┘           │          • (opt) normal      │      prompt+Ghibli LoRA ─▶ SDXL
        (geometry only,              └──────────────────────────────┘             │  KSampler
         texture irrelevant)              exact camera control HERE                ▼  VAE decode
                                                                          new-angle in-style image
                                                                                 │ (opt) polish →
                                                                          Midjourney / nano-banana
```

**Why this split:** the 2D generator never "understands" 3D — it just obeys the depth/pose maps the proxy
produced. The proxy supplies *where things are at the new camera*; the generator supplies *how it looks*.

## 3. Stage A — proxy (rough 3D)
- **Tool:** Tripo (hosted, free tier) for the first test; Hunyuan3D 2.5 / TRELLIS on a GPU for control.
- **Input tip:** feed a **fuller/¾ shot**, not a face crop (image→3D loses identity from crops).
- **Export:** a mesh (`.glb`/`.obj`). Texture quality is irrelevant — we only use geometry.

## 4. Stage B — control-map render (Blender)
Render the proxy from the **target camera**, at the **output resolution/aspect** (e.g., 1024×1024):
| Output | How | Used by |
|---|---|---|
| **Depth map** | Z-pass / Mist, normalized **near = bright** (ControlNet-depth convention) | depth ControlNet (primary geometry) |
| **Proxy RGB** | flat render from the new camera | → DWPose preprocessor → pose ControlNet |
| **Normal map** *(optional)* | normal-pass | optional normal ControlNet (extra shape stability) |
- Keep the camera intrinsics fixed; export camera params for reproducibility.
- *(Humanoid)* if the proxy is rigged, you can render an OpenPose image directly; otherwise derive pose from
  the proxy RGB via DWPose in Stage C.

## 5. Stage C — the ComfyUI graph (node-by-node)

**Model set (SDXL track — recommended v1; see decision §9):**
| Role | Model (verify exact version at build) |
|---|---|
| Base checkpoint | anime/illustration SDXL (e.g., Animagine-XL-class) |
| Style | a **Ghibli LoRA** (strength 0.7–0.9) |
| Geometry | **ControlNet-Union-SDXL** (xinsir) *or* separate depth + openpose SDXL ControlNets |
| Identity/palette | **IP-Adapter-Plus (SDXL)** + CLIP-ViT-H image encoder |
| Pose preproc | **DWPose / OpenPose** preprocessor |

**Graph (wiring):**
1. `Load Checkpoint` → MODEL, CLIP, VAE
2. `Load LoRA` (Ghibli) ← MODEL, CLIP → MODEL₁, CLIP₁
3. `CLIPTextEncode` ×2 (positive: *subject + new-angle description + style tags*; negative: *artifacts,
   extra limbs, photo*) ← CLIP₁
4. `Load CLIP Vision` + `Load IPAdapter Model` → `IPAdapter Advanced` ← MODEL₁, image = **source image**,
   weight ≈ 0.7 → MODEL₂
5. `Load ControlNet`(depth/union) → `ControlNetApplyAdvanced` ← (pos, neg), image = **Blender depth**,
   strength ≈ 0.7, end_percent ≈ 0.8 → pos₁, neg₁
6. *(chain)* `DWPreprocessor`(proxy RGB) → `ControlNetApplyAdvanced`(pose) ← (pos₁, neg₁), strength ≈ 0.6
   → pos₂, neg₂
7. `EmptyLatentImage` (1024×1024 / target aspect)
8. `KSampler` ← MODEL₂, pos₂, neg₂, latent — **steps 30, CFG ≈ 6, dpmpp_2m + karras, denoise 1.0** (txt2img:
   build the new view from noise *guided by* depth+pose+IP-Adapter; do **not** img2img from the source —
   that fights the new geometry)
9. `VAEDecode` → `SaveImage`
10. *(optional)* hi-res-fix / upscale pass; then **external polish** (Midjourney/nano-banana with source as
    character reference) to lock final style.

**LoadImage inputs:** source image (→IP-Adapter), Blender depth (→depth CN), proxy RGB (→DWPose→pose CN).

## 6. The three-way tension (the core craft)
Three controls compete — **tune in this order**, one at a time:
| Knob | Raises | Too high → | Start |
|---|---|---|---|
| **Depth ControlNet strength** | geometry/new-angle adherence | rigid, kills style, "3D-render" look | 0.7 |
| **IP-Adapter weight** | identity/palette/design fidelity | copies the *original pose*, fights new angle | 0.7 |
| **Ghibli LoRA strength** | the flat 2D look | washes out identity/detail | 0.8 |
Rule of thumb: depth ≥ IP-Adapter for big angle changes (geometry must win); lower IP-Adapter if the output
refuses to leave the original pose; raise depth end_percent if the silhouette ignores the new camera.

## 7. Validation gates (cheap → expensive)
- **G1 — maps align:** Blender depth + DWPose register to the target resolution and read correctly. ($0)
- **G2 — geometry holds:** one generation matches the **new angle** silhouette (tune depth strength). 
- **G3 — identity+style hold:** same character + Ghibli look (tune IP-Adapter + LoRA).
- **G4 — extreme angle acceptable:** top-down looks plausible in-style (unseen parts invented — OK).
**Cheapest first test:** Tripo proxy → Blender depth (+DWPose) → a ComfyUI SDXL+depth-CN+IP-Adapter graph on
one image; judge G2+G3 before investing further.

## 8. Failure modes & mitigations
| Failure | Fix |
|---|---|
| Output keeps the *original* pose | lower IP-Adapter weight; raise depth strength/end_percent |
| Looks like a 3D render, not flat 2D | raise Ghibli LoRA; lower depth strength; final nano-banana polish |
| Identity drifts at new angle | raise IP-Adapter; add a second reference; restyle pass with source as ref |
| Garbled disocclusions (top-down) | inpaint the invented region; accept in-style invention |
| Whole-**scene** breaks | v1 is character-only; for scenes use a coarse scene proxy or warped-depth + in-style inpaint (fragile, extreme angles worst) |

## 9. Decisions needed (before `/sc:implement`)
1. **Base model:** **SDXL** (recommended v1 — most mature multi-ControlNet + IP-Adapter, fits 16 GB) vs
   **Flux** (higher quality, needs ~24 GB / a pod, newer control ecosystem).
2. **Run location:** **local 3080** (this is *inference*, light — fine for SDXL) vs a **RunPod ComfyUI pod**
   (needed for Flux / heavier).
3. **Scope:** **character-only** (v1, solid) vs attempt **whole-scene** now (harder).
4. **Style source:** **Ghibli LoRA + IP-Adapter** (self-contained) vs lean on a **final Midjourney/
   nano-banana polish** for the look (simpler graph, external step).

## 10. Next step
`/sc:implement` builds the concrete artifacts: the **ComfyUI workflow `.json`** (wired per §5), a **Blender
export script** (depth/RGB/normal at a chosen camera), and a short **runbook** + the G1–G4 test images.

*Recommended defaults: SDXL · local 3080 · character-only · Ghibli-LoRA + IP-Adapter (polish optional).*
