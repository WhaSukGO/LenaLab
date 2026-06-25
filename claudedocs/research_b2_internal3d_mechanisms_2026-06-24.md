# Research: B2 Methods — How Each Uses *Internal 3D* to Generate an Image (mid-2026)

*Research report · 2026-06-24 · the B2-only deep dive: methods whose OUTPUT is an image/video (not a mesh)
but whose architecture may use 3D internally. Organized by **internal-3D representation + query mechanism**,
ending at the open seam. Paper-grounded. Report only — no implementation.*

## Executive summary
B2 splits cleanly into **two mechanism families**:
- **B2-i — no held 3D** ("3D" = a pose/ray *condition* + **attention** as the consistency glue). There is
  **no addressable volume** — you can't look up a 3D point; you resample the whole network under new camera
  conditioning and *hope attention re-agrees*. **Breaks at wide angles** (the correspondence attention
  relies on weakens) → flicker/inconsistency. *Members: Zero123/123++, MV-Adapter, SEVA, CAT3D, Cavia.*
- **B2-ii — a held internal 3D structure** that is turned into a conditioning signal. *Members: GEN3C, Mirage
  /Latent Spatial Memory, Gen3R, Captain Safari, Geometry Forcing.* These differ on **(a) what's stored**
  and **(b) how it's queried** (render/warp-then-condition vs latent-concat/align vs **QKV attention**), and
  crucially **all take geometry from off-the-shelf estimators** (none learn it end-to-end).

**The open seam (your hypothesis, now precisely pinned):** the cell **{geometry LEARNED end-to-end by
analysis-by-synthesis} × {queried via QKV into a held 3D store} × {image output}** is **empty.** And the most
*extensible* method toward it has open code — **Captain Safari** (it already does QKV-into-a-3D-store; you'd
make its store *learned* instead of frozen). This is a single-H100-feasible wedge.

## Family B2-i — "3D" = pose/ray conditioning + attention (no held volume)
| Method | What stands in for 3D | Consistency mechanism | Stylized? | Output |
|---|---|---|---|---|
| Zero-1-to-3 / 123++ | posed-CLIP `(R,T)` embedding | per-view conditioning (123++ adds reference/tiled attn) | incidental | image |
| **MV-Adapter** | **raymap** (Plücker rays) | **parallel/decoupled multi-view attention** bolted on frozen SDXL | **yes — anime/style LoRAs plug in** | images |
| SEVA / Stable Virtual Camera | Plücker cameras | 2D self-attn **"inflated" to 3D** over all views + two-pass anchor→interp | no claim | image+video |
| CAT3D | raymap concat to latent | **3D self-attention** across views (no epipolar) → then robust NeRF | no | images→NeRF |
| Cavia | Plücker concat | view-integrated cross-view + cross-frame attn (SVD) | no | videos |

**Key property:** "3D" here is never a *structure you can query* — it's a **ray hint + soft attention exchange**.
**Where it breaks:** the bigger the angle from the input, the weaker the attention correspondence → **flicker
at wide baselines** (SEVA admits this verbatim; CAT3D concedes views are "generally not perfectly 3D
consistent"). The tells: CAT3D launders inconsistent views through a **robust NeRF**; SEVA needs a **two-pass**
sampler to suppress flicker — both admissions that attention alone ≠ true wide-baseline consistency.
**Best stylized fit:** **MV-Adapter** (only one with a real anime/LoRA story) — but it inherits the wide-angle limit.

## Family B2-ii — a held internal 3D, turned into conditioning
| Method | Held representation | **Query mechanism** | Geometry source | Output |
|---|---|---|---|---|
| GEN3C | explicit **RGB** point cloud | **render-then-condition** | **given** (DepthAnything+DROID) | video |
| Mirage / Latent Spatial Memory | **latent** point cloud (f∈ℝ⁴⁸) | **warp-then-condition** (ControlNet branch) | **given** (DepthAnything3) | video |
| Gen3R | per-gen **geometric latents** (VGGT adapter) | **latent concat + KL align** | **given** (VGGT *frozen*) | video+geometry |
| **Captain Safari** | **pose-aligned latent tokens** | **QKV cross-attention** ← the only one | **given** (StreamVGGT *frozen* + COLMAP) | video |
| Geometry Forcing | none (aligned latents) | **alignment loss** (regularizer) | **given** (geo foundation model) | video |

**Two axes that matter:**
- **Query:** render/warp-then-condition (GEN3C, Mirage) · latent-concat/align (Gen3R, Geometry Forcing) ·
  **QKV (Captain Safari only)**.
- **Geometry:** **every one is *given*** (off-the-shelf depth/pose/VGGT) — **none learned end-to-end by
  analysis-by-synthesis.**

## The open seam (explicit)
**No method satisfies BOTH** {geometry learned end-to-end} **AND** {queried via QKV}:
- **Captain Safari** is the *only* QKV-into-3D-store method — but its store is built from **frozen StreamVGGT
  + COLMAP poses** → geometry handed in.
- **Gen3R / Geometry Forcing** have "learned-ish" geometry but **freeze** the encoder / use it as a
  **regularizer**, and **don't query via attention** (concat / align).
- **GEN3C / Mirage** take depth from off-the-shelf estimators and **render/warp-then-condition**.
→ Nobody builds a 3D feature store that **emerges purely from the generation objective** *and* **attends
(QKV) into it** to produce the view. **That empty cell is exactly your hypothesis.**

## Why this explains your real failures
Your hard case = **wide angle (close-up→top-down) on flat 2D**. The taxonomy predicts the failure:
- **B2-i** (incl. MV-Adapter, the stylized one) = attention-only → **degrades exactly at wide angles**.
- **B2-ii** holds real 3D but **gets geometry from estimators trained on photos** → **those fail on flat 2D
  art** (the stylized problem). So the held-3D methods can't even build a correct store from your input.
→ The consumer tools you already know are all B2-i-pure-2D or B2-ii-given-geometry; **both wings break at
your exact corner**, which is why none satisfy you. The corner that *would* — a **learned** 3D store
**queried by attention**, robust to bad/flat-art depth because geometry is corrected by the generation
objective — **is the open seam.**

## The buildable wedge (1 H100 + ~1000 free hrs)
**Extend Captain Safari** (open code + weights, Apache-2.0; already has the QKV-into-store machinery):
- **Smallest change:** replace its **frozen StreamVGGT** memory-token extraction with a **trainable,
  lightweight pose-tagged 3D encoder trained jointly under the DiT denoising loss** (keep pose tags from
  camera params; drop the StreamVGGT supervision). The store's geometry then **emerges from the generation
  objective (analysis-by-synthesis)** while retrieval **stays cross-attention (QKV)** → lands the empty cell.
- Scale: LoRA on the DiT + a small trainable encoder — fine-tune-scale, single-H100 feasible.
- *(Fallbacks: Gen3R, open — but you'd convert concat→attention AND unfreeze VGGT, larger. Mirage code
  "coming soon".)*

## If the goal stays "tool, not research"
Among B2, **MV-Adapter** is the most stylized-ready (anime/LoRA-native, SDXL) — but accept its **wide-angle
limit** (it's B2-i). For wide angles on flat 2D, no B2 method is clean today; that's the gap.

## Sources
- Zero-1-to-3 https://arxiv.org/abs/2303.11328 · MV-Adapter https://arxiv.org/abs/2412.03632 · SEVA https://arxiv.org/abs/2503.14489 · CAT3D https://arxiv.org/abs/2405.10314 · Cavia https://arxiv.org/abs/2410.10774
- GEN3C https://research.nvidia.com/labs/toronto-ai/GEN3C/ · Mirage/Latent Spatial Memory https://arxiv.org/html/2606.09828 (code https://github.com/microsoft/LatentSpatialMemory) · Gen3R https://arxiv.org/html/2601.04090v1 (https://github.com/JaceyHuang/Gen3R) · Captain Safari https://arxiv.org/html/2511.22815v1 (https://github.com/johnson111788/Captain-Safari) · Geometry Forcing https://arxiv.org/abs/2507.07982

*Confidence: HIGH on the two-family taxonomy and the query-mechanism/geometry-source classification (paper
quotes). HIGH that the {learned-geometry × QKV} cell is empty. MEDIUM on the Captain-Safari-extension being
the *smallest* viable wedge (depends on how tightly its retrieval is coupled to StreamVGGT features — verify
at the code level before committing).*
