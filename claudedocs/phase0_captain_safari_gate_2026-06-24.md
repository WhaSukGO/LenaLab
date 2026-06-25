# Phase 0 — Gate D: Captain Safari Coupling Check (PASSED)

*Implementation note · 2026-06-24 · `/sc:implement` Phase-0 architecture gate for the learned-3D+QKV wedge
(`design_learned3d_qkv_wedge_2026-06-24.md`). Pure code investigation (repo cloned + read; no GPU/training).*

## Verdict: ✅ FEASIBLE on 1×H100 (via the PreEnc path). The swap plan is sound.
Repo `github.com/johnson111788/Captain-Safari` (Apache-2.0, CVPR'26, built on DiffSynth-Studio). **Training
code + weights + demo data are released** (not inference-only) — entry points exist for retriever warmup,
joint LoRA training, and inference.

## The key finding: coupling is two-tier
- **DiT ↔ memory = LOOSELY coupled (good).** The DiT only ever sees `memory_context = memory_emb(retrieved)`
  — a plain `[B, N, 3072]` sequence consumed by an ordinary cross-attention injected into every block
  (`wan_video_dit.py:282-283`, `255`). It is blind to token count, pose, and the upstream feature source.
  **Swapping what produces that sequence is clean.**
- **Retriever ↔ StreamVGGT = TIGHTLY coupled (the one constraint).** StreamVGGT's exact output format is
  baked into `MemoryRetriever` (`wan_video_dit.py:571-759`): input dim **1024**, token count **4×782**, and a
  **L=4 / H=21 / W=37 / 5-special-token** layout hard-wired as literals across ~20 sites + the 3D-RoPE
  builder. The retriever is *parameterized, not architecturally locked* — but those dims are literals you
  either **match** or **edit**.

**The store contract** (what a replacement must emit): per frame `[4, 782, 1024]` = 4 ViT layers × (1 camera
+ 4 register + 777 patch = 782 tokens, patch grid 21×37) × 1024 (DINOv2-L dim); poses carried *separately*
as **9-dim** encodings; the retriever consumes `memory` as `[B, T×4×782, 1024]` + key/target pose tokens.

## The single most important code change
Today `memory` is **pre-baked to `.npy` on disk** and enters the pipeline as a tensor (`wan_video_new.py:
1318-1376`) — **nothing backprops into StreamVGGT.** To train the store end-to-end you must **relocate
feature extraction INTO the forward graph** (around `wan_video_new.py:1376`), so `memory = store_builder(
frames, poses)` is in-graph and receives the diffusion-loss gradient. *This relocation is the crux of the
whole wedge.*

## Concrete swap plan (cheapest path — zero retriever edits)
1. **Trainable store-builder** that emits the **same `[*,4,782,1024]` contract** (e.g. a small ViT/depth-prior
   backbone with a 4-layer × 782-token head) → **no retriever changes needed**. *(Or pick your own
   `(L,H,W,dim)` and edit the ~20 literal sites + RoPE defaults — mechanical.)*
2. **Move extraction in-graph** at `wan_video_new.py:~1376` (the crux change above).
3. **Unfreeze** the store-builder in `train.py:118-139` (the unfreeze plumbing for `memory_emb` /
   `memory_retriever` / `memory_cross_attn` already exists — add `store_builder` alongside).
4. **Warm-start** the store-builder from a **depth prior**, then train on the **joint diffusion loss** (+
   optional aux geometry loss). **Drop the repo's MSE-to-StreamVGGT retriever warmup** (its target *is*
   StreamVGGT — the opposite of "learned geometry").
- **Frozen:** base Wan2.2 DiT (adapted via **LoRA rank 32** on q,k,v,o,ffn). **Trained:** LoRA + memory_emb +
  memory_cross_attn + memory_retriever + the new store_builder. (Mirrors the existing recipe.)

## Run reality (1×H100)
- **Weights:** base `PAI/Wan2.2-Fun-5B-Control-Camera` (~10 GB) + UMT5-XXL (~11 GB) + Wan2.2 VAE; auto-download,
  no gating. StreamVGGT (~5 GB) is needed **only** for the *old* offline data prep — **we don't need it.**
- **VRAM:** inference fits 80 GB comfortably (CPU-offload T5/VAE). Training is **tight but feasible via the
  PreEnc path** (pre-cache latents → only the 5B DiT in VRAM) + LoRA + grad-checkpointing + bs1 + shorter
  clips + a **small** store backbone (ViT-B/depth head, *not* a 1B model — the in-graph store activations are
  the main 80 GB risk). Reference recipe is 8-GPU; solo = PreEnc single-H100.

## Big simplification for OUR plan (synthetic toon data)
The agent flagged the **hardest part is the data** — the OpenSafari offline pipeline (COLMAP/hloc SfM +
StreamVGGT over many videos). **Our toon-render data sidesteps all of it:** we render toon-3D from **known
cameras** with **GT depth**, so we have poses + depth *for free* → **no COLMAP, no SfM, no StreamVGGT**. The
depth-prior warm-start is also free (rendered GT). This removes the biggest cost the reference project paid.

## De-risking order (next)
1. **Overfit-one-clip smoke test** on the released `demo_data.tar.gz`: confirm gradients flow
   **store → retriever → DiT** and improve the diffusion loss vs the frozen-StreamVGGT baseline. (Validates
   the crux change before any data work.)
2. **Toon-multiview data pipeline** emitting the training contract (RGB + GT depth + 9-dim poses).
3. If 5B is too tight on 1×H100 → retarget the shipped **Wan2.1-Fun-1.3B** variant.
- Fallback if training proves infeasible: smaller Wan + smaller store; Gen3R is strictly *more* work (here
  the DiT, retriever, training loop, LoRA plumbing, and unfreeze logic already exist).

## Bottom line
The architecture gate **passes**: DiT is swap-friendly, the retriever is the (editable) coupling point, the
training infra exists, and our synthetic-toon data removes the reference project's worst cost. The wedge is a
**well-scoped fork**, not a from-scratch build — feasible for a solo researcher on 1×H100 via PreEnc + LoRA +
a small in-graph store-builder.
