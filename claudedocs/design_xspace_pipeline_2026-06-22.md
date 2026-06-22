# Design: Cross-Space Scale-Up Pipeline (network volume + prep-once)

*Design spec · 2026-06-22 · LenaLab · `/sc:design` (spec only — no implementation).*
*Grounds: `research_xspace_datasets_2026-06-22.md` (more scenes), `research_training_pipeline_arch_2026-06-22.md`
(don't move raw video; network-volume tier), and existing `prep_smartspace_xspace.py` +
`smartspace_xspace_ref.py` (already multi-scene + checkpoint + resume).*

## 1. Goal & scope
Lift cross-space generalization from the failed 4-scene baseline (unseen IoU ~0.04) by training the
**scene-agnostic** model on **many diverse spaces**, with a data architecture that **never copies raw
video to the GPU** and **preps once, reuses forever** (RunPod network volume). Resume from our saved
pretrained checkpoint. Keep every checkpoint.

**v1 scope (recommended):** 2026 edition only (`Warehouse_*`, schema we already parse) — **~17 train /
3 unseen val** warehouses. Same env *type* (warehouse) but a real cross-*space* test. Multi-env-type
(2024 retail/hospital) is a **deferred Phase B** (needs a schema check; bigger payoff for true diversity).

## 2. The principle (from the pipeline research)
**Decouple prep from train.** Raw video (~1.5–3 GB/scene) is processed once into compact npz
(~0.4–0.5 GB/scene); only the compact data lives on the **network volume** and is mounted by GPU pods.
17 train + 3 val ≈ **~10 GB prepped** (vs ~40 GB raw) → a small, persistent, reusable dataset.

## 3. Data-flow architecture

```
                         ┌─────────────────────────────────────────────┐
  HuggingFace (raw video, ~40GB)                                        │
        │  stream/download per scene                                    │
        ▼                                                               │
  ┌──────────────┐   prep_smartspace_xspace.py (CPU, one-time, idempotent)
  │  PREP stage  │   video → 150 frames @128×352 → padded npz (+cam_valid, grid_bounds, bev)
  │ (local OR    │   discard raw video after each scene
  │  cheap CPU pod)
  └──────┬───────┘
         │  upload compact npz  (RunPod S3 API — no pod needed)
         ▼
  ┌─────────────────────────────────────────────┐
  │  RunPod NETWORK VOLUME  /vol  (persistent)   │   ← prep once, reuse across all runs
  │   /vol/xspace/Warehouse_000/*.npz ...        │
  │   /vol/xspace/Warehouse_020/*.npz  (val)     │
  │   /vol/checkpoints/                          │
  └──────┬───────────────────────────────────────┘
         │  MOUNT (read) on GPU pod
         ▼
  ┌──────────────┐   smartspace_xspace_ref.py  (resume from pretrained ckpt)
  │  TRAIN stage │   train on N warehouses → val on UNSEEN ones → save ckpt → /vol/checkpoints
  │  (GPU pod)   │
  └──────┬───────┘
         │  best ckpt
         ▼
  EVAL / AGENT stage (mount /vol) → unseen-space IoU; optionally agent authors cross-space model
```

Key property: the **GPU pod only ever sees the ~10 GB compact volume** — raw video never touches it, and
no re-upload between pods (the volume persists).

## 4. Components (concrete, mapped to files)

| Component | File | Status |
|---|---|---|
| Scene manifest (train/val split, edition) | `configs/xspace_scenes.json` (new) | to add |
| Prep adapter (video→padded npz, idempotent skip) | `scripts/prep_smartspace_xspace.py` | **have** — add skip-if-exists + out=volume path |
| Network-volume provision + S3 upload | `scripts/cloud/xspace_volume.sh` (new) | to add (RunPod API + S3) |
| Scene-agnostic trainer (multi-scene, ckpt, resume) | `scripts/smartspace_xspace_ref.py` | **have** — point train/val dirs at `/vol` |
| Eval-on-unseen | reuse `evaluate()` in the trainer | **have** |
| Agent cross-space task (Phase 3) | `vo_lab/agents/smartspace_xspace_implementer.py` (new) | deferred |

## 5. Network-volume spec (from RunPod research)
- **Provision** a network volume (RunPod API/console) in a **known-good datacenter** (one where pods
  reliably expose SSH — e.g., the console-pod DC). *Caveat: the volume locks pods to that DC.*
- **Size:** ~30 GB (≈10 GB data + checkpoints + headroom) → ~$2/mo.
- **Populate without a pod:** upload prepped npz via RunPod's **S3-compatible API** (`runpodctl` or any
  S3 tool) — prep locally, push to the volume, done.
- **Use:** every train/eval/agent pod **mounts** the volume read-(write for ckpts); no re-download.
- **Layout:** `/vol/xspace/<Scene>/*.npz` + `meta.json`; `/vol/checkpoints/<run>.pt(.full.pt)`.

## 6. Storage-format decision
Keep **decoded-frame npz** for v1 (our code works; ~10 GB is fine; no per-epoch video decode).
*Deferred scale-up:* if we go to all ~141 scenes, switch to **JPEG-frames-in-shards / WebDataset `.tar`**
(smaller, streamable) + GPU decode — noted, not built now.

## 7. Compute plan
- **Prep:** CPU-bound (video decode). 20 scenes locally ≈ tens of minutes (we did 5 in ~10 min). Free.
- **Train:** 17 scenes ≈ ~4× our 4-scene run. On the local 3080 that's multi-hour; on a **cloud 4090
  it's ~1 hr**. Recommend the cloud pod (mounting the volume) for the real run; local 3080 for a smoke.
- **Resume** from `xspace_ref_pretrained.pt.full.pt` so we build on, not restart, prior training.

## 8. Phased execution (for `/sc:implement`)
- **Phase 0 — volume + plumbing.** Provision the network volume (DC choice); verify S3 upload of one
  prepped scene + mount-read from a pod. *(If Phase B: verify the adapter on one 2024 `scene_*`.)*
- **Phase 1 — prep+upload (one-time).** Prep the curated ~20 scenes locally → push npz to `/vol` via S3.
  Idempotent (skip already-uploaded).
- **Phase 2 — train.** GPU pod mounts `/vol`; train scene-agnostic ref on the 17 train warehouses,
  **resume from the pretrained ckpt**; val on unseen W020/021/022; save ckpt to `/vol/checkpoints` +
  commit best to repo. Report unseen-space IoU vs the 0.04 baseline.
- **Phase 3 (optional) — agent.** Agent authors a cross-space model against the same volume.
- **Phase 4 — docs.** Update the domain report / RESULTS with the scaled cross-space number (honest).

## 9. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Network volume **locks to a DC** (+ our past pod-SSH issues) | Provision in the DC where the console pod worked; keep a local npz copy as backup. |
| **2024 schema differs** (`scene_*`, multi-type) | Keep v1 to 2026 warehouses; gate Phase B on a one-scene adapter check. |
| Prep time / disk for raw video | Stream+prep per scene, **delete raw after each**; only npz persists. |
| Train compute (17 scenes) | Cloud 4090 for the real run; resume to avoid wasted epochs. |
| **Synthetic-only** (may not = real cameras) | Note MMPTRACK (real) as the sim→real cross-check after v1. |
| More scenes still may not close the gap | Honest either way — a scaled negative result is still a finding (the gap is fundamental). |

## 10. Decisions needed (before `/sc:implement`)
1. **Scope:** v1 = 2026 warehouses (~17 train / 3 val, easy) — or include 2024 multi-type now (more
   diverse, +schema work)? *(Rec: v1 first.)*
2. **Train compute:** cloud 4090 pod (mount volume, ~1 hr) — or local 3080 (free, multi-hr)? *(Rec: cloud
   for the real run; local smoke first.)*
3. **Network volume now, or skip it for v1** and just upload ~10 GB to one pod? *(Rec: network volume —
   it's the whole point of this design and ~$2/mo; but a one-pod upload is fine if you want to defer it.)*

*Recommended path: v1 = 2026 warehouses · network volume in the known-good DC · prep locally → S3 upload
· train on a cloud 4090 mounting the volume · resume from the pretrained checkpoint · keep all ckpts on
the volume + commit best. Next: `/sc:implement` Phase 0.*
