# Research: Training-Pipeline Architecture for the Large Smart-Space Dataset

*Research report · 2026-06-22 · LenaLab · "do we download all of it and copy to the pod? is there a
better pipeline architecture?" Report only — no implementation.*

## Executive summary

**No — you do not download the whole dataset, and you never copy raw videos to the pod.** The "massive"
size (~200 GB+ for all scenes) is **raw 1080p video**; our training input is **128×352 downsampled
frames**, which our adapter already produces. So the architecture is **decouple prep from train**:
preprocess video → *compact* shards **once**, and only the compact data ever reaches the GPU. For our
target (~30 scenes) the prepped data is **~12–15 GB** — small enough to just upload, and ideally parked
on a **RunPod network volume** (prep once, reuse across pods, no re-upload). True **streaming**
(WebDataset / MosaicML / HF `streaming=True`) is the right answer *only* if we scale to all ~141 scenes
or want a zero-copy pipeline — overkill at 15 GB. *(Confidence: high.)*

## The core principle: decouple preprocessing from training

Every scalable ML-data design says the same thing (Anyscale, AIStore/NVIDIA, MosaicML): **don't ship raw
media to the GPU and don't re-decode it every epoch.** Run a one-time **prep** step (video → extracted
frames + labels → compact shards), store the *derived* data, and let training read shards. *"Any time
spent in the preprocessing stage is repaid many times in training."*

For us this is already half-built — `prep_smartspace*.py` turns a 1.5–3 GB scene of video into a small
npz. The size collapse:

| | per scene | 30 scenes |
|---|---|---|
| raw 1080p video | ~1.5–3 GB | ~60–90 GB |
| **our prepped npz** (150 frames · 128×352) | **~0.4–0.5 GB** | **~12–15 GB** |

→ The 200 GB never needs to move. **Prep where the videos are; move only the ~15 GB derived data.**

## Where to put the prepped data — three tiers (pick by scale)

**Tier 1 — Upload compact shards to the pod (simplest; fits us now).**
Prep locally → `scp`/upload the ~15 GB npz to the pod's container disk → train. Zero new infra. Downside:
re-upload on every fresh pod (and we churn pods).

**Tier 2 — RunPod *network volume* (recommended sweet spot).**
Prep once → store the ~15 GB on a **persistent network volume** → mount it on any pod. Benefits:
- Survives pod termination; **no re-upload** each run; multiple pods (prep / train / eval) share it.
- Can be **populated via RunPod's S3-compatible API without launching a pod**.
- Cost ~$0.07/GB·mo (≈ $1/mo for 15 GB).
- Caveat: **locks the pod to that volume's datacenter** — pick a DC where pods actually come up
  (relevant given our earlier SSH-exposure issues), and it doesn't auto-sync across DCs.

**Tier 3 — True streaming from object storage (scale-up path; not needed yet).**
Store shards in S3/HF and **stream during training** — no local copy at all:
- **WebDataset** — `.tar` shards, sequential streaming; the de-facto standard for many small files
  (our per-frame samples fit this well).
- **MosaicML StreamingDataset (MDS)** — purpose-built drop-in `IterableDataset` for cloud streaming;
  deterministic shuffling, multi-node, convergence == local disk.
- **HF `datasets` streaming** (`load_dataset(..., streaming=True)`) — stream straight from the HF hub
  (recent: 100× fewer requests, 2× throughput); zero download.
- Worth it at **100s of GB–TB / many nodes**; adds a re-shard step + complexity. **Overkill for 15 GB.**

**Tier 4 (frontier) — disaggregated CPU-prep → GPU-train (Ray Data / Anyscale).**
CPU fleet decodes+preprocesses and streams batches to the GPU fleet concurrently, no disk
materialization. The end-state for continuous large-scale pipelines; far beyond our needs.

## Recommendation for *us*

1. **Keep prepping to compact npz; never move raw video to the pod.** (Already how it works.)
2. **Stand up a RunPod network volume** in a known-good datacenter; prep the curated ~20–30 scenes once
   (locally or a cheap CPU pod) and write the npz to the volume via the **S3 API**. Then every training/
   agent pod just **mounts** it — no re-download, survives pod churn, ~$1/mo. This directly fixes the
   "copy it to the pod every time" pain.
3. **Optionally shard to WebDataset `.tar`** if loading many small npz becomes an I/O bottleneck — it's
   a small change and future-proofs toward streaming.
4. **Defer MDS/HF-streaming** until/unless we train on all ~141 scenes — note it as the documented
   scale-up path.

A small storage-vs-speed note: our npz stores *decoded* frames (fast to train, no per-epoch video
decode, but larger). At our scale that's the right trade. At 141-scene scale, store **JPEG-encoded**
frames in shards (smaller) and decode on the GPU.

## Sources
- [MosaicML StreamingDataset (Databricks)](https://www.databricks.com/blog/mosaicml-streamingdataset) · [streaming lib](https://github.com/mosaicml/streaming)
- [HF datasets streaming](https://huggingface.co/docs/datasets/en/stream) · [streaming 100× more efficient](https://huggingface.co/blog/streaming-datasets)
- [RunPod network volumes](https://docs.runpod.io/storage/network-volumes) · [why network volumes](https://www.runpod.io/blog/network-volumes-on-runpod-secure-cloud)
- [Anyscale: multimodal data pipelines at scale](https://www.anyscale.com/blog/architecting-multimodal-data-pipelines-that-scale-with-ray) · [NVIDIA AIStore sharding (WebDataset)](https://aistore.nvidia.com/blog/2024/08/16/ishard) · [Vid Prepper (video prep)](https://towardsdatascience.com/introducing-vid-prepper/)

*Confidence: high that raw video need not touch the pod and that a network volume is the right tier for
~15 GB; medium on exact prepped-size (depends on frames/cameras/resolution kept) and on whether
WebDataset sharding is worth it before the full-scale jump.*
