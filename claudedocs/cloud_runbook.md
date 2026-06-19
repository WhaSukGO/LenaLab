# Running LenaLab on a cloud GPU — runbook

Goal: run the agent research lab on a rented GPU (bigger VRAM, no contention, parallel experiments).
API cost is company-backed, so the only spend is GPU-hours (≈ **$0.40–1.40/hr**; a full domain study ≈ **$3–10**).

## The one architecture fact that decides everything
The lab sandboxes each agent job with `docker run --gpus all` (a real Docker daemon + GPU). So target a
**bare GPU VM with Docker + NVIDIA Container Toolkit** — then the lab runs **exactly as it does locally,
zero changes**. A RunPod *pod* is itself a container (nested Docker is fiddly), so prefer a true VM:
**Lambda Cloud**, **your company's GCP/AWS** (g5/A2 with the "Deep Learning" image), or RunPod's VM tier.

---

## ✅ What YOU need to do (human steps — can't be scripted)

1. **Pick + create the account.**
   - Easiest managed: **Lambda Cloud** (lambdalabs.com) — clean GPU VMs with Docker preinstalled.
   - If your company already has **GCP/AWS** with GPU quota + credits → use that (most natural, billed to the company).
2. **Add billing / confirm credits.** GPU instances need a payment method or company credits enabled. (Budget is tiny — tens of dollars — but the account still needs billing active.)
3. **Add your SSH public key** to the provider (so you can `ssh` into the VM). On your laptop:
   `cat ~/.ssh/id_ed25519.pub` → paste into the provider's SSH-keys page (or create one with `ssh-keygen -t ed25519`).
4. **Launch a GPU instance** with a CUDA/Docker image:
   - **Routine lab work:** 1× **RTX 4090 / L40S (24–48 GB)** — runs everything we've built.
   - **Full-scale headline number:** 1× **A100-80GB or H100**.
   - Choose an image that already has **NVIDIA driver + Docker + nvidia-container-toolkit** (Lambda's default + RunPod "PyTorch" / GCP "Deep Learning VM" all do).
5. **One secret to paste on the VM:** `ANTHROPIC_API_KEY` = your company key (for live/billed runs;
   calibration runs without it). The Touchstone spine (`github.com/WhaSukGO/touchstone`) is **public**,
   so the bring-up clones it automatically — nothing for you to do there.
6. **`ssh` into the VM** and run the bring-up (below). When done for the session, **stop/terminate the
   instance** (you only pay while it runs).

That's the whole human side: *account → billing → SSH key → launch a GPU VM → ssh in → paste API key → run script → stop when done.*

---

## 🤖 Bring-up (on the VM, scripted)

```bash
export ANTHROPIC_API_KEY=<company key>                 # for live runs (calibration works without it)
curl -fsSL https://raw.githubusercontent.com/WhaSukGO/LenaLab/main/scripts/cloud/bringup_lenalab.sh -o bringup.sh
bash bringup.sh
```
The script: checks GPU+Docker → clones LenaLab + the Touchstone spine (siblings) → builds the `vo-gpu-torch`
and `vo-bev` images → downloads nuScenes mini (public, ~4 GB) → preps the BEV + occupancy caches →
writes `.env` → runs an offline calibration smoke test. ~10–15 min the first time.

## ▶️ Running

```bash
cd ~/devel/whasuk/LenaLab && export PYTHONPATH=.:../blueberry_ver2
# non-billed gate (sanity):
python3 -m vo_lab.run_occ_scaffold_calibration
# live (billed) agent run:
python3 -m vo_lab.run_occ_scaffold_implement 0.051
# n runs + collect (sequential on one pod):
scripts/cloud/fanout.sh occ-scaffold 3 0.051
```

## ⚡ Parallel experiments (the real speed-up)
The harness grabs the whole GPU per job, so n runs on **one** VM go sequentially. For true parallelism,
**launch N single-GPU VMs** (cheap — API is free, GPUs are bursty) and run one experiment on each, e.g.
3 VMs → n=3 in the time of one run. (To instead share one multi-GPU VM, the job's `--gpus all` would
need to become `--gpus device=$ID` — a one-line harness tweak; ask if you want it.)

## 💾 Data persistence
The VM is ephemeral. Two options: (a) just re-run the bring-up each session (nuScenes mini is only 4 GB,
~3 min), or (b) attach a **persistent volume** at `~/.cache/vo_lab` so the prepped caches survive restarts.

## 🛑 Teardown (cost control)
**Stop or terminate the instance when idle** — billing is per-second/minute and only while running. A
forgotten A100 is ~$33/day; stopped, it's $0.

## 💰 Cost reminder
RTX 4090 ~$0.40/hr · A100-80 ~$1.39/hr (RunPod) / $2.49/hr (Lambda) · H100 ~$1.99–2.99/hr.
One agent run ≈ $0.40–1.40 · a full domain study (n=3 + scaffold) ≈ $3–10 · the one full-scale
headline ≈ $40–85 one-time. (Live pricing: see `claudedocs/` notes / re-check before a big run.)
