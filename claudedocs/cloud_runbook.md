# Running LenaLab on a cloud GPU — runbook

Goal: run the agent research lab on a rented GPU (bigger VRAM, no contention, parallel experiments).
API cost is company-backed, so the only spend is GPU-hours (≈ **$0.40–1.40/hr**; a full domain study ≈ **$3–10**).

> ✅ **Validated end-to-end (2026-06-20)** on a RunPod RTX 4090 in `local` mode: deps + `claude` CLI
> installed, occupancy cache prepped, calibration gate OPEN, and a live agent run **VERIFIED at
> held-out IoU 0.0701** — total ~$2–3, pod then terminated via the RunPod API to $0.
> **Billing note:** *stopping* a pod still bills storage ($0.20/GB·mo container, $0.07/GB·mo network
> volume); only **terminate** (delete) reaches $0. Terminate the pod when done.

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

## 🪙 Cheaper path — RunPod pod, `local` mode (no nested Docker)
A RunPod *pod* is a container, so instead of nesting Docker we run the lab **directly in the pod** via
`LAB_JOB_MODE=local` (the job runner runs the agent's code as a subprocess on the pod's GPU). Integrity
that matters (held-out split + independent grader) is preserved; only the per-job container boundary is
dropped — fine for a cooperative agent on public data with a rotatable, company-managed key.

**On RunPod:** prefer **Secure Cloud** for keyed runs (rotate the key after); launch a **PyTorch** pod
(cheapest viable: RTX 4090 ~$0.34/hr or A100-80 ~$1.39/hr). Then in the pod:
```bash
# deps: ML libs + Touchstone reqs (claude-agent-sdk) + the claude CLI on PATH
pip install opencv-python-headless nuscenes-devkit pyquaternion shapely einops matplotlib scipy torchvision
git clone https://github.com/WhaSukGO/LenaLab && git clone https://github.com/WhaSukGO/touchstone blueberry_ver2
pip install -r blueberry_ver2/requirements.txt        # claude-agent-sdk, etc.
# install the `claude` CLI on PATH (live runs need it), then:
export LAB_JOB_MODE=local ANTHROPIC_API_KEY=<key> PYTHONPATH=LenaLab:blueberry_ver2
cd LenaLab && python3 scripts/prep_nuscenes_occ.py ~/.cache/vo_lab/nuscenes ~/.cache/vo_lab/occ
python3 -m vo_lab.run_occ_scaffold_calibration
```
> The exact dep set / `claude` CLI install on a fresh pod is best **finalized live** (image variants
> differ) — spin a pod up, share `user@<ip>`, and it can be installed + debugged in real time rather
> than guessed. The `LAB_JOB_MODE` switch itself is in place and verified.

## 💰 Cost reminder
RTX 4090 ~$0.40/hr · A100-80 ~$1.39/hr (RunPod) / $2.49/hr (Lambda) · H100 ~$1.99–2.99/hr.
One agent run ≈ $0.40–1.40 · a full domain study (n=3 + scaffold) ≈ $3–10 · the one full-scale
headline ≈ $40–85 one-time. (Live pricing: see `claudedocs/` notes / re-check before a big run.)
