"""
Task 3 + Task 4: Captain-Safari trainable-memory-store overfit experiment.

Task 3: Compute the FROZEN-memory baseline diffusion (flow-matching) loss from the
        real released model on ONE demo clip, averaged over several fixed (noise,timestep)
        pairs.
Task 4: Make the memory a TRAINABLE in-graph store (Parameter init = base memory clone),
        overfit it on the SAME clip at a FIXED (noise, timestep) for ~80 steps, and show
        the loss drops clearly below the frozen baseline.

REALITY NOTE (blocker found + how handled):
  The demo assets DO NOT contain pre-encoded prompt_embedding / video_latent / image latents
  (the metadata.csv has no such columns; data-slices/ holds only an .mp4). The Wan2.2 VAE and
  UMT5 T5 weights are NOT present locally and downloading them from ModelScope is forbidden/slow.
  Therefore we cannot VAE-encode the real video latent. We instead use a FIXED synthetic target
  latent + fixed synthetic prompt-context of the correct shapes. Everything that the experiment
  is actually about is REAL: the released 5B DiT, the loaded LoRA, the real memory_emb /
  memory_cross_attn / memory_retriever weights, the REAL pre-baked memory tensor, and the REAL
  camera intrinsics/extrinsics. The store learns to steer the real DiT's noise_pred toward a
  fixed target at a fixed (noise,timestep). This is a faithful test of:
  "a learned in-graph memory store reduces the real denoising loss."
"""
import os, sys
import numpy as np
import pandas as pd
import torch

REPO = "/workspace/Captain-Safari/captain_safari"
sys.path.insert(0, REPO)
os.chdir(REPO)

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_wan_video

DEVICE = "cuda"
DTYPE = torch.bfloat16
LOCAL_DIT = "/workspace/models/Wan2.2-Fun-5B/diffusion_pytorch_model.safetensors"
CKPT = "/workspace/cs_ckpt/models/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors"
ASSETS = "/workspace/cs_assets"
RESULT_TXT = "/workspace/overfit_result.txt"

NUM_KEY = 4          # model expects 4 key frames (memory viewed as [B,4,4*782,1024])
TOK_PER_FRAME = 782
MEM_DIM = 1024
N_BASELINE_SEEDS = 8 # fixed seeds/timesteps averaged for the frozen baseline
OVERFIT_STEPS = 100
LR = 5e-3
OVERFIT_TID = 950   # scheduler index -> sigma~0.21 (low noise, memory has leverage)

torch.manual_seed(0)


def load_pipe():
    print("[load] building pipeline with LOCAL DiT only (no T5/VAE)...", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=DTYPE,
        device=DEVICE,
        model_configs=[
            ModelConfig(path=LOCAL_DIT, offload_device=None),
        ],
    )
    pipe.dit.use_memory_retrieval = True
    pipe.dit.use_memory_cross_attn = True
    pipe.dit.require_vae_embedding = True   # so y (52ch) is concatenated -> 100ch DiT input
    pipe.load_lora(pipe.dit, CKPT, alpha=1)
    pipe.load_memory(pipe.dit, CKPT)
    pipe.load_cross_attn(pipe.dit, CKPT)
    pipe.load_memory_retriever_from_dit(pipe.dit, CKPT)
    pipe.dit.to(device=DEVICE, dtype=DTYPE)
    if hasattr(pipe.dit, "memory_retriever"):
        pipe.dit.memory_retriever.to(device=DEVICE, dtype=DTYPE)
    # training scheduler so add_noise/training_target/timesteps are set up
    pipe.scheduler.set_timesteps(1000, training=True)
    print("[load] done.", flush=True)
    return pipe


def build_clip_inputs(pipe):
    row = pd.read_csv(os.path.join(ASSETS, "metadata.csv")).iloc[0]
    P = lambda k: os.path.join(ASSETS, row[k])

    # --- real pre-baked memory: npy (16,4,782,1024) -> first NUM_KEY frames ---
    mem = np.load(P("memory"), allow_pickle=True)          # (16, 4, 782, 1024)
    mem = mem[:NUM_KEY]                                     # (4, 4, 782, 1024)
    base_memory = torch.tensor(mem, dtype=DTYPE, device=DEVICE)
    base_memory = base_memory.reshape(1, NUM_KEY * 4 * TOK_PER_FRAME, MEM_DIM)  # [1, 4*4*782, 1024]

    # --- real camera params ---
    ex_q = torch.tensor(np.load(P("extrinsic_query")), dtype=DTYPE, device=DEVICE)  # (3,4)
    in_q = torch.tensor(np.load(P("intrinsic_query")), dtype=DTYPE, device=DEVICE)  # (3,3)
    ex_k = torch.tensor(np.load(P("extrinsic_key")), dtype=DTYPE, device=DEVICE)    # (16,3,4)
    in_k = torch.tensor(np.load(P("intrinsic_key")), dtype=DTYPE, device=DEVICE)    # (16,3,3)
    ex_k = ex_k[:NUM_KEY]   # (4,3,4)
    in_k = in_k[:NUM_KEY]   # (4,3,3)

    # model_fn memory block expects:
    #   extrinsic_query: [B,3,4]  (indexed [b:b+1])   -> add batch
    #   intrinsic_query: [B,3,3]
    #   extrinsic_key:   [B,T,1,3,4] (.squeeze(2)->[B,T,3,4])
    #   intrinsic_key:   [B,T,1,3,3]
    extrinsic_query = ex_q.unsqueeze(0)                       # [1,3,4]
    intrinsic_query = in_q.unsqueeze(0)                       # [1,3,3]
    extrinsic_key = ex_k.unsqueeze(0).unsqueeze(2)            # [1,4,1,3,4]
    intrinsic_key = in_k.unsqueeze(0).unsqueeze(2)            # [1,4,1,3,3]

    # --- FIXED synthetic target latent + prompt context (see REALITY NOTE) ---
    # latent shape [1,48,T,H,W]; pick a small but real-ish video latent size.
    g = torch.Generator(device="cpu").manual_seed(1234)
    Tl, Hl, Wl = 3, 30, 52
    input_latents = torch.randn(1, 48, Tl, Hl, Wl, generator=g).to(DEVICE, DTYPE)
    # y: image/camera conditioning, in_dim(100) - latent(48) = 52 channels, fixed.
    y = torch.randn(1, 52, Tl, Hl, Wl, generator=g).to(DEVICE, DTYPE) * 0.1
    # context: raw prompt embedding shape [1,512,4096] (DiT.text_embedding projects -> dim)
    context = torch.randn(1, 512, 4096, generator=g).to(DEVICE, DTYPE) * 0.1

    inputs = dict(
        dit=pipe.dit,
        motion_controller=getattr(pipe, "motion_controller", None),
        vace=getattr(pipe, "vace", None),
        input_latents=input_latents,
        y=y,
        context=context,
        intrinsic_query=intrinsic_query,
        extrinsic_query=extrinsic_query,
        intrinsic_key=intrinsic_key,
        extrinsic_key=extrinsic_key,
        height=Hl * 16, width=Wl * 16, num_frames=1 + (Tl - 1) * 4,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )
    return inputs, base_memory


def compute_loss(pipe, inputs, memory, noise, timestep):
    """Replicates WanVideoPipeline.training_loss but with explicit fixed noise+timestep
    and an explicit memory tensor (so we can swap frozen base vs trainable store)."""
    sched = pipe.scheduler
    latents = sched.add_noise(inputs["input_latents"], noise, timestep)
    target = sched.training_target(inputs["input_latents"], noise, timestep)
    fn_inputs = dict(inputs)
    fn_inputs["latents"] = latents
    fn_inputs["memory"] = memory
    noise_pred = model_fn_wan_video(timestep=timestep, **fn_inputs)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
    loss = loss * sched.training_weight(timestep)
    return loss


def fixed_noise_timestep(pipe, inputs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(inputs["input_latents"].shape, generator=g).to(DEVICE, DTYPE)
    # Spread baseline timesteps across the meaningful (high-weight) part of the
    # shifted schedule: indices ~[600, 980) cover sigma ~0.77 down to ~0.1.
    tids = [620, 680, 740, 800, 860, 900, 940, 970]
    tid = tids[seed % len(tids)]
    timestep = pipe.scheduler.timesteps[tid:tid + 1].to(dtype=DTYPE, device=DEVICE)
    return noise, timestep


def main():
    pipe = load_pipe()
    inputs, base_memory = build_clip_inputs(pipe)
    print(f"[shapes] memory={tuple(base_memory.shape)} latent={tuple(inputs['input_latents'].shape)}", flush=True)

    # freeze ALL DiT params (grad still flows to the memory input tensor)
    for p in pipe.dit.parameters():
        p.requires_grad_(False)

    # ---------------- Task 3: FROZEN baseline ----------------
    print("\n[Task3] computing frozen-memory baseline loss over fixed seeds...", flush=True)
    base_losses = []
    with torch.no_grad():
        for s in range(N_BASELINE_SEEDS):
            noise, ts = fixed_noise_timestep(pipe, inputs, seed=s)
            l = compute_loss(pipe, inputs, base_memory, noise, ts)
            base_losses.append(l.item())
            print(f"  seed={s} t={ts.item():.1f} loss={l.item():.6f}", flush=True)
    baseline = float(np.mean(base_losses))
    print(f"BASELINE_LOSS={baseline:.4f}", flush=True)

    # ---------------- Task 4: trainable in-graph store ----------------
    # FIXED (noise, timestep) for the overfit: low-noise index where the memory
    # conditioning has the most leverage on the velocity prediction.
    of_noise = torch.randn(inputs["input_latents"].shape,
                           generator=torch.Generator(device="cpu").manual_seed(0)).to(DEVICE, DTYPE)
    of_ts = pipe.scheduler.timesteps[OVERFIT_TID:OVERFIT_TID + 1].to(dtype=DTYPE, device=DEVICE)
    frozen_at_fixed = compute_loss(pipe, inputs, base_memory, of_noise, of_ts).item()
    print(f"\n[Task4] frozen loss at the FIXED overfit (noise,t={of_ts.item():.1f}): {frozen_at_fixed:.6f}", flush=True)

    # Trainable in-graph store = fp32 Parameter init = base memory clone. Optimised
    # stably (fp32 store) at the fixed (noise,t); cast to bf16 in-graph each step.
    store = torch.nn.Parameter(base_memory.detach().clone().float())
    opt = torch.optim.Adam([store], lr=LR)
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OVERFIT_STEPS)

    print(f"[Task4] overfitting in-graph store for {OVERFIT_STEPS} steps at fixed (noise,t)...", flush=True)
    final = None
    for step in range(OVERFIT_STEPS):
        opt.zero_grad()
        mem = store.to(DTYPE)                      # in-graph store -> memory
        loss = compute_loss(pipe, inputs, mem, of_noise, of_ts)
        loss.backward()
        opt.step()
        sched_lr.step()
        final = loss.item()
        if step % 10 == 0 or step == OVERFIT_STEPS - 1:
            print(f"  step {step:3d}  loss={final:.6f}", flush=True)

    print(f"\nBASELINE_LOSS={baseline:.4f}", flush=True)
    print(f"OVERFIT_FINAL_LOSS={final:.4f}", flush=True)
    beat = final < frozen_at_fixed and final < baseline
    print(f"frozen_at_fixed_pair={frozen_at_fixed:.4f}", flush=True)
    print(f"LEARNED_STORE_BEAT_BASELINE={beat}", flush=True)

    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    with open(RESULT_TXT, "w") as f:
        f.write(f"BASELINE_LOSS={baseline:.4f}\n")
        f.write(f"OVERFIT_FINAL_LOSS={final:.4f}\n")
        f.write(f"frozen_at_fixed_pair={frozen_at_fixed:.4f}\n")
        f.write(f"LEARNED_STORE_BEAT_BASELINE={beat}\n")
        f.write(f"baseline_per_seed={['%.4f'%x for x in base_losses]}\n")
        f.write(f"overfit_steps={OVERFIT_STEPS} lr={LR}\n")
        f.write(f"max_vram_alloc_GB={mem_alloc:.2f}\n")
        f.write("NOTE: synthetic fixed target latent + prompt context (VAE/T5 weights not "
                "local, no pre-encoded latents in assets). DiT+LoRA+memory modules+memory "
                "tensor+camera params are all REAL.\n")
    print(f"[done] wrote {RESULT_TXT}; max VRAM alloc {mem_alloc:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
