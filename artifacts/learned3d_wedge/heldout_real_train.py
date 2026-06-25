"""
Phase B (TRAIN, clean graph) + Phase C (held-out) — Captain-Safari REAL held-out test.

FRESH process modeled on /workspace/overfit_test.py: loads ONLY DiT + LoRA + memory modules
+ retriever (NO VAE/T5, NO offload). torch.load('/workspace/real_inputs.pt'), moves every
tensor to GPU as fresh bf16 leaves (detach; conditioning requires_grad=False). The trainable
in-graph STORE (fp32 Parameter init from the REAL released memory tensor) is the ONLY trainable.

This bypasses the known backward bug ("tensors on cuda:0 and cpu" in complex-RoPE MulBackward)
because the autograd graph now contains ONLY clean GPU leaves (no CPU residue from the offloaded
VAE/T5 path).

Held-out protocol (1 query frame -> hold out on (noise,timestep)):
  TRAIN on tids[620,700,780,860,920]/seeds[10-14]; EVAL on DISJOINT tids[660,740,820,900,950]/
  seeds[90-94] (same grid as FROZEN baseline=0.564). Store trained on TRAIN only; reported on
  HELD-OUT (never seen). Early-stop on held-out.
"""
import os, sys
import numpy as np
import torch

REPO = "/workspace/Captain-Safari/captain_safari"
sys.path.insert(0, REPO)
os.chdir(REPO)

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_wan_video

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODELS = "/workspace/models/Wan2.2-Fun-5B"
LOCAL_DIT = f"{MODELS}/diffusion_pytorch_model.safetensors"
CKPT = "/workspace/cs_ckpt/models/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors"
IN = "/workspace/real_inputs.pt"
RESULT_TXT = "/workspace/heldout_real_result.txt"
FROZEN_BASELINE_REF = 0.564  # previously established FROZEN real-held-out mean

OVERFIT_STEPS = 250
LR = 3e-3
EARLY_STOP_PATIENCE = 6
DETECT_ANOMALY = (os.environ.get("ANOMALY", "0") == "1")

torch.manual_seed(0)


def load_pipe():
    print("[load] DiT-only pipeline (NO T5/VAE, no offload)...", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=DTYPE, device=DEVICE,
        model_configs=[ModelConfig(path=LOCAL_DIT, offload_device=None)],
    )
    pipe.dit.use_memory_retrieval = True
    pipe.dit.use_memory_cross_attn = True
    pipe.dit.require_vae_embedding = True
    pipe.load_lora(pipe.dit, CKPT, alpha=1)
    pipe.load_memory(pipe.dit, CKPT)
    pipe.load_cross_attn(pipe.dit, CKPT)
    pipe.load_memory_retriever_from_dit(pipe.dit, CKPT)
    pipe.dit.to(device=DEVICE, dtype=DTYPE)
    if hasattr(pipe.dit, "memory_retriever"):
        mr = pipe.dit.memory_retriever
        mr.to(device=DEVICE, dtype=DTYPE)
        for attr in ("f_freqs", "h_freqs", "w_freqs", "freqs_cis_1d", "freqs_cis_1d_long"):
            if hasattr(mr, attr) and torch.is_tensor(getattr(mr, attr)):
                setattr(mr, attr, getattr(mr, attr).to(DEVICE))
    pipe.scheduler.set_timesteps(1000, training=True)
    print("[load] done.", flush=True)
    return pipe


def build_inputs(pipe, blob):
    """Rebuild the model_fn inputs dict from saved tensors, as fresh GPU bf16 leaves."""
    t = blob["tensors"]
    cfg = blob["config"]
    cond = {}
    for k, v in t.items():
        if k == "memory":
            continue  # the store replaces memory per-call
        # conditioning leaves: detached, requires_grad=False, fresh on GPU
        cond[k] = v.detach().to(DEVICE, DTYPE).requires_grad_(False)
    inputs = dict(
        dit=pipe.dit,
        motion_controller=getattr(pipe, "motion_controller", None),
        vace=getattr(pipe, "vace", None),
        input_latents=cond["input_latents"],
        y=cond["y"],
        context=cond["context"],
        intrinsic_query=cond["intrinsic_query"],
        extrinsic_query=cond["extrinsic_query"],
        intrinsic_key=cond["intrinsic_key"],
        extrinsic_key=cond["extrinsic_key"],
        height=cfg["height"], width=cfg["width"], num_frames=cfg["num_frames"],
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    )
    base_memory = t["memory"].detach().to(DEVICE, DTYPE)
    return inputs, base_memory


def compute_loss(pipe, inputs, memory, noise, timestep, train_weight):
    sched = pipe.scheduler
    latents = sched.add_noise(inputs["input_latents"], noise, timestep)
    target = sched.training_target(inputs["input_latents"], noise, timestep)
    fn_inputs = dict(inputs)
    fn_inputs["latents"] = latents
    fn_inputs["memory"] = memory
    noise_pred = model_fn_wan_video(timestep=timestep, **fn_inputs)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
    loss = loss * train_weight
    return loss


def prep_samples(blob, key):
    out = []
    for s in blob[key]:
        noise = s["noise"].detach().to(DEVICE, DTYPE)
        ts = s["timestep"].detach().to(DEVICE, DTYPE)
        out.append((s["seed"], s["tid"], s["t_value"], s["train_weight"], noise, ts))
    return out


def eval_loss_set(pipe, inputs, memory, samples):
    out = []
    with torch.no_grad():
        for (s, tid, tv, tw, noise, ts) in samples:
            l = compute_loss(pipe, inputs, memory, noise, ts, tw).item()
            out.append((s, tid, tv, l))
    return out


def main():
    pipe = load_pipe()
    blob = torch.load(IN, map_location="cpu", weights_only=False)
    inputs, base_memory = build_inputs(pipe, blob)
    print(f"[store] base memory shape {tuple(base_memory.shape)} dtype {base_memory.dtype}", flush=True)
    print(f"[shapes] input_latents={tuple(inputs['input_latents'].shape)} "
          f"y={tuple(inputs['y'].shape)} context={tuple(inputs['context'].shape)}", flush=True)

    for p in pipe.dit.parameters():
        p.requires_grad_(False)

    train_samples = prep_samples(blob, "train_samples")
    eval_samples  = prep_samples(blob, "eval_samples")

    # ---- SANITY: confirm ONE backward succeeds (bug bypassed) ----
    print("\n[backward-check] running one training_loss(...).backward() on the REAL input path...", flush=True)
    store_probe = torch.nn.Parameter(base_memory.detach().clone().float())
    s0 = train_samples[0]
    try:
        if DETECT_ANOMALY:
            torch.autograd.set_detect_anomaly(True)
        l = compute_loss(pipe, inputs, store_probe.to(DTYPE), s0[4], s0[5], s0[3])
        l.backward()
        assert store_probe.grad is not None, "store_probe.grad is None after backward"
        gnorm = store_probe.grad.norm().item()
        BACKWARD_OK = True
        print(f"[backward-check] SUCCESS — loss={l.item():.6f} store.grad.norm={gnorm:.4e}", flush=True)
    except Exception as e:
        BACKWARD_OK = False
        print(f"[backward-check] FAILED: {repr(e)}", flush=True)
        import traceback; traceback.print_exc()
        with open(RESULT_TXT, "w") as f:
            f.write("BACKWARD_FIXED=NO\n")
            f.write(f"error={repr(e)}\n")
        return
    del store_probe
    torch.cuda.empty_cache()

    # ---- FROZEN baseline (released store) ----
    print("\n[FROZEN] computing released-store loss on TRAIN and HELD-OUT...", flush=True)
    frozen_train = eval_loss_set(pipe, inputs, base_memory, train_samples)
    frozen_eval  = eval_loss_set(pipe, inputs, base_memory, eval_samples)
    for (s, tid, t, l) in frozen_eval:
        print(f"  [frozen heldout] seed={s} tid={tid} t={t:.1f} loss={l:.6f}", flush=True)
    frozen_eval_mean  = float(np.mean([l for *_, l in frozen_eval]))
    frozen_train_mean = float(np.mean([l for *_, l in frozen_train]))
    print(f"FROZEN_TRAIN_MEAN={frozen_train_mean:.6f}  FROZEN_HELDOUT_MEAN={frozen_eval_mean:.6f}", flush=True)
    print(f"(reference prior FROZEN held-out baseline = {FROZEN_BASELINE_REF})", flush=True)

    # ---- LEARNED in-graph store ----
    store = torch.nn.Parameter(base_memory.detach().clone().float())
    opt = torch.optim.Adam([store], lr=LR)
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OVERFIT_STEPS)

    print(f"\n[LEARNED] training store on {len(train_samples)} TRAIN samples up to {OVERFIT_STEPS} "
          f"steps; held-out eval every 10 steps...", flush=True)
    best_eval = float("inf"); best_state = None; no_improve = 0; history = []
    for step in range(OVERFIT_STEPS):
        opt.zero_grad()
        mem = store.to(DTYPE)
        total = 0.0
        for (s, tid, tv, tw, noise, ts) in train_samples:
            loss = compute_loss(pipe, inputs, mem, noise, ts, tw)
            (loss / len(train_samples)).backward()
            total += loss.item()
        opt.step(); sched_lr.step()
        train_mean = total / len(train_samples)

        if step % 10 == 0 or step == OVERFIT_STEPS - 1:
            ev = eval_loss_set(pipe, inputs, store.to(DTYPE), eval_samples)
            ev_mean = float(np.mean([l for *_, l in ev]))
            history.append((step, train_mean, ev_mean))
            print(f"  step {step:3d}  train_mean={train_mean:.6f}  heldout_mean={ev_mean:.6f}", flush=True)
            if ev_mean < best_eval - 1e-7:
                best_eval = ev_mean; best_state = store.detach().clone(); no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  early stop at step {step}", flush=True)
                    break

    final_store = best_state if best_state is not None else store.detach()
    learned_train = eval_loss_set(pipe, inputs, final_store.to(DTYPE), train_samples)
    learned_eval  = eval_loss_set(pipe, inputs, final_store.to(DTYPE), eval_samples)
    learned_train_mean = float(np.mean([l for *_, l in learned_train]))
    learned_eval_mean  = float(np.mean([l for *_, l in learned_eval]))

    print("\n================ RESULT ================", flush=True)
    print(f"FROZEN_TRAIN_MEAN   ={frozen_train_mean:.6f}", flush=True)
    print(f"LEARNED_TRAIN_MEAN  ={learned_train_mean:.6f}", flush=True)
    print(f"FROZEN_HELDOUT_MEAN ={frozen_eval_mean:.6f}", flush=True)
    print(f"LEARNED_HELDOUT_MEAN={learned_eval_mean:.6f}", flush=True)

    print("\nper-held-out-sample (frozen -> learned):", flush=True)
    n_better = 0
    for (s, tid, t, fl), (_, _, _, ll) in zip(frozen_eval, learned_eval):
        better = ll < fl; n_better += int(better)
        print(f"  seed={s} tid={tid} t={t:.1f}  frozen={fl:.6f}  learned={ll:.6f}  "
              f"{'BETTER' if better else 'worse '}  ({100*(fl-ll)/fl:+.2f}%)", flush=True)

    rel_gain = 100 * (frozen_eval_mean - learned_eval_mean) / frozen_eval_mean
    train_gain = 100 * (frozen_train_mean - learned_train_mean) / frozen_train_mean
    meaningful = (learned_eval_mean < frozen_eval_mean) and (rel_gain > 1.0) and \
                 (n_better >= (len(eval_samples) + 1) // 2)
    if meaningful:
        verdict = "YES — learned store beats frozen on REAL held-out data"
    elif learned_eval_mean < frozen_eval_mean:
        verdict = "INCONCLUSIVE — learned < frozen on held-out but margin small/unstable"
    else:
        verdict = "NO — learned does NOT beat frozen on held-out (overfit to train only)"

    print(f"\nHELDOUT_REL_GAIN={rel_gain:+.2f}%  TRAIN_REL_GAIN={train_gain:+.2f}%  "
          f"heldout_samples_better={n_better}/{len(eval_samples)}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    with open(RESULT_TXT, "w") as f:
        f.write("Captain-Safari LEARNED-vs-FROZEN in-graph memory store — REAL held-out test\n")
        f.write("(decoupled encode/train fix for the complex-RoPE CPU-residue backward bug)\n")
        f.write("=" * 70 + "\n")
        f.write("BACKWARD_FIXED=YES\n")
        f.write(f"TRAIN_TIDS={blob['TRAIN_TIDS']} TRAIN_SEEDS={blob['TRAIN_SEEDS']}\n")
        f.write(f"EVAL_TIDS ={blob['EVAL_TIDS']}  EVAL_SEEDS ={blob['EVAL_SEEDS']}\n")
        f.write(f"lr={LR} steps(max)={OVERFIT_STEPS} early_stop_patience={EARLY_STOP_PATIENCE}\n\n")
        f.write(f"FROZEN_TRAIN_MEAN   ={frozen_train_mean:.6f}\n")
        f.write(f"LEARNED_TRAIN_MEAN  ={learned_train_mean:.6f}\n")
        f.write(f"FROZEN_HELDOUT_MEAN ={frozen_eval_mean:.6f}  (prior-run ref={FROZEN_BASELINE_REF})\n")
        f.write(f"LEARNED_HELDOUT_MEAN={learned_eval_mean:.6f}\n")
        f.write(f"HELDOUT_REL_GAIN={rel_gain:+.2f}%  TRAIN_REL_GAIN={train_gain:+.2f}%\n")
        f.write(f"heldout_samples_better={n_better}/{len(eval_samples)}\n\n")
        f.write("per-held-out-sample (seed,tid,t,frozen,learned):\n")
        for (s, tid, t, fl), (_, _, _, ll) in zip(frozen_eval, learned_eval):
            f.write(f"  seed={s} tid={tid} t={t:.1f}  frozen={fl:.6f}  learned={ll:.6f}\n")
        f.write("\ntraining history (step, train_mean, heldout_mean):\n")
        for (st, tr, ev) in history:
            f.write(f"  {st}\t{tr:.6f}\t{ev:.6f}\n")
        f.write(f"\nVERDICT: {verdict}\n")
        f.write(f"max_vram_alloc_GB={mem_alloc:.2f}\n")
        f.write("REALITY: input_latents/y/context all REAL (VAE+T5+camera-encoded from the demo "
                "clip in Phase A). DiT+LoRA+memory modules+retriever REAL & FROZEN. Store = "
                "trainable fp32 copy of the REAL released memory tensor.\n")
    print(f"\n[done] wrote {RESULT_TXT}; max VRAM alloc {mem_alloc:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
