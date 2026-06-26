"""
Phase B/C — Captain-Safari CROSS-VIEWPOINT held-out test (DiT-only, clean graph).

THE QUESTION: does a LEARNED in-graph memory store beat the FROZEN released store across
UNSEEN CAMERA VIEWPOINTS?

Loads /workspace/multi_vp_inputs.pt (per-viewpoint input_latents + query cameras; shared
y/context/memory/key-cameras). Fresh DiT-only process (no VAE/T5) -> clean GPU bf16 leaves
(bypasses the complex-RoPE CPU-residue backward bug). The trainable in-graph STORE is an fp32
copy of the released memory; DiT FROZEN.

VIEWPOINT SPLIT (disjoint by camera):
  V_train = views {2, 6, 10, 14, 18}     (18 is OUTSIDE the key window)
  V_test  = views {4, 8, 12, 16, 19}     (16,19 are OUTSIDE the key window)
The store is trained ONLY on V_train viewpoints (multiple noise seeds x timesteps each),
early-stopped on V_test, then we report FROZEN vs LEARNED denoising loss on the HELD-OUT
V_test viewpoints. Also reports the OUTSIDE-window-only subset (16,19) to separate memory
adaptation from genuine viewpoint generalization.
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
IN = "/workspace/multi_vp_inputs.pt"
RESULT_TXT = "/workspace/crossview_result.txt"

V_TRAIN = [2, 6, 10, 14, 18]
V_TEST  = [4, 8, 12, 16, 19]

# (noise,timestep) grid applied to EVERY viewpoint (averaged over, so loss is viewpoint-driven)
# 3 points (mid timesteps) keeps 5views*3=15 fwd+bwd/step affordable on the 5B DiT.
TIDS  = [660, 780, 900]
SEEDS = [11, 12, 13]

OVERFIT_STEPS = 80
LR = 3e-3
EARLY_STOP_PATIENCE = 4
EVAL_EVERY = 5

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


def to_leaf(v):
    return v.detach().to(DEVICE, DTYPE).requires_grad_(False)


def build_views(pipe, blob):
    """Build a per-viewpoint model_fn inputs dict (shared y/context/keys + per-view latents/query)."""
    st = blob["shared_tensors"]
    cfg = blob["config"]
    shared = {
        "y": to_leaf(st["y"]),
        "context": to_leaf(st["context"]),
        "intrinsic_key": to_leaf(st["intrinsic_key"]),
        "extrinsic_key": to_leaf(st["extrinsic_key"]),
    }
    base_memory = st["memory"].detach().to(DEVICE, DTYPE)  # released store (frozen ref / init)
    views = {}
    for v in blob["views"]:
        idx = v["view_idx"]
        inputs = dict(
            dit=pipe.dit,
            motion_controller=getattr(pipe, "motion_controller", None),
            vace=getattr(pipe, "vace", None),
            input_latents=to_leaf(v["input_latents"]),
            y=shared["y"],
            context=shared["context"],
            intrinsic_query=to_leaf(v["intrinsic_query"]),
            extrinsic_query=to_leaf(v["extrinsic_query"]),
            intrinsic_key=shared["intrinsic_key"],
            extrinsic_key=shared["extrinsic_key"],
            height=cfg["height"], width=cfg["width"], num_frames=cfg["num_frames"],
            use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False,
        )
        views[idx] = {"inputs": inputs, "outside": v["outside_key_window"],
                      "mp4_window": v["mp4_window"]}
    return views, base_memory


def make_grid(pipe, shape):
    """Fixed (noise, timestep value, train_weight) grid, shared across viewpoints."""
    sched = pipe.scheduler
    grid = []
    for s, tid in zip(SEEDS, TIDS):
        g = torch.Generator(device="cpu").manual_seed(s)
        noise = torch.randn(shape, generator=g).to(DEVICE, DTYPE)
        ts = sched.timesteps[tid:tid + 1].detach().to(DEVICE, DTYPE)
        tw = float(sched.training_weight(ts))
        grid.append((s, tid, float(ts.item()), tw, noise, ts))
    return grid


def compute_loss(pipe, inputs, memory, noise, timestep, train_weight):
    sched = pipe.scheduler
    latents = sched.add_noise(inputs["input_latents"], noise, timestep)
    target = sched.training_target(inputs["input_latents"], noise, timestep)
    fn_inputs = dict(inputs)
    fn_inputs["latents"] = latents
    fn_inputs["memory"] = memory
    noise_pred = model_fn_wan_video(timestep=timestep, **fn_inputs)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * train_weight
    return loss


def view_loss(pipe, vinfo, memory, grid):
    """Mean denoising loss for ONE viewpoint over the full (noise,timestep) grid."""
    ls = []
    for (s, tid, tv, tw, noise, ts) in grid:
        ls.append(compute_loss(pipe, vinfo["inputs"], memory, noise, ts, tw).item())
    return float(np.mean(ls))


def eval_set(pipe, views, memory, grid, idxs):
    with torch.no_grad():
        return {i: view_loss(pipe, views[i], memory, grid) for i in idxs}


def main():
    pipe = load_pipe()
    blob = torch.load(IN, map_location="cpu", weights_only=False)
    views, base_memory = build_views(pipe, blob)
    print(f"[store] base memory {tuple(base_memory.shape)} {base_memory.dtype}", flush=True)
    any_idx = next(iter(views))
    shape = tuple(views[any_idx]["inputs"]["input_latents"].shape)
    print(f"[shapes] input_latents={shape}", flush=True)
    grid = make_grid(pipe, shape)

    for p in pipe.dit.parameters():
        p.requires_grad_(False)

    have = set(views.keys())
    v_train = [i for i in V_TRAIN if i in have]
    v_test  = [i for i in V_TEST if i in have]
    v_test_outside = [i for i in v_test if views[i]["outside"]]
    print(f"[split] V_train={v_train}  V_test={v_test}  V_test_outside_window={v_test_outside}", flush=True)
    assert set(v_train).isdisjoint(set(v_test)), "TRAIN/TEST viewpoints overlap!"

    # ---- SANITY: one backward on the REAL cross-view path ----
    print("\n[backward-check] one training_loss(...).backward() on the cross-view input path...", flush=True)
    store_probe = torch.nn.Parameter(base_memory.detach().clone().float())
    g0 = grid[0]; v0 = views[v_train[0]]
    try:
        l = compute_loss(pipe, v0["inputs"], store_probe.to(DTYPE), g0[4], g0[5], g0[3])
        l.backward()
        assert store_probe.grad is not None
        gnorm = store_probe.grad.norm().item()
        BACKWARD_OK = True
        print(f"[backward-check] SUCCESS — loss={l.item():.6f} store.grad.norm={gnorm:.4e}", flush=True)
    except Exception as e:
        BACKWARD_OK = False
        print(f"[backward-check] FAILED: {repr(e)}", flush=True)
        import traceback; traceback.print_exc()
        with open(RESULT_TXT, "w") as f:
            f.write("BACKWARD_OK=NO\n"); f.write(f"error={repr(e)}\n")
        return
    del store_probe; torch.cuda.empty_cache()

    # ---- FROZEN baseline (released store) on every viewpoint ----
    print("\n[FROZEN] released-store loss per viewpoint...", flush=True)
    frozen_train = eval_set(pipe, views, base_memory, grid, v_train)
    frozen_test  = eval_set(pipe, views, base_memory, grid, v_test)
    for i in v_test:
        print(f"  [frozen] view={i} outside={views[i]['outside']} loss={frozen_test[i]:.6f}", flush=True)
    frozen_train_mean = float(np.mean(list(frozen_train.values())))
    frozen_test_mean  = float(np.mean(list(frozen_test.values())))
    print(f"FROZEN_TRAIN_MEAN={frozen_train_mean:.6f}  FROZEN_TEST_MEAN={frozen_test_mean:.6f}", flush=True)

    # ---- LEARNED store (trained on V_train only) ----
    store = torch.nn.Parameter(base_memory.detach().clone().float())
    opt = torch.optim.Adam([store], lr=LR)
    sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=OVERFIT_STEPS)

    print(f"\n[LEARNED] training store on V_train={v_train} up to {OVERFIT_STEPS} steps; "
          f"held-out V_test eval every 10 steps...", flush=True)
    best_eval = float("inf"); best_state = None; no_improve = 0; history = []
    n_terms = len(v_train) * len(grid)
    import time as _time
    _t0 = _time.time()
    for step in range(OVERFIT_STEPS):
        opt.zero_grad()
        mem = store.to(DTYPE)
        total = 0.0
        for i in v_train:
            for (s, tid, tv, tw, noise, ts) in grid:
                loss = compute_loss(pipe, views[i]["inputs"], mem, noise, ts, tw)
                (loss / n_terms).backward()
                total += loss.item()
        opt.step(); sched_lr.step()
        train_mean = total / n_terms
        print(f"  [tick] step {step:3d} train_mean={train_mean:.6f} "
              f"({(_time.time()-_t0)/(step+1):.1f}s/step)", flush=True)

        if step % EVAL_EVERY == 0 or step == OVERFIT_STEPS - 1:
            ev = eval_set(pipe, views, store.to(DTYPE), grid, v_test)
            ev_mean = float(np.mean(list(ev.values())))
            history.append((step, train_mean, ev_mean))
            print(f"  step {step:3d}  train_mean={train_mean:.6f}  heldout_test_mean={ev_mean:.6f}", flush=True)
            if ev_mean < best_eval - 1e-7:
                best_eval = ev_mean; best_state = store.detach().clone(); no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  early stop at step {step}", flush=True)
                    break

    final_store = best_state if best_state is not None else store.detach()
    learned_train = eval_set(pipe, views, final_store.to(DTYPE), grid, v_train)
    learned_test  = eval_set(pipe, views, final_store.to(DTYPE), grid, v_test)
    learned_train_mean = float(np.mean(list(learned_train.values())))
    learned_test_mean  = float(np.mean(list(learned_test.values())))

    # ---- report ----
    print("\n================ CROSS-VIEW RESULT ================", flush=True)
    print(f"FROZEN_TRAIN_MEAN  ={frozen_train_mean:.6f}", flush=True)
    print(f"LEARNED_TRAIN_MEAN ={learned_train_mean:.6f}", flush=True)
    print(f"FROZEN_TEST_MEAN   ={frozen_test_mean:.6f}", flush=True)
    print(f"LEARNED_TEST_MEAN  ={learned_test_mean:.6f}", flush=True)

    print("\nper-HELD-OUT-VIEWPOINT (frozen -> learned):", flush=True)
    n_better = 0
    rows = []
    for i in v_test:
        fl = frozen_test[i]; ll = learned_test[i]
        better = ll < fl; n_better += int(better)
        rel = 100 * (fl - ll) / fl
        tag = "OUTSIDE-key" if views[i]["outside"] else "in-key"
        rows.append((i, views[i]["outside"], fl, ll, rel, better))
        print(f"  view={i:2d} [{tag:11s}] frozen={fl:.6f} learned={ll:.6f} "
              f"{'BETTER' if better else 'worse '} ({rel:+.2f}%)", flush=True)

    rel_gain = 100 * (frozen_test_mean - learned_test_mean) / frozen_test_mean
    train_gain = 100 * (frozen_train_mean - learned_train_mean) / frozen_train_mean

    # outside-window-only subset
    out_line = ""
    if v_test_outside:
        fo = float(np.mean([frozen_test[i] for i in v_test_outside]))
        lo = float(np.mean([learned_test[i] for i in v_test_outside]))
        og = 100 * (fo - lo) / fo
        out_line = (f"OUTSIDE-WINDOW-ONLY V_test={v_test_outside}: "
                    f"frozen={fo:.6f} learned={lo:.6f} rel_gain={og:+.2f}%")
        print(f"\n{out_line}", flush=True)

    # verdict
    if learned_test_mean < frozen_test_mean and rel_gain > 1.0 and n_better >= (len(v_test) + 1) // 2:
        if rel_gain >= 3.0 and n_better >= len(v_test) - 1:
            verdict = "STRONG GO — learned store clearly beats frozen across held-out viewpoints"
        else:
            verdict = "WEAK GO — learned store modestly but consistently beats frozen across held-out viewpoints"
    elif learned_test_mean < frozen_test_mean:
        verdict = "WEAK GO / borderline — learned < frozen on held-out viewpoints but small/few-better"
    else:
        verdict = "NO-GO — learned does NOT beat frozen on held-out viewpoints"

    print(f"\nTEST_REL_GAIN={rel_gain:+.2f}%  TRAIN_REL_GAIN={train_gain:+.2f}%  "
          f"heldout_views_better={n_better}/{len(v_test)}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

    mem_alloc = torch.cuda.max_memory_allocated() / 1e9
    with open(RESULT_TXT, "w") as f:
        f.write("Captain-Safari LEARNED-vs-FROZEN in-graph memory store — CROSS-VIEWPOINT held-out test\n")
        f.write("Held-out variable = CAMERA VIEWPOINT (query pose drives memory retrieval).\n")
        f.write("=" * 78 + "\n")
        f.write(f"BACKWARD_OK={'YES' if BACKWARD_OK else 'NO'}\n")
        f.write(f"V_train(views)={v_train}\nV_test(views) ={v_test}\n")
        f.write(f"V_test OUTSIDE key window (genuinely new viewpoints)={v_test_outside}\n")
        f.write(f"grid TIDS={TIDS} SEEDS={SEEDS}\n")
        f.write(f"lr={LR} steps(max)={OVERFIT_STEPS} early_stop_patience={EARLY_STOP_PATIENCE}\n\n")
        f.write(f"FROZEN_TRAIN_MEAN  ={frozen_train_mean:.6f}\n")
        f.write(f"LEARNED_TRAIN_MEAN ={learned_train_mean:.6f}\n")
        f.write(f"FROZEN_TEST_MEAN   ={frozen_test_mean:.6f}\n")
        f.write(f"LEARNED_TEST_MEAN  ={learned_test_mean:.6f}\n")
        f.write(f"TEST_REL_GAIN={rel_gain:+.2f}%  TRAIN_REL_GAIN={train_gain:+.2f}%\n")
        f.write(f"heldout_views_better={n_better}/{len(v_test)}\n")
        if out_line:
            f.write(out_line + "\n")
        f.write("\nper-held-out-viewpoint (view, outside_key, frozen, learned, rel_gain%, better):\n")
        for (i, outside, fl, ll, rel, better) in rows:
            f.write(f"  view={i:2d} outside={outside} frozen={fl:.6f} learned={ll:.6f} "
                    f"rel={rel:+.2f}% {'BETTER' if better else 'worse'}\n")
        f.write("\ntraining history (step, train_mean, heldout_test_mean):\n")
        for (st, tr, ev) in history:
            f.write(f"  {st}\t{tr:.6f}\t{ev:.6f}\n")
        f.write(f"\nVERDICT: {verdict}\n")
        f.write(f"max_vram_alloc_GB={mem_alloc:.2f}\n")
        f.write("REALITY: per-viewpoint input_latents = VAE-encoded real mp4 window ending at that "
                "viewpoint's frame; query camera = clip_0_20 pose at that index. Shared released "
                "y/context/memory/key-cameras. DiT+LoRA+memory+retriever REAL & FROZEN. "
                "Store = trainable fp32 copy of released memory.\n")
    print(f"\n[done] wrote {RESULT_TXT}; max VRAM {mem_alloc:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
