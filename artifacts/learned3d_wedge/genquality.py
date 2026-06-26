"""
Decisive generation-QUALITY experiment for Captain-Safari learned vs frozen memory store.

PROPER generation: classifier-free guidance ON (cond = cached prompt context,
uncond = empty-prompt context). ~50 denoise steps. Image-quality metrics
(LPIPS / PSNR / SSIM) of the generated query frame vs the REAL frame at that camera
(VAE-decode of that viewpoint's input_latents, last frame).

Viewpoints:
  IN-WINDOW (view_idx <= KEY_WINDOW_MAX=15):  4, 8, 12
  HELD-OUT-OUTSIDE (outside_key_window=True): 16, 18, 19   (view 20 is NOT in the data)

For each: generate with FROZEN store and LEARNED store (same seed, CFG on), decode the
query (last) frame. Sanity control first: regenerate an in-window view with the FROZEN
store + CFG and check it looks like the real frame (LPIPS).
"""
import os, sys, time, json
import numpy as np
import torch

REPO = "/workspace/Captain-Safari/captain_safari"
sys.path.insert(0, REPO)
os.chdir(REPO)

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, model_fn_wan_video
from diffsynth.models import ModelManager

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODELS = "/workspace/models/Wan2.2-Fun-5B"
LOCAL_DIT = f"{MODELS}/diffusion_pytorch_model.safetensors"
LOCAL_VAE = f"{MODELS}/Wan2.2_VAE.pth"
CKPT = "/workspace/cs_ckpt/models/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors"
IN = "/workspace/multi_vp_inputs.pt"
LEARNED = "/workspace/gen_demo/learned_store.pt"
UNCOND = "/workspace/genq/uncond_context.pt"
OUT = "/workspace/genq"
os.makedirs(OUT, exist_ok=True)

KEY_WINDOW_MAX = 15
INWINDOW = [4, 8, 12]
HELDOUT = [16, 18, 19]   # view 20 absent from data
PNG_VIEWS = [4, 16, 19]  # one in-window, two held-out
SANITY_VIEW = 4

N_INFER_STEPS = 50
SIGMA_SHIFT = 5.0
GEN_SEED = 1234
CFG_SCALE = 6.0


def load_pipe():
    print("[load] DiT-only pipeline...", flush=True)
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
    for p in pipe.dit.parameters():
        p.requires_grad_(False)
    print("[load] DiT done.", flush=True)
    return pipe


def load_vae_into(pipe):
    print("[load] VAE...", flush=True)
    mm = ModelManager()
    mm.load_model(LOCAL_VAE, device=DEVICE, torch_dtype=DTYPE)
    pipe.vae = mm.fetch_model("wan_video_vae")
    assert pipe.vae is not None, "VAE failed to load"
    pipe.vae.to(device=DEVICE)
    print("[load] VAE done.", flush=True)
    return pipe


def to_leaf(v):
    return v.detach().to(DEVICE, DTYPE).requires_grad_(False)


def build_views(pipe, blob):
    st = blob["shared_tensors"]
    cfg = blob["config"]
    shared = {
        "y": to_leaf(st["y"]),
        "context": to_leaf(st["context"]),            # cached COND context
        "intrinsic_key": to_leaf(st["intrinsic_key"]),
        "extrinsic_key": to_leaf(st["extrinsic_key"]),
    }
    base_memory = st["memory"].detach().to(DEVICE, DTYPE)
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


@torch.no_grad()
def generate_cfg(pipe, vinputs, store, uncond_context, shape, seed, n_steps, cfg_scale):
    """Full flow-matching denoise with classifier-free guidance.
    cond context = vinputs['context'] (cached); uncond context = uncond_context."""
    sched = pipe.scheduler
    sched.set_timesteps(n_steps, denoising_strength=1.0, shift=SIGMA_SHIFT)
    g = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(shape, generator=g).to(DEVICE, DTYPE)

    base = dict(vinputs)
    base.pop("input_latents", None)
    base["memory"] = store.to(DTYPE)

    cond_ctx = base["context"]
    uncond_ctx = uncond_context.to(DEVICE, DTYPE)

    for i, t in enumerate(sched.timesteps):
        ts = t.unsqueeze(0).to(dtype=DTYPE, device=DEVICE)
        # cond
        base["context"] = cond_ctx
        np_posi = model_fn_wan_video(latents=latents, timestep=ts, **base)
        if cfg_scale != 1.0:
            base["context"] = uncond_ctx
            np_nega = model_fn_wan_video(latents=latents, timestep=ts, **base)
            noise_pred = np_nega + cfg_scale * (np_posi - np_nega)
        else:
            noise_pred = np_posi
        latents = sched.step(noise_pred, sched.timesteps[i], latents)
    base["context"] = cond_ctx
    return latents


@torch.no_grad()
def decode_last_frame(pipe, latents):
    video = pipe.vae.decode(latents.to(DEVICE, DTYPE), device=DEVICE,
                            tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    frames = pipe.vae_output_to_video(video)  # list of PIL
    return frames[-1], frames


# ---------------- metrics ----------------
def pil_to_t(img):
    a = np.asarray(img).astype(np.float32) / 255.0  # HWC RGB
    return torch.from_numpy(a)


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim(a, b):
    # simple global gaussian-window SSIM on luminance
    import torch.nn.functional as F
    def lum(x):  # HWC -> 1,1,H,W
        r, g, bl = x[..., 0], x[..., 1], x[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * bl
        return y.unsqueeze(0).unsqueeze(0)
    x = lum(a); y = lum(b)
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    k = 11; sigma = 1.5
    coords = torch.arange(k).float() - k // 2
    gss = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); gss /= gss.sum()
    win = (gss[:, None] @ gss[None, :]).unsqueeze(0).unsqueeze(0)
    def filt(z): return F.conv2d(z, win, padding=k // 2)
    mu_x = filt(x); mu_y = filt(y)
    mu_x2 = mu_x ** 2; mu_y2 = mu_y ** 2; mu_xy = mu_x * mu_y
    sx = filt(x * x) - mu_x2; sy = filt(y * y) - mu_y2; sxy = filt(x * y) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx + sy + C2))
    return ssim_map.mean().item()


def main():
    blob = torch.load(IN, map_location="cpu", weights_only=False)
    uncond_context = torch.load(UNCOND, map_location="cpu", weights_only=False)
    print(f"[ctx] uncond {tuple(uncond_context.shape)} cached-cond "
          f"{tuple(blob['shared_tensors']['context'].shape)}", flush=True)

    pipe = load_pipe()
    views, base_memory = build_views(pipe, blob)
    any_idx = next(iter(views))
    shape = tuple(views[any_idx]["inputs"]["input_latents"].shape)
    print(f"[shapes] latents={shape} memory={tuple(base_memory.shape)}", flush=True)

    frozen_store = base_memory.detach().clone()
    learned_store = torch.load(LEARNED, map_location="cpu", weights_only=False)
    if not torch.is_tensor(learned_store):
        learned_store = learned_store.get("store", learned_store.get("memory"))
    learned_store = learned_store.to(DEVICE, DTYPE)
    print(f"[store] frozen norm={frozen_store.float().norm():.2f} "
          f"learned norm={learned_store.float().norm():.2f} "
          f"meanabsdiff={(frozen_store.float()-learned_store.float()).abs().mean():.5f}", flush=True)

    load_vae_into(pipe)

    # LPIPS
    import lpips
    lpips_alex = lpips.LPIPS(net="alex").to(DEVICE).eval()
    lpips_vgg = lpips.LPIPS(net="vgg").to(DEVICE).eval()

    def lpips_dist(net, gen_img, real_img):
        # expects [-1,1], NCHW
        def t(img):
            x = pil_to_t(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            return x * 2 - 1
        with torch.no_grad():
            return net(t(gen_img), t(real_img)).item()

    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # decode real reference (last frame) per view ONCE
    real_imgs = {}
    for vidx in INWINDOW + HELDOUT:
        ri, _ = decode_last_frame(pipe, views[vidx]["inputs"]["input_latents"])
        real_imgs[vidx] = ri
        ri.save(f"{OUT}/view{vidx}_real.png")
    print("[real] decoded real references", flush=True)

    results = {}
    lines = []

    # ---- SANITY CONTROL ----
    print(f"\n=== SANITY CONTROL: regenerate in-window view {SANITY_VIEW} with FROZEN store, CFG={CFG_SCALE} ===", flush=True)
    t0 = time.time()
    lat = generate_cfg(pipe, views[SANITY_VIEW]["inputs"], frozen_store, uncond_context,
                       shape, GEN_SEED, N_INFER_STEPS, CFG_SCALE)
    sane_img, _ = decode_last_frame(pipe, lat)
    sane_img.save(f"{OUT}/sanity_view{SANITY_VIEW}_frozen.png")
    s_lp_a = lpips_dist(lpips_alex, sane_img, real_imgs[SANITY_VIEW])
    s_lp_v = lpips_dist(lpips_vgg, sane_img, real_imgs[SANITY_VIEW])
    s_psnr = psnr(pil_to_t(sane_img), pil_to_t(real_imgs[SANITY_VIEW]))
    s_ssim = ssim(pil_to_t(sane_img), pil_to_t(real_imgs[SANITY_VIEW]))
    sanity = dict(lpips_alex=s_lp_a, lpips_vgg=s_lp_v, psnr=s_psnr, ssim=s_ssim,
                  secs=time.time() - t0)
    print(f"[SANITY] view {SANITY_VIEW} frozen+CFG vs real: "
          f"LPIPS(alex)={s_lp_a:.4f} LPIPS(vgg)={s_lp_v:.4f} PSNR={s_psnr:.2f} SSIM={s_ssim:.4f} "
          f"({sanity['secs']:.0f}s)", flush=True)
    lines.append(f"SANITY (view {SANITY_VIEW}, frozen+CFG{CFG_SCALE} vs real): "
                 f"LPIPS_alex={s_lp_a:.4f} LPIPS_vgg={s_lp_v:.4f} PSNR={s_psnr:.2f} SSIM={s_ssim:.4f}")

    # ---- main loop ----
    gen_cache = {}  # (vidx, which) -> PIL
    for vidx in INWINDOW + HELDOUT:
        real_img = real_imgs[vidx]
        row = {}
        for which, store in [("frozen", frozen_store), ("learned", learned_store)]:
            t0 = time.time()
            lat = generate_cfg(pipe, views[vidx]["inputs"], store, uncond_context,
                               shape, GEN_SEED, N_INFER_STEPS, CFG_SCALE)
            img, _ = decode_last_frame(pipe, lat)
            gen_cache[(vidx, which)] = img
            img.save(f"{OUT}/view{vidx}_{which}.png")
            la = lpips_dist(lpips_alex, img, real_img)
            lv = lpips_dist(lpips_vgg, img, real_img)
            ps = psnr(pil_to_t(img), pil_to_t(real_img))
            ss = ssim(pil_to_t(img), pil_to_t(real_img))
            row[which] = dict(lpips_alex=la, lpips_vgg=lv, psnr=ps, ssim=ss, secs=time.time() - t0)
            print(f"[view {vidx}/{which}] LPIPS_alex={la:.4f} LPIPS_vgg={lv:.4f} "
                  f"PSNR={ps:.2f} SSIM={ss:.4f} ({row[which]['secs']:.0f}s)", flush=True)
        results[vidx] = row
        f, l = row["frozen"], row["learned"]
        dl = f["lpips_alex"] - l["lpips_alex"]
        lines.append(f"view {vidx:2d} ({'IN ' if vidx<=KEY_WINDOW_MAX else 'OUT'}): "
                     f"frozen LPIPSa={f['lpips_alex']:.4f}/v={f['lpips_vgg']:.4f} PSNR={f['psnr']:.2f} SSIM={f['ssim']:.4f} | "
                     f"learned LPIPSa={l['lpips_alex']:.4f}/v={l['lpips_vgg']:.4f} PSNR={l['psnr']:.2f} SSIM={l['ssim']:.4f} | "
                     f"dLPIPSa={dl:+.4f}")

    # ---- means split ----
    def agg(idxs, which, key):
        return float(np.mean([results[i][which][key] for i in idxs]))
    mean_lines = []
    for label, idxs in [("IN-WINDOW", INWINDOW), ("HELD-OUT", HELDOUT)]:
        for which in ("frozen", "learned"):
            mean_lines.append(
                f"{label} mean {which}: LPIPSa={agg(idxs,which,'lpips_alex'):.4f} "
                f"LPIPSv={agg(idxs,which,'lpips_vgg'):.4f} "
                f"PSNR={agg(idxs,which,'psnr'):.2f} SSIM={agg(idxs,which,'ssim'):.4f}")
        dla = agg(idxs, "frozen", "lpips_alex") - agg(idxs, "learned", "lpips_alex")
        dlv = agg(idxs, "frozen", "lpips_vgg") - agg(idxs, "learned", "lpips_vgg")
        mean_lines.append(f"{label} learned-improvement: dLPIPSa={dla:+.4f} dLPIPSv={dlv:+.4f} "
                          f"(positive = learned better)")

    # ---- side-by-side PNGs ----
    saved = []
    for vidx in PNG_VIEWS:
        fig, axs = plt.subplots(1, 3, figsize=(21, 4.5))
        f, l = results[vidx]["frozen"], results[vidx]["learned"]
        imgs = [real_imgs[vidx], gen_cache[(vidx, "frozen")], gen_cache[(vidx, "learned")]]
        titles = [f"REAL (view {vidx}, last frame)",
                  f"FROZEN+CFG{int(CFG_SCALE)}  LPIPSa={f['lpips_alex']:.3f} PSNR={f['psnr']:.1f}",
                  f"LEARNED+CFG{int(CFG_SCALE)} LPIPSa={l['lpips_alex']:.3f} PSNR={l['psnr']:.1f}"]
        for ax, img, ti in zip(axs, imgs, titles):
            ax.imshow(img); ax.set_title(ti, fontsize=12); ax.axis("off")
        tag = "IN-WINDOW" if vidx <= KEY_WINDOW_MAX else "HELD-OUT (outside key window)"
        fig.suptitle(f"Captain-Safari view {vidx} [{tag}] — {N_INFER_STEPS}-step CFG{int(CFG_SCALE)} gen, seed {GEN_SEED}",
                     fontsize=13)
        fig.tight_layout()
        p = f"{OUT}/view{vidx}_sidebyside.png"
        fig.savefig(p, dpi=90, bbox_inches="tight"); plt.close(fig)
        saved.append(p)
        print(f"[saved] {p}", flush=True)

    # ---- write result file ----
    with open("/workspace/genquality_result.txt", "w") as fh:
        fh.write(f"Captain-Safari generation-QUALITY experiment (CFG ON)\n")
        fh.write(f"steps={N_INFER_STEPS} cfg={CFG_SCALE} seed={GEN_SEED} shift={SIGMA_SHIFT}\n")
        fh.write(f"real reference = VAE-decode of viewpoint input_latents (last frame)\n\n")
        for ln in [lines[0], ""] + lines[1:] + [""] + mean_lines:
            fh.write(ln + "\n")
        fh.write("\nSIDE_BY_SIDE_PNGS:\n")
        for s in saved:
            fh.write("  " + s + "\n")
        fh.write(f"\nmax_vram_GB={torch.cuda.max_memory_allocated()/1e9:.2f}\n")
    with open("/workspace/genquality_result.json", "w") as fh:
        json.dump(dict(sanity=sanity, results=results, png=saved), fh, indent=2)

    print("\n================ RESULT TABLE ================", flush=True)
    for ln in lines:
        print(ln, flush=True)
    print("---- means ----", flush=True)
    for ln in mean_lines:
        print(ln, flush=True)
    print(f"max_vram_GB={torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
    print("GENQUALITY_DONE", flush=True)


if __name__ == "__main__":
    main()
