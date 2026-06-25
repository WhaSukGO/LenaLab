"""
Phase A (ENCODE) — Captain-Safari REAL held-out test, decoupled.

Builds the REAL conditioning (VAE-encode demo mp4 -> input_latents, FunCameraControl ->
y, T5 -> context, released memory tensor, real camera key/query poses) using the FULL
pipeline (DiT+T5+VAE), under torch.no_grad(), then torch.save()s every tensor needed by
the denoising loss to /workspace/real_inputs.pt (all moved .cpu()). Also saves the held-out
noise tensors + timestep ids/values for the SAME (noise,timestep) grid the frozen baseline
used. Then this process exits, freeing VAE/T5.

This is the encode half of the decouple fix for the known backward bug ("tensors on cuda:0
and cpu" in the complex-RoPE MulBackward) — Phase B trains in a FRESH DiT-only process so the
autograd graph has only clean GPU leaves.
"""
import os, sys, json
import numpy as np
import torch

REPO = "/workspace/Captain-Safari/captain_safari"
sys.path.insert(0, REPO)
os.chdir(REPO)

from diffsynth.pipelines.wan_video_new import (
    WanVideoPipeline, ModelConfig, model_fn_wan_video,
)
from diffsynth.trainers.utils import VideoDataset

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODELS = "/workspace/models/Wan2.2-Fun-5B"
LOCAL_DIT = f"{MODELS}/diffusion_pytorch_model.safetensors"
LOCAL_T5  = f"{MODELS}/models_t5_umt5-xxl-enc-bf16.pth"
LOCAL_VAE = f"{MODELS}/Wan2.2_VAE.pth"
CKPT = "/workspace/cs_ckpt/models/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors"
ASSETS = "/workspace/cs_assets"
OUT = "/workspace/real_inputs.pt"

HEIGHT, WIDTH, NUM_FRAMES = 704, 1280, 21

# SAME grid as the frozen baseline (=0.564) used.
TRAIN_TIDS = [620, 700, 780, 860, 920]
EVAL_TIDS  = [660, 740, 820, 900, 950]
TRAIN_SEEDS = [10, 11, 12, 13, 14]
EVAL_SEEDS  = [90, 91, 92, 93, 94]

torch.manual_seed(0)


def load_pipe():
    print("[load] FULL pipeline (DiT+T5+VAE on GPU)...", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=DTYPE, device=DEVICE,
        model_configs=[
            ModelConfig(path=LOCAL_DIT, offload_device=None),
            ModelConfig(path=LOCAL_T5,  offload_device=None),
            ModelConfig(path=LOCAL_VAE, offload_device=None),
        ],
    )
    if pipe.vae is not None:
        pipe.vae.to(device=DEVICE, dtype=DTYPE)
    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device=DEVICE, dtype=DTYPE)
    pipe.dit.use_memory_retrieval = True
    pipe.dit.use_memory_cross_attn = True
    pipe.dit.require_vae_embedding = True
    pipe.load_lora(pipe.dit, CKPT, alpha=1)
    pipe.load_memory(pipe.dit, CKPT)
    pipe.load_cross_attn(pipe.dit, CKPT)
    pipe.load_memory_retriever_from_dit(pipe.dit, CKPT)
    pipe.dit.to(device=DEVICE, dtype=DTYPE)
    pipe.scheduler.set_timesteps(1000, training=True)
    print("[load] done.", flush=True)
    return pipe


def build_real_inputs(pipe):
    ds = VideoDataset(
        base_path=ASSETS,
        metadata_path=os.path.join(ASSETS, "metadata.csv"),
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        data_file_keys=("video", "memory", "intrinsic_query", "extrinsic_query",
                        "intrinsic_key", "extrinsic_key", "extrinsic_clip", "intrinsic_clip"),
    )
    data = ds[0]
    prompt = data["prompt"]
    video_frames = data["video"]
    num_frames = len(video_frames)
    print(f"[data] prompt={prompt!r} num_frames={num_frames}", flush=True)

    inputs_shared = {
        "input_video": video_frames,
        "height": HEIGHT, "width": WIDTH, "num_frames": num_frames,
        "cfg_scale": 1, "tiled": False, "rand_device": pipe.device,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "cfg_merge": False, "vace_scale": 1,
        "max_timestep_boundary": 1, "min_timestep_boundary": 0,
        "input_image": video_frames[0],
        "memory": data["memory"],
        "intrinsic_query": data["intrinsic_query"],
        "extrinsic_query": data["extrinsic_query"],
        "intrinsic_key": data["intrinsic_key"],
        "extrinsic_key": data["extrinsic_key"],
        "extrinsic_clip": data["extrinsic_clip"],
        "intrinsic_clip": data["intrinsic_clip"],
    }
    inputs_posi = {"prompt": prompt}
    inputs_nega = {}

    print("[units] running real units (VAE encode, T5 encode, FunCameraControl y)...", flush=True)
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    inputs = {**inputs_shared, **inputs_posi}
    print(f"[shapes] input_latents={tuple(inputs['input_latents'].shape)} "
          f"y={tuple(inputs['y'].shape)} context={tuple(inputs['context'].shape)} "
          f"memory={tuple(inputs['memory'].shape)}", flush=True)
    return inputs, prompt, num_frames


def main():
    pipe = load_pipe()
    with torch.no_grad():
        inputs, prompt, num_frames = build_real_inputs(pipe)

        sched = pipe.scheduler

        # tensors consumed by the denoising loss / model_fn_wan_video
        tensor_keys = ["input_latents", "y", "context", "memory",
                       "intrinsic_query", "extrinsic_query", "intrinsic_key", "extrinsic_key"]
        saved = {}
        for k in tensor_keys:
            v = inputs[k]
            assert torch.is_tensor(v), f"{k} is not a tensor: {type(v)}"
            saved[k] = v.detach().cpu()
            print(f"[save] {k} {tuple(saved[k].shape)} {saved[k].dtype}", flush=True)

        # scalar config the model_fn needs
        config = {
            "height": int(inputs.get("height", HEIGHT)),
            "width": int(inputs.get("width", WIDTH)),
            "num_frames": int(num_frames),
        }

        # Build the held-out (noise, timestep) grid — fixed CPU-generated noise + timestep
        # VALUES (not just ids) so Phase B reproduces them exactly without the scheduler.
        shape = tuple(inputs["input_latents"].shape)

        def make(seeds, tids):
            out = []
            for s, tid in zip(seeds, tids):
                g = torch.Generator(device="cpu").manual_seed(s)
                noise = torch.randn(shape, generator=g)  # cpu fp32
                ts = sched.timesteps[tid:tid + 1].detach().cpu()  # the actual timestep value
                tw = float(sched.training_weight(ts.to(DEVICE)))
                out.append({"seed": int(s), "tid": int(tid),
                            "noise": noise, "timestep": ts,
                            "t_value": float(ts.item()), "train_weight": tw})
            return out

        train_samples = make(TRAIN_SEEDS, TRAIN_TIDS)
        eval_samples  = make(EVAL_SEEDS,  EVAL_TIDS)

        blob = {
            "tensors": saved,
            "config": config,
            "prompt": prompt,
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "TRAIN_TIDS": TRAIN_TIDS, "EVAL_TIDS": EVAL_TIDS,
            "TRAIN_SEEDS": TRAIN_SEEDS, "EVAL_SEEDS": EVAL_SEEDS,
        }
        torch.save(blob, OUT)
        print(f"[done] wrote {OUT}", flush=True)
        for s in eval_samples:
            print(f"  eval sample seed={s['seed']} tid={s['tid']} t={s['t_value']:.1f} w={s['train_weight']:.4f}", flush=True)


if __name__ == "__main__":
    main()
