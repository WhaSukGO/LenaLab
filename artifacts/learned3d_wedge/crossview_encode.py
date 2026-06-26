"""
Phase A (ENCODE) — Captain-Safari CROSS-VIEWPOINT held-out test.

Holds out CAMERA VIEWPOINTS (not noise/timestep). For each target viewpoint i in the
clip_0_20 trajectory we build a per-viewpoint sample:
  - extrinsic_query/intrinsic_query = clip_0_20 camera at index i   -> drives memory retrieval
  - input_latents = VAE-encode of a 21-mp4-frame window ENDING at that viewpoint's frame
                    (mp4 frame = 6*i; window = frames [6*i-20 .. 6*i] clamped to [0,120])
                    -> the real image the camera at viewpoint i sees (last frame = the query view)
Shared across all viewpoints (the released conditioning):
  - y (FunCameraControl plucker embedding from full clip_0_20 trajectory + input_image=frame0)
  - context (T5 prompt), memory (released 16-frame streamvggt store), key cameras.

Decoupled: encode here under no_grad with full DiT+T5+VAE, save every tensor to
/workspace/multi_vp_inputs.pt (.cpu()), then exit so VAE/T5 are freed. Phase B trains DiT-only.
"""
import os, sys
import numpy as np
import torch

REPO = "/workspace/Captain-Safari/captain_safari"
sys.path.insert(0, REPO)
os.chdir(REPO)

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import VideoDataset

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODELS = "/workspace/models/Wan2.2-Fun-5B"
LOCAL_DIT = f"{MODELS}/diffusion_pytorch_model.safetensors"
LOCAL_T5  = f"{MODELS}/models_t5_umt5-xxl-enc-bf16.pth"
LOCAL_VAE = f"{MODELS}/Wan2.2_VAE.pth"
CKPT = "/workspace/cs_ckpt/models/Wan2.2-Fun-5B-Control-Camera_Captain-Safari.PreEnc/epoch-4.safetensors"
ASSETS = "/workspace/cs_assets"
FRAMES_DIR = f"{ASSETS}/data-frames_4fps-camera-fixed-local-interpolated-slices"
UID = "0bb4eeb9-5e17-4743-acd4-fd97379d2353"
MP4 = f"{ASSETS}/data-slices/{UID}_0.mp4"
OUT = "/workspace/multi_vp_inputs.pt"

HEIGHT, WIDTH, NUM_FRAMES = 704, 1280, 21
KEY_WINDOW_MAX = 15          # key_0_16 covers clip indices 0..15; >15 = OUTSIDE the key window
MP4_STRIDE = 6               # 121 mp4 frames = 20 clip frames * 6 + 1 (4fps)

# Target viewpoints to encode: span the trajectory; include 16..19 (OUTSIDE key window).
TARGET_VIEWS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 19]

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


def build_shared(pipe):
    """Run the full unit chain ONCE with the released demo inputs to get shared y/context/memory/keys."""
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
    print("[units] running full unit chain for shared y/context/memory...", flush=True)
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    inputs = {**inputs_shared, **inputs_posi}
    print(f"[shapes] input_latents={tuple(inputs['input_latents'].shape)} "
          f"y={tuple(inputs['y'].shape)} context={tuple(inputs['context'].shape)} "
          f"memory={tuple(inputs['memory'].shape)} "
          f"extr_key={tuple(inputs['extrinsic_key'].shape)} intr_key={tuple(inputs['intrinsic_key'].shape)}",
          flush=True)
    return inputs, prompt, num_frames


def load_mp4_window(end_mp4_idx):
    """Load 21 mp4 frames ending at end_mp4_idx (clamped), crop/resize as VideoDataset does."""
    import imageio
    from PIL import Image
    start = max(0, end_mp4_idx - (NUM_FRAMES - 1))
    end = start + NUM_FRAMES - 1
    if end > 120:
        end = 120; start = end - (NUM_FRAMES - 1)
    idxs = list(range(start, end + 1))
    # reuse VideoDataset crop_and_resize for identical preprocessing
    ds = VideoDataset(base_path=ASSETS, metadata_path=os.path.join(ASSETS, "metadata.csv"),
                      height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES, data_file_keys=("video",))
    reader = imageio.get_reader(MP4)
    frames = []
    for fi in idxs:
        fr = Image.fromarray(reader.get_data(fi))
        fr = ds.crop_and_resize(fr, *ds.get_height_width(fr))
        frames.append(fr)
    reader.close()
    return frames, (start, end)


def encode_window_latents(pipe, frames):
    """VAE-encode a 21-frame window -> input_latents (same path as WanVideoUnit_InputVideoEmbedder)."""
    vid = pipe.preprocess_video(frames)  # [1,3,21,H,W]
    lat = pipe.vae.encode(vid, device=pipe.device, tiled=False).to(dtype=DTYPE, device=pipe.device)
    return lat


def main():
    pipe = load_pipe()
    with torch.no_grad():
        shared, prompt, num_frames = build_shared(pipe)

        # shared tensors (released conditioning)
        shared_tensors = {
            "y": shared["y"].detach().cpu(),
            "context": shared["context"].detach().cpu(),
            "memory": shared["memory"].detach().cpu(),
            "intrinsic_key": shared["intrinsic_key"].detach().cpu(),
            "extrinsic_key": shared["extrinsic_key"].detach().cpu(),
        }
        for k, v in shared_tensors.items():
            print(f"[shared] {k} {tuple(v.shape)} {v.dtype}", flush=True)

        # per-viewpoint cameras from clip_0_20
        clip_ext = np.load(f"{FRAMES_DIR}/{UID}_clip_0_20_extrinsic.npy")   # (20,3,4)
        clip_int = np.load(f"{FRAMES_DIR}/{UID}_clip_0_20_intrinsic.npy")   # (20,3,3)
        n_clip = clip_ext.shape[0]
        print(f"[clip] {n_clip} trajectory cameras", flush=True)

        views = []
        for i in TARGET_VIEWS:
            assert 0 <= i < n_clip, i
            end_mp4 = min(120, MP4_STRIDE * i)
            frames, (s, e) = load_mp4_window(end_mp4)
            lat = encode_window_latents(pipe, frames)             # [1,48,fl,h,w]
            extr_q = torch.tensor(clip_ext[i], dtype=DTYPE).unsqueeze(0)   # [1,3,4]
            intr_q = torch.tensor(clip_int[i], dtype=DTYPE).unsqueeze(0)   # [1,3,3]
            outside = i > KEY_WINDOW_MAX
            views.append({
                "view_idx": int(i),
                "mp4_window": (int(s), int(e)),
                "outside_key_window": bool(outside),
                "input_latents": lat.detach().cpu(),
                "extrinsic_query": extr_q.detach().cpu(),
                "intrinsic_query": intr_q.detach().cpu(),
            })
            print(f"[view {i:2d}] outside_key={outside} mp4[{s}..{e}] "
                  f"latents={tuple(lat.shape)} cam_center={np.round(clip_ext[i,:,3],3)}", flush=True)

        config = {"height": HEIGHT, "width": WIDTH, "num_frames": num_frames}
        blob = {
            "shared_tensors": shared_tensors,
            "views": views,
            "config": config,
            "prompt": prompt,
            "TARGET_VIEWS": TARGET_VIEWS,
            "KEY_WINDOW_MAX": KEY_WINDOW_MAX,
        }
        torch.save(blob, OUT)
        print(f"[done] wrote {OUT} with {len(views)} viewpoints", flush=True)


if __name__ == "__main__":
    main()
