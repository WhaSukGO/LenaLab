#!/usr/bin/env python
"""
heldout_real.py — Captain-Safari GO/NO-GO: does a LEARNED in-graph memory store beat the
FROZEN released store on REAL HELD-OUT data?

This is the DECOUPLED fix for the known backward bug: on the real-input path (full pipeline
with CPU-offloaded VAE/T5) loss.backward() died with "tensors on cuda:0 and cpu" inside a
complex-RoPE MulBackward, because the offloaded encode path left a CPU tensor in the autograd
graph. Fix = split encoding from training into two PROCESSES:

  Phase A (encode):  python heldout_real.py encode
      -> loads FULL pipeline (DiT+T5+VAE), builds REAL input_latents (VAE-encoded demo mp4),
         y (FunCameraControl), context (T5), the released memory store, and real camera
         key/query poses, all under torch.no_grad(); torch.save()s them (.cpu()) to
         /workspace/real_inputs.pt together with the held-out (noise,timestep) grid. Exits
         (frees VAE/T5).  [implementation: heldout_real_encode.py]

  Phase B+C (train): python heldout_real.py train
      -> FRESH DiT-only process (no VAE/T5, no offload). Loads real_inputs.pt as clean GPU
         bf16 leaves. The only trainable is the fp32 STORE (Parameter init from the real
         released memory). Confirms one backward SUCCEEDS (bug bypassed), trains the store on
         the TRAIN (noise,tid) split, and reports FROZEN vs LEARNED on the DISJOINT HELD-OUT
         split. Writes /workspace/heldout_real_result.txt.  [implementation: heldout_real_train.py]

Held-out protocol (1 query frame in demo -> hold out on the (noise,timestep) axis):
  TRAIN tids[620,700,780,860,920]/seeds[10-14];  EVAL tids[660,740,820,900,950]/seeds[90-94]
  (disjoint on BOTH axes; same grid as the FROZEN baseline so it is comparable to 0.564).
"""
import sys, runpy, os
os.chdir("/workspace")
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "train"
    if phase == "encode":
        runpy.run_path("/workspace/heldout_real_encode.py", run_name="__main__")
    elif phase == "train":
        runpy.run_path("/workspace/heldout_real_train.py", run_name="__main__")
    else:
        print("usage: heldout_real.py [encode|train]"); sys.exit(2)
