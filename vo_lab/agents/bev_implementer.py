"""Track B — the BEV Implementer: the solver AUTHORS a multi-camera Bird's-Eye-View perception
network, then the unchanged independent evaluator grades it on held-out nuScenes scenes by IoU.

This is the lab's SECOND problem class (multi-view perception, not ego-motion) — proof the
verification-first harness generalizes beyond VO/SLAM. The agent writes only main.py (a torch
training + inference pipeline that runs on the GPU as a harness job); the harness owns the grader
(eval_bev.py) and the oracle (IoU >= bar). The held-out BEV ground truth is never in the
trainer's input, and the grader is restored from the task spec before judging.
"""
from __future__ import annotations

from pathlib import Path

from lab.agents.implementer import ImplementationTask
from lab.models import Usage

from ..plugins.vo import VO_CODE_DIR, _GPU_FW

_BEV_EVAL_CODE = (Path(VO_CODE_DIR) / "eval_bev.py").read_text()        # harness-owned grader
_BEV_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_bev_learned.py").read_text()  # known-good baseline

BEV_TASK_DESCRIPTION = (
    "Implement a multi-camera BIRD'S-EYE-VIEW vehicle-occupancy network with PyTorch that TRAINS "
    "ON THE GPU. This is surround-camera perception, NOT visual odometry.\n"
    "INPUT: from $LAB_DATA/train/<token>.npz read (the supervised training set):\n"
    "  imgs    (6,128,352,3) uint8  -- 6 surround cameras, order "
    "[FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT]\n"
    "  intrins (6,3,3) float32      -- pinhole K per camera, already scaled to the 128x352 images\n"
    "  cam2ego (6,4,4) float32      -- camera->ego extrinsic (rotation+translation), metres\n"
    "  bev     (200,200) uint8      -- TARGET: vehicle occupancy in the ego BEV grid\n"
    "BEV GRID (fixed): ego frame, x forward / y left, range x in [-50,50] m and y in [-50,50] m at "
    "0.5 m/cell -> 200x200. Cell (row=ix, col=iy) covers x = -50 + 0.5*ix, y = -50 + 0.5*iy. The "
    "ego/camera rig is at the grid centre.\n"
    "TASK: train a network that lifts the 6 camera images into this BEV grid using the given "
    "intrinsics+extrinsics (e.g. Lift-Splat: per-pixel depth distribution x context, project to "
    "3-D ego points via K^-1 and cam2ego, voxel-pool into the grid, conv head) and predicts "
    "vehicle occupancy. There is NO pretrained-weight download (no network) -- train from scratch.\n"
    "OUTPUT: for each $LAB_DATA/test_input/<token>.npz (same fields EXCEPT no bev), run inference "
    "and write $LAB_ARTIFACTS/pred_<token>.npy -- a (200,200) uint8 array of {0,1} predicted "
    "vehicle occupancy (you choose the threshold). The <token> is the npz filename stem.\n"
    "You are graded by held-out per-sample IoU (intersection-over-union of your mask vs the secret "
    "GT) on nuScenes scenes whose BEV labels you never see. torch (CUDA), torchvision (architectures "
    "only, weights=None), numpy, opencv are available. Do not attempt to read any test-set bev GT — "
    "there is none in test_input.")


def bev_impl_task(threshold: float = 0.12, *, train_max=None, test_max=None) -> ImplementationTask:
    """BEV Track-B task: the agent authors a torch BEV pipeline (framework=torch -> vo-gpu-torch:1,
    --gpus all). Graded by eval_bev.py (per-sample IoU on held-out nuScenes mini_val). Training is a
    harness JOB (wall-clock, not tokens). The bar is the from-scratch reference's held-out IoU."""
    from ..plugins.bev_nuscenes import bev_datasets

    return ImplementationTask(
        description=BEV_TASK_DESCRIPTION,
        framework=_GPU_FW,
        entry_command='python3 "$LAB_CODE/main.py"',
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_BEV_EVAL_CODE,
        metric="miou", op=">=", threshold=threshold,
        datasets=bev_datasets(),
        entry_filename="main.py",
    )


def bev_reference_author():
    """Writes the from-scratch Lift-Splat reference as main.py (trains on GPU; the baseline/bar)."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_BEV_REFERENCE_MAIN)
        return Usage()
    return author


def bev_degenerate_author():
    """Empty (all-zero) BEV per test sample — the negative control (IoU ~ 0)."""
    src = ("import os, glob, numpy as np\n"
           "d=os.environ['LAB_DATA']; a=os.environ['LAB_ARTIFACTS']; os.makedirs(a,exist_ok=True)\n"
           "for f in glob.glob(os.path.join(d,'test_input','*.npz')):\n"
           "    tok=os.path.splitext(os.path.basename(f))[0]\n"
           "    np.save(os.path.join(a,f'pred_{tok}.npy'), np.zeros((200,200),np.uint8))\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
        return Usage()
    return author


# --- SCAFFOLD variant: lock the fragile parts, agent authors ONLY the network ---------------
# Tests the n=3 finding's prescription: the variance was the agent's design latitude over the
# geometry/augmentation, not network capacity. Lock geometry + correct flip aug + training in a
# seeded bev_core.py; the agent authors only model.py (build_encoder + build_bev_head).
_BEV_SCAFFOLD_CORE = (Path(VO_CODE_DIR) / "bev_scaffold.py").read_text()         # the LOCKED core
_BEV_SCAFFOLD_MODEL_REF = (Path(VO_CODE_DIR) / "bev_scaffold_model_ref.py").read_text()

BEV_SCAFFOLD_DESCRIPTION = (
    "Author the NEURAL NETWORK for a multi-camera BEV vehicle-occupancy model. The hard parts are "
    "ALREADY DONE and LOCKED in `bev_core.py` (in your working dir — DO NOT edit or recreate it): the "
    "Lift-Splat geometry (frustum -> ego -> voxel-pool using the real intrinsics+extrinsics), the "
    "correct surround-camera flip augmentation, the training loop, threshold calibration, and "
    "inference/output. You write ONLY `model.py`, which must expose exactly two factory functions:\n"
    "  build_encoder() -> nn.Module: forward(imgs[B*6,3,128,352]) -> (depth_logits[B*6,D,h,w], "
    "context[B*6,C,h,w]). D is fixed — import it: `from bev_core import DEPTH_BINS`. You choose C "
    "(context channels) and the downsample h,w (e.g. ResNet to /16 gives 8x22). No pretrained weights "
    "(no network) — build from scratch (torchvision architectures with weights=None are fine).\n"
    "  build_bev_head() -> nn.Module: forward(bev[B,C,X,Y]) -> occupancy_logits[B,200,200]. Must "
    "consume the same C your encoder emits.\n"
    "bev_core.py wires them: enc -> lift_splat(depth_logits, context, K, cam2ego) -> head. You are "
    "graded by held-out IoU on nuScenes scenes you never see. Design the network (backbone, depth/"
    "context heads, BEV decoder, capacity) for the best held-out IoU. Do NOT touch geometry, "
    "augmentation, training, or the grader — only model.py. torch (CUDA), torchvision, numpy available.")


def bev_scaffold_seed() -> dict:
    """Files seeded (LOCKED) into the agent's workspace before authoring."""
    return {"bev_core.py": _BEV_SCAFFOLD_CORE}


def bev_impl_task_scaffold(threshold: float = 0.10, *, train_max=None, test_max=None) -> ImplementationTask:
    """SCAFFOLD Track-B BEV: geometry+augmentation+training LOCKED (seeded bev_core.py); the agent
    authors only model.py (the network). entry runs the locked core. Same held-out IoU grader."""
    from ..plugins.bev_nuscenes import bev_datasets

    return ImplementationTask(
        description=BEV_SCAFFOLD_DESCRIPTION,
        framework=_GPU_FW,
        entry_command='python3 "$LAB_CODE/bev_core.py"',     # runs the LOCKED core (imports model.py)
        eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_BEV_EVAL_CODE,
        metric="miou", op=">=", threshold=threshold,
        datasets=bev_datasets(),
        entry_filename="model.py",                            # the agent's ONLY deliverable
    )


def seeded(inner_author, seed: dict):
    """Wrap an author so the LOCKED seed files are written into code_dir before authoring."""
    def author(task, code_dir: Path, rec) -> Usage:
        for name, src in seed.items():
            (Path(code_dir) / name).write_text(src)
        return inner_author(task, code_dir, rec)
    return author


def bev_scaffold_reference_author():
    """Writes the reference model.py into the scaffold (the known-good network; for calibration)."""
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "bev_core.py").write_text(_BEV_SCAFFOLD_CORE)
        (Path(code_dir) / "model.py").write_text(_BEV_SCAFFOLD_MODEL_REF)
        return Usage()
    return author


def bev_scaffold_degenerate_author():
    """A trivial constant network (predicts from a learned bias only) — the scaffold negative control."""
    src = ("import torch.nn as nn\n"
           "from bev_core import DEPTH_BINS\n"
           "class E(nn.Module):\n"
           "    def __init__(s):\n"
           "        super().__init__(); s.c=nn.Conv2d(3,DEPTH_BINS+8,1)\n"
           "    def forward(s,x):\n"
           "        import torch.nn.functional as F\n"
           "        y=F.avg_pool2d(s.c(x),16); return y[:,:DEPTH_BINS], y[:,DEPTH_BINS:]\n"
           "class H(nn.Module):\n"
           "    def __init__(s):\n"
           "        super().__init__(); s.c=nn.Conv2d(8,1,1)\n"
           "    def forward(s,b):\n"
           "        return s.c(b).squeeze(1)\n"
           "def build_encoder(): return E()\n"
           "def build_bev_head(): return H()\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "bev_core.py").write_text(_BEV_SCAFFOLD_CORE)
        (Path(code_dir) / "model.py").write_text(src)
        return Usage()
    return author
