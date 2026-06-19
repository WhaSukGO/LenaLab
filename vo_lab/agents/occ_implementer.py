"""Track B — the Occupancy Implementer (the 6th domain): the solver AUTHORS a camera->3D-occupancy
network, graded on held-out nuScenes scenes by voxel IoU. Same harness contract as the BEV domain,
extended to a 3D voxel grid. The harness owns the grader (eval_occ.py) and the held-out 3D GT."""
from __future__ import annotations

from pathlib import Path

from lab.agents.implementer import ImplementationTask
from lab.models import Usage

from ..plugins.vo import VO_CODE_DIR, _GPU_FW

_OCC_EVAL_CODE = (Path(VO_CODE_DIR) / "eval_occ.py").read_text()
_OCC_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_occ_learned.py").read_text()

OCC_TASK_DESCRIPTION = (
    "Implement a multi-camera 3D OCCUPANCY network with PyTorch that TRAINS ON THE GPU: predict, for "
    "a voxel grid around the ego vehicle, which voxels are occupied by a vehicle.\n"
    "INPUT: from $LAB_DATA/train/<token>.npz read (the supervised set):\n"
    "  imgs    (6,128,352,3) uint8  -- 6 surround cameras [FL,F,FR,BL,B,BR]\n"
    "  intrins (6,3,3) float32      -- pinhole K per camera (scaled to 128x352)\n"
    "  cam2ego (6,4,4) float32      -- camera->ego extrinsic (metres)\n"
    "  occ     (200,200,12) uint8   -- TARGET: vehicle occupancy in the ego VOXEL grid\n"
    "VOXEL GRID (fixed): ego frame, x forward / y left / z up; x in [-50,50], y in [-50,50], "
    "z in [-2,4] metres at 0.5 m/voxel -> 200 x 200 x 12. Voxel (ix,iy,iz) centre = "
    "(-50+0.5*(ix+.5), -50+0.5*(iy+.5), -2+0.5*(iz+.5)). The ego is at the grid centre in x,y.\n"
    "TASK: lift the 6 images into this 3D grid using the intrinsics+extrinsics (e.g. Lift-Splat to "
    "3D: per-pixel depth distribution x context, project to ego 3D points via K^-1 and cam2ego, "
    "voxel-pool into X*Y*Z, then a 3D-conv head) and predict per-voxel occupancy. Vehicle voxels are "
    "SPARSE (~0.3%% of the grid) -- weight the positive class. No pretrained weights (no network); "
    "train from scratch (torchvision architectures with weights=None are fine).\n"
    "OUTPUT: for each $LAB_DATA/test_input/<token>.npz (cams+calib, no occ) write "
    "$LAB_ARTIFACTS/pred_<token>.npy -- a (200,200,12) uint8 {0,1} voxel mask (you choose the "
    "threshold). Graded by held-out per-sample voxel IoU on nuScenes scenes you never see. torch "
    "(CUDA), torchvision, numpy, opencv available. Do not read any test-set occ GT.")


def occ_impl_task(threshold: float = 0.05, *, train_max=None, test_max=None) -> ImplementationTask:
    """Occupancy Track-B task (framework=torch -> vo-gpu-torch:1). Graded by eval_occ.py (held-out
    voxel IoU). 3D voxels are sparse, so the bar is lower in absolute terms than the 2D BEV bar."""
    from ..plugins.occ_nuscenes import occ_datasets
    return ImplementationTask(
        description=OCC_TASK_DESCRIPTION, framework=_GPU_FW,
        entry_command='timeout 3600 python3 "$LAB_CODE/main.py"', eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_OCC_EVAL_CODE, metric="miou", op=">=", threshold=threshold,
        datasets=occ_datasets(), entry_filename="main.py")


def occ_reference_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_OCC_REFERENCE_MAIN)
        return Usage()
    return author


def occ_degenerate_author():
    src = ("import os, glob, numpy as np\n"
           "d=os.environ['LAB_DATA']; a=os.environ['LAB_ARTIFACTS']; os.makedirs(a,exist_ok=True)\n"
           "for f in glob.glob(os.path.join(d,'test_input','*.npz')):\n"
           "    tok=os.path.splitext(os.path.basename(f))[0]\n"
           "    np.save(os.path.join(a,f'pred_{tok}.npy'), np.zeros((200,200,12),np.uint8))\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
        return Usage()
    return author


# --- SCAFFOLD variant: lock the 3D geometry + augmentation, agent authors ONLY the network ------
_OCC_SCAFFOLD_CORE = (Path(VO_CODE_DIR) / "occ_scaffold.py").read_text()
_OCC_SCAFFOLD_MODEL_REF = (Path(VO_CODE_DIR) / "occ_scaffold_model_ref.py").read_text()

OCC_SCAFFOLD_DESCRIPTION = (
    "Author the NEURAL NETWORK for a multi-camera 3D-occupancy model. The hard parts are DONE and "
    "LOCKED in `occ_core.py` (in your working dir -- DO NOT edit it): the Lift-Splat-to-3D geometry "
    "(frustum -> ego 3D points -> voxel-pool into the 200x200x12 grid), the correct surround flip "
    "augmentation, the training loop, threshold calibration, and inference/output. You write ONLY "
    "`model.py`, exposing exactly two factory functions:\n"
    "  build_encoder() -> nn.Module: forward(imgs[B*6,3,128,352]) -> (depth_logits[B*6,D,h,w], "
    "context[B*6,C,h,w]). D is fixed -- `from occ_core import DEPTH_BINS`. You choose C and the "
    "downsample h,w. No pretrained weights (no network); build from scratch (weights=None).\n"
    "  build_occ_head() -> nn.Module: forward(vox[B,C,200,200,12]) -> occupancy_logits[B,200,200,12]. "
    "Must consume the same C your encoder emits.\n"
    "occ_core.py wires them: enc -> lift_splat_3d(depth_logits, context, K, cam2ego) -> head. Graded "
    "by held-out voxel IoU on scenes you never see. Design the network for the best held-out IoU. Do "
    "NOT touch geometry/augmentation/training/grader -- only model.py. torch (CUDA), torchvision, numpy.")


def occ_scaffold_seed() -> dict:
    return {"occ_core.py": _OCC_SCAFFOLD_CORE}


def occ_impl_task_scaffold(threshold: float = 0.06, *, train_max=None, test_max=None) -> ImplementationTask:
    from ..plugins.occ_nuscenes import occ_datasets
    return ImplementationTask(
        description=OCC_SCAFFOLD_DESCRIPTION, framework=_GPU_FW,
        entry_command='timeout 3600 python3 "$LAB_CODE/occ_core.py"', eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_OCC_EVAL_CODE, metric="miou", op=">=", threshold=threshold,
        datasets=occ_datasets(), entry_filename="model.py")


def seeded(inner_author, seed: dict):
    def author(task, code_dir: Path, rec) -> Usage:
        for name, src in seed.items():
            (Path(code_dir) / name).write_text(src)
        return inner_author(task, code_dir, rec)
    return author


def occ_scaffold_reference_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "occ_core.py").write_text(_OCC_SCAFFOLD_CORE)
        (Path(code_dir) / "model.py").write_text(_OCC_SCAFFOLD_MODEL_REF)
        return Usage()
    return author


def occ_scaffold_degenerate_author():
    src = ("import torch.nn as nn\nfrom occ_core import DEPTH_BINS\n"
           "class E(nn.Module):\n"
           "    def __init__(s):\n        super().__init__(); s.c=nn.Conv2d(3,DEPTH_BINS+8,1)\n"
           "    def forward(s,x):\n        import torch.nn.functional as F\n        y=F.avg_pool2d(s.c(x),16); return y[:,:DEPTH_BINS], y[:,DEPTH_BINS:]\n"
           "class Hd(nn.Module):\n"
           "    def __init__(s):\n        super().__init__(); s.c=nn.Conv3d(8,1,1)\n"
           "    def forward(s,v):\n        return s.c(v).squeeze(1)\n"
           "def build_encoder(): return E()\ndef build_occ_head(): return Hd()\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "occ_core.py").write_text(_OCC_SCAFFOLD_CORE)
        (Path(code_dir) / "model.py").write_text(src)
        return Usage()
    return author
