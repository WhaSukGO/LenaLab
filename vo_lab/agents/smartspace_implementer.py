"""Track B — the Smart-Space Implementer (the 7th domain): the solver AUTHORS a static-multi-camera
floor-occupancy network, graded on held-out (unseen-time) frames of a warehouse by floor IoU. Same
harness contract as the occupancy domain, but STATIC cameras (world-anchored grid) + 2D floor grid.
The harness owns the grader (eval_smartspace.py) and the held-out GT. Per-space self-verification:
train = the scene's first 70% of time, held-out = the last 30%."""
from __future__ import annotations

from pathlib import Path

from lab.agents.implementer import ImplementationTask
from lab.models import Usage

from ..plugins.vo import VO_CODE_DIR, _GPU_FW

_SS_EVAL_CODE = (Path(VO_CODE_DIR) / "eval_smartspace.py").read_text()
_SS_REFERENCE_MAIN = (Path(VO_CODE_DIR) / "run_smartspace_learned.py").read_text()

SMARTSPACE_TASK_DESCRIPTION = (
    "Implement a STATIC multi-camera FLOOR-OCCUPANCY network with PyTorch that TRAINS ON THE GPU: from "
    "many fixed overhead cameras watching a warehouse, predict which cells of a top-down floor grid are "
    "occupied by an agent (person/forklift/robot).\n"
    "INPUT: from $LAB_DATA/train/<token>.npz read (the supervised set):\n"
    "  imgs       (N,128,352,3) uint8  -- N fixed cameras (same N every sample; this scene has 19)\n"
    "  intrins    (N,3,3) float32      -- pinhole K per camera (scaled to 128x352)\n"
    "  cam_proj   (N,3,4) float32      -- world->image projection per camera (already includes K and the\n"
    "                                     extrinsic): pixel ~ cam_proj @ [X,Y,Z,1]. Cameras are STATIC.\n"
    "  grid_bounds (4,) float32        -- [x0,x1,y0,y1] world extent the floor grid covers (metres)\n"
    "  bev        (XG,YG) uint8        -- TARGET: floor occupancy on the world grid (this scene 206x203)\n"
    "FLOOR GRID (fixed per scene): world-anchored, 0.5 m/cell; cell (ix,iy) centre = "
    "(x0+0.5*(ix+.5), y0+0.5*(iy+.5)) with x0,y0 from grid_bounds. Read XG,YG from bev.shape.\n"
    "TASK: because cameras are static and you predict a ground plane, Inverse Perspective Mapping fits "
    "well -- for each floor cell at a few height planes z, project (X,Y,z) to each camera via cam_proj, "
    "bilinearly sample that camera's CNN features (grid_sample), fuse across the cameras that see the "
    "cell, and a 2D-conv head predicts per-cell occupancy. (A lift-splat-style approach is allowed too.) "
    "Occupied cells are SPARSE (~0.5%% of the grid) -- weight the positive class. No pretrained weights "
    "(no network); train from scratch (torchvision architectures with weights=None are fine).\n"
    "OUTPUT: for each $LAB_DATA/test_input/<token>.npz (cams+cam_proj+grid_bounds, no bev) write "
    "$LAB_ARTIFACTS/pred_<token>.npy -- an (XG,YG) uint8 {0,1} floor mask (you choose the threshold). "
    "Graded by held-out per-sample floor IoU on UNSEEN-TIME frames of the same scene. torch (CUDA), "
    "torchvision, numpy available. Do not read any test-set bev GT.")


def smartspace_impl_task(threshold: float = 0.166, *, train_max=None, test_max=None) -> ImplementationTask:
    """Smart-space Track-B task (framework=torch -> vo-gpu-torch:1). Graded by eval_smartspace.py
    (held-out floor IoU). Default threshold = reference/1.3 (reference IPM ~0.216)."""
    from ..plugins.smartspace import smartspace_datasets
    return ImplementationTask(
        description=SMARTSPACE_TASK_DESCRIPTION, framework=_GPU_FW,
        entry_command='timeout 3600 python3 "$LAB_CODE/main.py"', eval_command='python3 "$LAB_CODE/eval.py"',
        eval_code=_SS_EVAL_CODE, metric="miou", op=">=", threshold=threshold,
        datasets=smartspace_datasets(), entry_filename="main.py")


def smartspace_reference_author():
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(_SS_REFERENCE_MAIN)
        return Usage()
    return author


def smartspace_degenerate_author():
    src = ("import os, glob, numpy as np\n"
           "d=os.environ['LAB_DATA']; a=os.environ['LAB_ARTIFACTS']; os.makedirs(a,exist_ok=True)\n"
           "for f in glob.glob(os.path.join(d,'test_input','*.npz')):\n"
           "    tok=os.path.splitext(os.path.basename(f))[0]\n"
           "    g=np.load(f)['grid_bounds']\n"
           "    import numpy as _np; XG=int(round((g[1]-g[0])/0.5)); YG=int(round((g[3]-g[2])/0.5))\n"
           "    np.save(os.path.join(a,f'pred_{tok}.npy'), np.zeros((XG,YG),np.uint8))\n")
    def author(task, code_dir: Path, rec) -> Usage:
        (Path(code_dir) / "main.py").write_text(src)
        return Usage()
    return author
