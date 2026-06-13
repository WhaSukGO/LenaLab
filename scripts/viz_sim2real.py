import sys; sys.path.insert(0,"/home/ws/devel/whasuk/LenaLab")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
R=Path("/home/ws/devel/whasuk/LenaLab")

def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src-mu_s, dst-mu_d
    S = (d.T@s)/len(src)
    U,D,Vt = np.linalg.svd(S)
    Rm = U@Vt
    if np.linalg.det(Rm)<0: U[:,-1]*=-1; Rm=U@Vt
    var = (s**2).sum()/len(src)
    c = (D.sum())/var if var>1e-12 else 1.0
    t = mu_d - c*Rm@mu_s
    return (c*(Rm@src.T)).T + t

def load(p): 
    a=np.loadtxt(p); return a.reshape(-1,3) if a.ndim>1 else a.reshape(1,3)

panels=[]  # (title, gt, pred, ate)
# in-domain synthetic
synth_art=R/"_vo_synth_learned_impl_run/workspaces/synth-learned-impl-001/artifacts"
synth_ho=R/"_vo_synth_learned_impl_run/cache/heldout"
for s,ate in [("lte2",0.18)]:
    g=next(synth_ho.rglob(f"seq_{s}/gt.txt")); t=synth_art/f"traj_{s}.txt"
    if t.exists(): panels.append((f"SYNTHETIC seq_{s} (in-domain)\nATE {ate:.2f} m  — tracks", load(g), load(t), ate))
# sim-to-real KITTI
k_art=R/"_vo_sim2real_run/workspaces/sim2real-kitti-001/artifacts"
k_ho=R/"_vo_sim2real_run/cache/heldout"
for s,ate in [("07",72.35),("09",66.78)]:
    g=next(k_ho.rglob(f"seq_{s}/gt.txt")); t=k_art/f"traj_{s}.txt"
    panels.append((f"REAL KITTI seq_{s} (sim-to-real)\nATE {ate:.1f} m  — collapses", load(g), load(t), ate))

fig,axes=plt.subplots(1,len(panels),figsize=(5.2*len(panels),5))
for ax,(title,gt,pred,ate) in zip(axes,panels):
    n=min(len(gt),len(pred)); gt,pred=gt[:n],pred[:n]
    al=umeyama_sim3(pred,gt)
    ax.plot(gt[:,0],gt[:,2],"k-",lw=2.5,label="ground truth",zorder=3)
    ax.plot(al[:,0],al[:,2],"r--",lw=1.8,label="learned VO (Sim3-aligned)",zorder=4)
    ax.scatter([gt[0,0]],[gt[0,2]],c="g",s=60,zorder=5,label="start")
    ax.set_title(title,fontsize=11); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.axis("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Rung 3 learned VO: generalises on UNSEEN SYNTHETIC (0.45 m) but does NOT transfer to REAL photos (~70 m)\n"
             "the textbook sim-to-real appearance gap — same model, ~150x worse on real KITTI",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.9])
out=R/"artifacts/blog/sim2real_kitti.png"; fig.savefig(out,dpi=120); print("wrote",out)
