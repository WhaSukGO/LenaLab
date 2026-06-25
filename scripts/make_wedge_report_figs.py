"""Generate figures for the learned-3D+QKV wedge research-state report. CPU-only (matplotlib)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "artifacts/learned3d_wedge/figs"; os.makedirs(OUT, exist_ok=True)
BLUE="#2c6fbb"; GREEN="#2e8b57"; RED="#c0392b"; ORANGE="#e08e0b"; GREY="#7f8c8d"; PURP="#7d3c98"

# ---------------------------------------------------------------- Fig 1: architecture
def box(ax,x,y,w,h,txt,fc,ec="#333",fs=9,tc="white",bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc,ec=ec,lw=1.4))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,
            fontweight="bold" if bold else "normal",wrap=True)
def arrow(ax,x1,y1,x2,y2,c="#333",ls="-",lw=1.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,color=c,lw=lw,ls=ls))

fig,ax=plt.subplots(figsize=(13,6.2)); ax.set_xlim(0,13); ax.set_ylim(0,6.2); ax.axis("off")
ax.text(6.5,5.95,"Learned-3D-Store + QKV  —  architecture (the wedge)",ha="center",fontsize=14,fontweight="bold")
# input
box(ax,0.2,3.0,1.9,1.0,"INPUT\nclip frames\n+ cameras",BLUE,fs=8.5)
# store builders (two variants)
box(ax,2.6,4.25,3.0,0.95,"FROZEN store-builder\n(StreamVGGT, pre-baked)\ngeometry GIVEN",GREY,fs=8.2)
box(ax,2.6,2.05,3.0,0.95,"★ LEARNED store-builder\n(trainable, in-graph)\ngeometry LEARNED",GREEN,fs=8.2)
arrow(ax,2.1,3.6,2.6,4.6,GREY); arrow(ax,2.1,3.4,2.6,2.6,GREEN)
# 3D store
box(ax,6.1,3.0,2.0,1.0,"3D LATENT STORE\nmemory tokens\n[4x782 x 1024]",PURP,fs=8.2)
arrow(ax,5.6,4.6,6.6,4.0,GREY); arrow(ax,5.6,2.5,6.6,3.0,GREEN)
# retriever (QKV)
box(ax,8.6,3.0,2.0,1.0,"QKV RETRIEVER\nattend(target cam\n-> 3D store)",ORANGE,fs=8.2,tc="white")
arrow(ax,8.1,3.5,8.6,3.5,"#333")
ax.text(8.35,3.75,"query=\ntarget cam",ha="center",fontsize=6.5,color="#333")
# DiT
box(ax,11.0,3.0,1.8,1.0,"DiT (frozen+LoRA)\ncross-attn\nevery block",BLUE,fs=8.2)
arrow(ax,10.6,3.5,11.0,3.5,"#333")
# output + loss
box(ax,11.0,1.1,1.8,0.8,"generated\nframe",BLUE,fs=8.5)
arrow(ax,11.9,3.0,11.9,1.9,"#333")
box(ax,7.6,0.5,3.0,0.8,"flow-matching loss\n(vs real target latent)",RED,fs=8.5)
arrow(ax,11.0,1.5,10.6,0.9,"#333")
# gradient backflow (the crux)
arrow(ax,7.6,0.9,4.1,2.05,GREEN,ls=(0,(4,2)),lw=2.2)
ax.text(5.6,1.15,"gradient → trains the LEARNED store\n(the crux: validated ✓)",
        ha="center",fontsize=8.5,color=GREEN,fontweight="bold")
ax.text(6.5,0.06,"Frozen path = released Captain-Safari baseline.  Learned path (green) = our wedge: the store is trained end-to-end by the loss.",
        ha="center",fontsize=8,color="#555",style="italic")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_architecture.png",dpi=130,bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 2: results (2 panels)
hist=np.array([
 [0,0.431882,0.468327],[10,0.427355,0.464310],[20,0.425459,0.462469],[30,0.423923,0.461023],
 [40,0.422718,0.460022],[50,0.421854,0.459202],[60,0.421160,0.458695],[70,0.420608,0.458262],
 [80,0.420106,0.457872],[90,0.419905,0.457662],[100,0.419449,0.457322],[120,0.418963,0.456949],
 [150,0.418464,0.456616],[200,0.418179,0.456348],[249,0.418095,0.456288]])
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
# panel A: training curve
a1.plot(hist[:,0],hist[:,1],"-o",color=GREEN,ms=3,label="LEARNED store — train")
a1.plot(hist[:,0],hist[:,2],"-o",color=BLUE,ms=3,label="LEARNED store — held-out")
a1.axhline(0.431882,ls="--",color=GREY,lw=1.5,label="FROZEN — train (0.432)")
a1.axhline(0.469014,ls="--",color=RED,lw=1.5,label="FROZEN — held-out (0.469)")
a1.set_xlabel("optimization step"); a1.set_ylabel("denoising loss (lower=better)")
a1.set_title("Training the learned store — it generalizes\n(held-out drops too, not just train)",fontsize=11)
a1.legend(fontsize=8,loc="upper right"); a1.grid(alpha=0.3)
# panel B: per-sample held-out bars
t=["t=720","t=637","t=523","t=357","t=208"]
froz=[0.266492,0.363340,0.487173,0.617943,0.610125]; lrn=[0.256416,0.350861,0.472297,0.602591,0.599276]
x=np.arange(5); w=0.38
a2.bar(x-w/2,froz,w,color=RED,alpha=0.85,label="FROZEN store")
a2.bar(x+w/2,lrn,w,color=GREEN,alpha=0.9,label="LEARNED store")
a2.set_xticks(x); a2.set_xticklabels(t,fontsize=9)
a2.set_ylabel("held-out denoising loss"); a2.set_title("Held-out samples: 5/5 improved\n(mean 0.469 → 0.456, +2.7%)",fontsize=11)
a2.legend(fontsize=9); a2.grid(alpha=0.3,axis="y")
for i in range(5): a2.text(x[i]+w/2,lrn[i]+0.008,"↓",ha="center",color=GREEN,fontsize=11,fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_results.png",dpi=130,bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 3: roadmap
fig,ax=plt.subplots(figsize=(13,4.6)); ax.set_xlim(0,13); ax.set_ylim(0,4.6); ax.axis("off")
ax.text(6.5,4.3,"Research state & roadmap",ha="center",fontsize=14,fontweight="bold")
done=[("Design\nwedge",), ("Coupling\ngate ✓",), ("Gradient\nflow ✓",), ("Real-model\nintegration ✓",),
      ("Backward\nbug fixed ✓",), ("GO/NO-GO\n+2.7% ✓",)]
for i,(t,) in enumerate(done):
    x=0.3+i*2.05; box(ax,x,2.6,1.75,1.0,t,GREEN,fs=8.5)
    if i>0: arrow(ax,x-0.3,3.1,x,3.1,"#333")
ax.text(6.5,2.35,"PHASE 0 — feasibility: COMPLETE (weak GO — learned beats frozen on real held-out, modest +2.7%, noise/timestep axis only)",
        ha="center",fontsize=8.5,color=GREEN,fontweight="bold")
# next steps
nx=[("NEXT (cheap):\ncross-VIEWPOINT test\nmulti-frame clip,\nheld-out view vs frozen",ORANGE),
    ("THEN (big):\ntoon-multiview\npipeline + train\non OUR domain",BLUE),
    ("GOAL:\nstylized world model\nimage+camera ->\nconsistent new view",PURP)]
for i,(t,c) in enumerate(nx):
    x=0.8+i*4.1; box(ax,x,0.5,3.4,1.3,t,c,fs=8.5)
    if i>0: arrow(ax,x-0.7,1.15,x,1.15,"#333")
arrow(ax,6.5,2.6,2.5,1.8,GREY,ls=(0,(3,2)))
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_roadmap.png",dpi=130,bbox_inches="tight"); plt.close()
print("wrote:", os.listdir(OUT))
