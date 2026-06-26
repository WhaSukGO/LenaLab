"""Generation-quality result figure (the B+C verdict). CPU-only."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="artifacts/learned3d_wedge/figs"; os.makedirs(OUT,exist_ok=True)
GREEN="#2e8b57"; RED="#c0392b"

views=[4,8,12,16,18,19]; outside=[0,0,0,1,1,1]
f_lpips=[0.4019,0.5257,0.6349,0.7104,0.7489,0.7711]
l_lpips=[0.4378,0.5360,0.6327,0.6923,0.7340,0.7535]
f_psnr=[18.34,14.11,11.85,10.29,9.69,9.21]
l_psnr=[16.78,13.83,12.17,10.58,9.84,9.20]
x=np.arange(len(views))

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
a1.plot(x,f_lpips,"-o",color=RED,label="FROZEN store")
a1.plot(x,l_lpips,"-o",color=GREEN,label="LEARNED store")
a1.axvspan(2.5,5.5,color="#f0c0c0",alpha=0.18); a1.text(4,0.43,"held-out\n(outside window)",ha="center",fontsize=8,color="#933",style="italic")
a1.set_xticks(x); a1.set_xticklabels([f"v{v}" for v in views])
a1.set_xlabel("viewpoint (→ farther from key window)"); a1.set_ylabel("LPIPS ↓ vs real frame (lower=better)")
a1.set_title("LPIPS: learned ≈ frozen (lines overlap)\nquality COLLAPSES with distance — 0.40→0.77",fontsize=11)
a1.legend(fontsize=9); a1.grid(alpha=0.3)

a2.plot(x,f_psnr,"-o",color=RED,label="FROZEN store")
a2.plot(x,l_psnr,"-o",color=GREEN,label="LEARNED store")
a2.axvspan(2.5,5.5,color="#f0c0c0",alpha=0.18)
a2.axhline(20,ls=":",color="#888",lw=1); a2.text(0.1,20.2,"~20 = a faithful match would be here+",fontsize=7.5,color="#666")
a2.set_xticks(x); a2.set_xticklabels([f"v{v}" for v in views])
a2.set_xlabel("viewpoint (→ farther from key window)"); a2.set_ylabel("PSNR ↑ vs real frame (higher=better)")
a2.set_title("PSNR: even in-window only ~18; far views ~9-10\nbase model can't reproduce the viewpoint",fontsize=11)
a2.legend(fontsize=9); a2.grid(alpha=0.3)
plt.suptitle("Generation-quality test (CFG on) — VERDICT B+C: loss gain doesn't translate to pixels; base model is the ceiling",
             fontsize=12,fontweight="bold",y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_genquality.png",dpi=130,bbox_inches="tight"); plt.close()
print("wrote fig5_genquality.png")
