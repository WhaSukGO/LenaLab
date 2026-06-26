"""Cross-viewpoint result figure. CPU-only."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="artifacts/learned3d_wedge/figs"; os.makedirs(OUT,exist_ok=True)
GREEN="#2e8b57"; RED="#c0392b"; BLUE="#2c6fbb"; GREY="#7f8c8d"

views=[4,8,12,16,19]; outside=[False,False,False,True,True]
froz=[0.553243,0.641239,0.690624,0.737630,0.697986]
lrn =[0.535211,0.544509,0.555693,0.589344,0.552181]
hist=np.array([[0,0.631676,0.662392],[5,0.624890,0.655266],[10,0.619919,0.648922],[15,0.613856,0.641047],
 [20,0.606705,0.631364],[25,0.596566,0.616871],[30,0.585997,0.600769],[35,0.572031,0.584252],
 [40,0.560870,0.573139],[45,0.553268,0.565582],[50,0.548518,0.561566],[55,0.545427,0.558769],
 [60,0.543470,0.557125],[65,0.542183,0.556028]])

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
x=np.arange(5); w=0.38
b1=a1.bar(x-w/2,froz,w,color=RED,alpha=0.85,label="FROZEN store")
b2=a1.bar(x+w/2,lrn,w,color=GREEN,alpha=0.9,label="LEARNED store")
a1.set_xticks(x)
a1.set_xticklabels([f"view {v}\n{'OUT-of-window' if o else 'in-window'}" for v,o in zip(views,outside)],fontsize=8.5)
a1.set_ylabel("held-out denoising loss (lower=better)")
a1.set_title("Held-out VIEWPOINTS: 5/5 improved\nmean 0.664 → 0.555 (+16.4%); out-of-window +20.5%",fontsize=11)
a1.legend(fontsize=9); a1.grid(alpha=0.3,axis="y")
for i in range(5):
    a1.text(x[i]+w/2,lrn[i]+0.012,f"+{100*(froz[i]-lrn[i])/froz[i]:.0f}%",ha="center",color=GREEN,fontsize=8.5,fontweight="bold")
# shade out-of-window region
a1.axvspan(2.5,4.5,color="#f0c0c0",alpha=0.18)
a1.text(3.5,0.05,"genuinely NEW\nviewpoints",ha="center",fontsize=8,color="#933",style="italic")

a2.plot(hist[:,0],hist[:,1],"-o",color=GREEN,ms=3,label="LEARNED — train viewpoints")
a2.plot(hist[:,0],hist[:,2],"-o",color=BLUE,ms=3,label="LEARNED — held-out viewpoints")
a2.axhline(0.631676,ls="--",color=GREY,lw=1.5,label="FROZEN — train (0.632)")
a2.axhline(0.664144,ls="--",color=RED,lw=1.5,label="FROZEN — held-out (0.664)")
a2.set_xlabel("optimization step"); a2.set_ylabel("denoising loss")
a2.set_title("Held-out-viewpoint loss drops below frozen\n(held-out gain +16.4% > train gain +14.3% = generalizes)",fontsize=11)
a2.legend(fontsize=8,loc="upper right"); a2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_crossview.png",dpi=130,bbox_inches="tight"); plt.close()
print("wrote fig4_crossview.png")
