import torch, torch.nn as nn
from diffsynth.models.wan_video_dit import MemoryRetriever, CrossAttention
dev="cuda"; B=1; dim=1024; ditdim=1024; Tkey=4; L=4; tok=782; Nmem=Tkey*L*tok
class StoreBuilder(nn.Module):           # the TRAINABLE store (what we want grads to reach)
    def __init__(s): super().__init__(); s.proj=nn.Linear(64,1024)
    def forward(s,feat): return s.proj(feat)        # [B,Nmem,64]->[B,Nmem,1024], in-graph
torch.manual_seed(0)
store=StoreBuilder().to(dev); retr=MemoryRetriever(dim=dim).to(dev)
mememb=nn.Linear(1024,ditdim).to(dev); xattn=CrossAttention(dim=ditdim,num_heads=8).to(dev)
feat=torch.randn(B,Nmem,64,device=dev); tgt=torch.randn(B,1,9,device=dev); key=torch.randn(B,Tkey,9,device=dev)
x=torch.randn(B,tok,ditdim,device=dev)
mem=store(feat)                                      # in-graph store
retrieved=retr(tgt,key,mem)                          # QKV retrieval
print("retrieved shape", tuple(retrieved.shape))
ctx=mememb(retrieved)
try: out=xattn(x,ctx)
except TypeError as e:
    import inspect; print("xattn sig:", inspect.signature(xattn.forward)); raise
loss=out.float().mean(); loss.backward()
g=store.proj.weight.grad
print("loss", round(loss.item(),5))
print("STORE grad norm:", None if g is None else round(g.norm().item(),6))
ok = g is not None and g.norm().item()>0
print("GRAD_FLOW_OK" if ok else "NO_GRAD_TO_STORE")
