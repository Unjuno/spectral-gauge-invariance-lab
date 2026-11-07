"""Reproducible fixed-protocol study of diagonal gauge / correlation control.

All mathematical quantities dimensionless. The positive diagonal gauge acts at the
FINAL ReLU layer, not on upstream nonlinear residual components. See PROTOCOL.md.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

D_IN=24
D_FEATURE=24
EPS=1e-16
LAM=0.5

class Net(nn.Module):
    def __init__(self, classes:int, architecture:str):
        super().__init__()
        assert architecture in ('plain','residual')
        self.architecture=architecture
        self.l1=nn.Linear(D_IN,48)
        if architecture=='residual':
            self.res=nn.Linear(48,48)
        self.l2=nn.Linear(48,D_FEATURE)
        self.out=nn.Linear(D_FEATURE,classes)

    def forward(self,x):
        a=F.relu(self.l1(x))
        if self.architecture=='residual':
            a=F.relu(a+0.5*F.gelu(self.res(a)))
        h=F.relu(self.l2(a))
        return self.out(h),h


def diag_scale(name:str,*,dtype=torch.float64):
    if name=='identity':
        return torch.ones(D_FEATURE,dtype=dtype)
    if name=='heterogeneous':
        return torch.tensor([.25,1.,4.]*(D_FEATURE//3),dtype=dtype)
    raise ValueError(name)


def gauge(model:Net,name:str):
    net=copy.deepcopy(model)
    s=diag_scale(name,dtype=net.l2.weight.dtype)
    with torch.no_grad():
        net.l2.weight.mul_(s[:,None])
        net.l2.bias.mul_(s)
        net.out.weight.div_(s[None,:])
    return net


def stats(h:torch.Tensor):
    z=h-h.mean(0,keepdim=True)
    cov=z.T@z/h.shape[0]
    var=cov.diag()
    active=var>EPS
    inv=torch.rsqrt(var.clamp_min(EPS))*active.to(h.dtype)
    corr=cov*inv[:,None]*inv[None,:]
    d=h.shape[1]
    mask=1-torch.eye(d,dtype=h.dtype,device=h.device)
    corr_pen=((corr*mask)**2).sum()/(d*(d-1))
    denom=(var.sum()/d).clamp_min(EPS)
    cov_pen=(((cov/denom)*mask)**2).sum()/(d*(d-1))
    def prank(m):
        return float((torch.trace(m).square()/((m*m).sum().clamp_min(1e-50))).detach()) if float(torch.trace(m).detach())>1e-20 else 0.
    return dict(corr=corr,corr_pen=corr_pen,cov_pen=cov_pen,trace=float(var.sum().detach()),
                active=int(active.sum()),near_guard=int(((var>0)&(var<=EPS*16)).sum()),
                corr_rank=prank(corr),cov_rank=prank(cov))


def obj(logits,h,y,kind:str):
    ce=F.cross_entropy(logits,y)
    if kind=='ce':
        return ce
    s=stats(h)
    return ce+LAM*(s['corr_pen'] if kind=='corr' else s['cov_pen'])


def update(model,tx,ty,rate,kind,variant):
    model.zero_grad(set_to_none=True)
    logits,h=model(tx)
    loss=obj(logits,h,ty,kind)
    if not bool(torch.isfinite(loss)):
        return False
    loss.backward()
    s=diag_scale('heterogeneous' if variant=='corrected' else 'identity')
    with torch.no_grad():
        for name,p in model.named_parameters():
            g=p.grad
            if g is None or not bool(torch.isfinite(g).all()):
                return False
            if variant=='corrected':
                if name=='l2.weight':g=g*s[:,None].square()
                elif name=='l2.bias':g=g*s.square()
                elif name=='out.weight':g=g/s[None,:].square()
            p.add_(g,alpha=-rate)
    return True


def _raw(task:str, n:int,seed:int, *,ood=False):
    rng=np.random.default_rng(seed)
    if task=='moons24':
        x2,y=make_moons(n_samples=n,noise=0.25 if not ood else 0.36,random_state=seed)
        if ood:
            angle=math.radians(18)
            rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            center=np.array([.5,.25])
            x2=(x2-center)@rot.T+center
        nuisance=rng.normal(loc=0.0 if not ood else 1.2,scale=1.0 if not ood else 1.55,size=(n,22))
        x=np.column_stack([x2,nuisance])
    elif task=='prototypes4':
        centers=np.random.default_rng(9901).normal(size=(4,8))*1.7
        base=rng.integers(0,4,size=n)
        # 8 latent causal coordinates expanded into 18 observable correlated channels,
        # and 6 independent nuisance channels. Same class centroids in ID/OOD.
        z=centers[base]+rng.normal(scale=0.75 if not ood else 1.25,size=(n,8))
        mat=np.random.default_rng(9902).normal(size=(8,18))
        q,_=np.linalg.qr(mat.T)
        xcore=z@q.T
        junk=rng.normal(loc=0.0 if not ood else 1.2,scale=1.0 if not ood else 1.7,size=(n,6))
        x=np.column_stack([xcore,junk]);y=base
    else:raise ValueError(task)
    return x.astype(np.float64),np.asarray(y,dtype=np.int64)


def dataset(task:str,include_holdout=False):
    # These first 2160 examples feed train/val/ID test, with disjoint fixed indices.
    x,y=_raw(task,2160,seed=2026110801 if task=='moons24' else 2026110802)
    a,b=train_test_split(np.arange(len(y)),train_size=640,stratify=y,random_state=8201)
    # Fix val=320 and ID holdout=1200, without touching holdout tensors during training.
    # Instead select exactly 320 validation samples from 1520 unused indices.
    va,id_test=train_test_split(b,train_size=320,stratify=y[b],random_state=8202)
    assert len(a)==640 and len(va)==320 and len(id_test)==1200
    assert len(set(a)&set(va))==len(set(a)&set(id_test))==len(set(va)&set(id_test))==0
    scaler=StandardScaler().fit(x[a])
    def pack(xx,yy):
        return (torch.tensor(scaler.transform(xx),dtype=torch.float64),torch.tensor(yy,dtype=torch.long))
    result={'train':pack(x[a],y[a]),'val':pack(x[va],y[va]),'classes':int(np.max(y)+1)}
    if include_holdout:
        result['id']=pack(x[id_test],y[id_test])
        ox,oy=_raw(task,1200,seed=2026110803 if task=='moons24' else 2026110804,ood=True)
        result['ood']=pack(ox,oy)
    return result


def evaluate_model(model,x,y):
    with torch.no_grad():
        logits,h=model(x)
        yh=logits.argmax(1)
        cm=[float((yh[y==cl]==cl).double().mean()) for cl in torch.unique(y)]
        st=stats(h)
        return {'ce':float(F.cross_entropy(logits,y)),'balanced_acc':float(np.mean(cm)),
          'corr_pen':float(st['corr_pen']),'corr_rank':st['corr_rank'],
          'cov_rank':st['cov_rank'],'feature_trace':st['trace'],
          'active':st['active'],'near_guard':st['near_guard']}
