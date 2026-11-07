"""Minimal reproducible experiment for diagonal-gauge invariant feature correlation control.

ReLU positive diagonal gauge: W2'=D W2, b2'=D b2, V'=V D^-1.
With gauge-compensated SGD, gradient multipliers are D^2 on W2,b2 and D^-2 on V.
All input/features/losses are treated as dimensionless.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

VAR_EPS = 1.e-16
FEATURE_DIM = 24
LAMBDA = 0.5

class Net(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.l1 = nn.Linear(24, 64)
        self.l2 = nn.Linear(64, FEATURE_DIM)
        self.out = nn.Linear(FEATURE_DIM, n_classes)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.l2(F.relu(self.l1(x))))
        return self.out(h), h


def scales(name: str, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if name == 'identity':
        return torch.ones(FEATURE_DIM, dtype=dtype)
    if name == 'heterogeneous':
        return torch.tensor([.25,1.,4.]*8, dtype=dtype)
    raise ValueError(name)


def apply_gauge(net: Net, gauge: str):
    copy_net = copy.deepcopy(net)
    s = scales(gauge, copy_net.l2.weight.dtype)
    with torch.no_grad():
        copy_net.l2.weight.mul_(s[:,None])
        copy_net.l2.bias.mul_(s)
        copy_net.out.weight.div_(s[None,:])
    return copy_net


def covariance_stats(h:torch.Tensor, include_ranks:bool=True):
    """Channel correlation is positive-diagonal-gauge invariant if variances stay above guard.

    Dead channels are assigned zero row/column. Exact invariance at nonzero channels
    depends on variance > VAR_EPS for both gauge versions.
    """
    z = h-h.mean(dim=0, keepdim=True)
    cov = z.T @ z / h.shape[0]
    var = torch.diagonal(cov)
    active = (var > VAR_EPS)
    invsd = torch.rsqrt(torch.clamp(var, min=VAR_EPS)) * active.to(dtype=h.dtype)
    corr = cov * invsd[:,None] * invsd[None,:]
    d = h.shape[1]
    offdiag = 1-torch.eye(d, dtype=h.dtype, device=h.device)
    corr_offdiag = ((corr*offdiag)**2).sum()/(d*(d-1))
    avg_var = (var.sum()/d).clamp_min(VAR_EPS)
    norm_cov = cov / avg_var
    cov_offdiag = ((norm_cov*offdiag)**2).sum()/(d*(d-1))
    r = dict(cov=cov,corr=corr,corr_offdiag=corr_offdiag,cov_offdiag=cov_offdiag,
             active_channels=active.sum(),borderline_channels=((var>0)&(var <= VAR_EPS*16)).sum(),
             cov_trace=var.sum())
    if include_ranks:
        def pr(mat):
            tr = torch.trace(mat)
            return torch.where(tr > 1e-20, tr**2/((mat*mat).sum().clamp_min(1e-40)), tr.new_zeros(()))
        r['cov_rank'] = pr(cov)
        r['corr_rank'] = pr(corr)
    return r


def objective(logits, h, y, mode: str):
    ce = F.cross_entropy(logits,y)
    if mode == 'ce':
        reg = h.new_zeros(())
    elif mode in ('corr','cov'):
        stats=covariance_stats(h, include_ranks=False)
        reg = stats['corr_offdiag'] if mode=='corr' else stats['cov_offdiag']
    else:
        raise ValueError(mode)
    return ce + (LAMBDA*reg), reg


def step(net, x, y, lr:float, mode:str, compensated:bool, gauge:str):
    net.zero_grad(set_to_none=True)
    logits,h=net(x)
    total,reg=objective(logits,h,y,mode)
    if not bool(torch.isfinite(total)):
        return {'loss':float('inf'),'reg':float('nan'),'finite':False}
    total.backward()
    s=scales(gauge,net.l2.weight.dtype)
    with torch.no_grad():
        for name,p in net.named_parameters():
            g=p.grad
            if not bool(torch.isfinite(g).all()):
                return {'loss':float('inf'),'reg':float('nan'),'finite':False}
            if compensated:
                if name=='l2.weight': g = g*(s[:,None]**2)
                elif name=='l2.bias': g = g*(s**2)
                elif name=='out.weight': g = g/(s[None,:]**2)
            p.add_(g,alpha=-lr)
    return {'loss':float(total.detach()),'reg':float(reg.detach()),'finite':True}


def synthetic_data(task: str, include_test:bool=False)->dict[str, Any]:
    if task=='classification6':
        x,y=make_classification(n_samples=3600,n_features=24,n_informative=10,
            n_redundant=8,n_classes=6,n_clusters_per_class=1,class_sep=1.0,
            flip_y=.08,random_state=20261121)
    elif task=='moons22':
        informative,y=make_moons(n_samples=3600,noise=.30,random_state=20261122)
        noise=np.random.default_rng(20261125).normal(size=(3600,22))
        x=np.concatenate([informative, noise],axis=1)
    else:
        raise ValueError(task)
    ix=np.arange(len(y))
    trainval, test = train_test_split(ix,train_size=1200,stratify=y,random_state=20261123)
    train, val = train_test_split(trainval,train_size=800,stratify=y[trainval],random_state=20261124)
    scaler=StandardScaler().fit(x[train])
    def pack(indices):
        return torch.tensor(scaler.transform(x[indices]),dtype=torch.float64),torch.tensor(y[indices],dtype=torch.long)
    out={'train':pack(train),'val':pack(val),'train_indices':train,'val_indices':val,'test_indices':test,'classes':len(np.unique(y))}
    if include_test:
        out['test']=pack(test)
    return out


def model_metrics(net, x, y, prefix=''):
    with torch.no_grad():
        logits,h=net(x)
        pred=logits.argmax(1)
        per_class=[(pred[y==c]==c).double().mean().item() for c in torch.unique(y)]
        st=covariance_stats(h)
        return {
            prefix+'ce':float(F.cross_entropy(logits,y)),
            prefix+'balanced_accuracy':float(np.mean(per_class)),
            prefix+'corr_penalty':float(st['corr_offdiag']),
            prefix+'cov_penalty':float(st['cov_offdiag']),
            prefix+'corr_rank':float(st['corr_rank']),
            prefix+'cov_rank':float(st['cov_rank']),
            prefix+'active_channels':int(st['active_channels']),
            prefix+'borderline_channels':int(st['borderline_channels']),
            prefix+'trace':float(st['cov_trace']),
        }
