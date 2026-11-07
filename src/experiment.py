"""Gauge transformations and paired optimization experiment.

A diagonal positive scaling of a ReLU feature layer and the inverse scaling
of the following linear layer preserve network outputs exactly in real arithmetic.
The compensated step is the exact coordinate transform of ordinary SGD.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

GAUGES = ('identity', 'shrink', 'expand', 'heterogeneous')
OPTIMIZERS = ('vanilla_sgd', 'gauge_compensated_sgd')
LEARNING_RATES = (0.08, 0.25)
SEEDS = tuple(range(6100, 6112))
STEPS = 100
WARMUP_STEPS = 15


class Net(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.l1 = nn.Linear(24, 64)
        self.l2 = nn.Linear(64, 24)
        self.out = nn.Linear(24, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = F.relu(self.l1(x))
        h = F.relu(self.l2(u))
        return self.out(h), h


def scale_vector(kind: str, dtype=torch.float64) -> torch.Tensor:
    if kind == 'identity': values = [1.] * 24
    elif kind == 'shrink': values = [.25] * 24
    elif kind == 'expand': values = [4.] * 24
    elif kind == 'heterogeneous': values = [.25, 1., 4.] * 8
    else: raise ValueError(kind)
    return torch.tensor(values, dtype=dtype)


def apply_gauge(net: Net, kind: str) -> Net:
    model = copy.deepcopy(net)
    s = scale_vector(kind, model.l2.weight.dtype)
    with torch.no_grad():
        model.l2.weight.mul_(s[:, None])
        model.l2.bias.mul_(s)
        model.out.weight.div_(s[None, :])
    return model


def train_step(net: Net, x: torch.Tensor, y: torch.Tensor, lr: float,
               compensated: bool, scales: str | None) -> float:
    net.zero_grad(set_to_none=True)
    logits, _ = net(x)
    loss = F.cross_entropy(logits, y)
    if not torch.isfinite(loss):
        return float('inf')
    loss.backward()
    s = scale_vector(scales or 'identity', net.l2.weight.dtype)
    with torch.no_grad():
        for name, p in net.named_parameters():
            grad = p.grad
            if not bool(torch.isfinite(grad).all()): return float('inf')
            if compensated:
                if name == 'l2.weight': grad = grad * (s[:, None]**2)
                elif name == 'l2.bias': grad = grad * (s**2)
                elif name == 'out.weight': grad = grad / (s[None, :]**2)
            p.add_(grad, alpha=-lr)
    return float(loss.detach())


@torch.no_grad()
def spectral_metrics(h: torch.Tensor) -> dict[str, float]:
    z = h.double() - h.double().mean(0, keepdim=True)
    sigma = z.T @ z / max(1, len(z))
    eig = torch.linalg.eigvalsh(sigma).clamp(min=0)
    tr = float(eig.sum())
    if tr <= 1e-30:
        return {'trace': tr, 'effective_rank': 0., 'top_fraction': 0., 'entropy_rank': 0.}
    p = (eig / tr).cpu().numpy()
    p_pos = p[p > 0]
    return dict(trace=tr, effective_rank=1 / float((p*p).sum()),
                top_fraction=float(p[-1]), entropy_rank=float(np.exp(-(p_pos*np.log(p_pos)).sum())))


def dataset(name: str, include_test=False):
    """A fixed, stratified, disjoint split. Test array only returned on final evaluation."""
    if name == 'classification6':
        x, y = make_classification(n_samples=3600, n_features=24, n_informative=10,
            n_redundant=8, n_repeated=0, n_classes=6, n_clusters_per_class=1,
            class_sep=1.0, flip_y=.08, random_state=20261107)
    elif name == 'moons':
        informative, y = make_moons(n_samples=3600, noise=.25, random_state=20261107)
        nuisance = np.random.default_rng(20261108).normal(size=(3600,22))
        x = np.hstack([informative, nuisance])
    else:
        raise ValueError(name)
    indices = np.arange(len(y))
    dev, holdout = train_test_split(indices, train_size=1200, stratify=y, random_state=20261109)
    tr, val = train_test_split(dev, train_size=800, stratify=y[dev], random_state=20261110)
    normalizer = StandardScaler().fit(x[tr])
    def pack(index):
        return (torch.tensor(normalizer.transform(x[index]),dtype=torch.float64),
                torch.tensor(y[index],dtype=torch.int64))
    result = {'train':pack(tr), 'validation':pack(val), 'classes':len(np.unique(y)),
              'train_indices':tr, 'val_indices':val, 'test_indices':holdout,
              'n_total':len(y)}
    if include_test: result['test']=pack(holdout)
    return result


@torch.no_grad()
def metrics(model, x, y):
    logits,h=model(x)
    if not bool(torch.isfinite(logits).all()):
        return dict(ce=float('inf'),accuracy=float('nan'),balanced_accuracy=float('nan'),**spectral_metrics(h))
    pred=logits.argmax(1)
    vals=[]
    for cls in y.unique():
        mask=y==cls
        vals.append(float((pred[mask]==y[mask]).double().mean()))
    return dict(ce=float(F.cross_entropy(logits,y)),
                accuracy=float((pred==y).double().mean()),
                balanced_accuracy=float(np.mean(vals)),**spectral_metrics(h))
