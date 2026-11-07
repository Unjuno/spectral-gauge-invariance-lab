import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from core import Net, apply_gauge, covariance_stats, objective, step, synthetic_data


def test_labelled_splits_do_not_overlap_and_test_is_not_default():
    d = synthetic_data('classification6', include_test=False)
    assert 'test' not in d
    assert len(d['train_indices']) == 800
    assert len(d['val_indices']) == 400
    assert len(d['test_indices']) == 2400
    assert set(d['train_indices']).isdisjoint(d['test_indices'])
    assert set(d['val_indices']).isdisjoint(d['test_indices'])
    assert set(d['val_indices']).isdisjoint(d['train_indices'])


def test_corr_invariant_covariance_rank_changes_and_function_invariant():
    torch.manual_seed(123)
    x = torch.randn(100, 24, dtype=torch.float64)
    m = Net(6).double()
    gauge = apply_gauge(m, 'heterogeneous')
    f,h = m(x)
    f2,h2 = gauge(x)
    torch.testing.assert_close(f, f2, atol=1e-12, rtol=1e-12)
    s = covariance_stats(h)
    s2 = covariance_stats(h2)
    torch.testing.assert_close(s['corr'], s2['corr'], atol=1e-11, rtol=1e-11)
    assert abs(float((s['cov_rank'] - s2['cov_rank']).detach())) > 0.05
    torch.testing.assert_close(objective(f,h,torch.arange(100)%6,'corr')[1],objective(f2,h2,torch.arange(100)%6,'corr')[1],atol=1e-12,rtol=1e-12)
    assert abs(float((objective(f,h,torch.arange(100)%6,'cov')[1]-objective(f2,h2,torch.arange(100)%6,'cov')[1]).detach()))>1e-5


def test_corr_invariance_with_zero_variance_channels_has_finite_gradient():
    h = torch.zeros(10, 24, dtype=torch.float64, requires_grad=True)
    r = covariance_stats(h)['corr_offdiag']
    r.backward()
    assert torch.isfinite(h.grad).all()
    assert float(r.detach()) == 0.0


def test_compensated_sgd_exact_path_with_corr_regularizer():
    torch.manual_seed(65)
    x = torch.randn(80, 24, dtype=torch.float64)
    y = torch.arange(80)%6
    reference = Net(6).double()
    gauged = apply_gauge(reference, 'heterogeneous')
    for i in range(25):
        step(reference, x, y, lr=.05, mode='corr',compensated=False, gauge='identity')
        step(gauged, x, y, lr=.05, mode='corr',compensated=True, gauge='heterogeneous')
    torch.testing.assert_close(reference(x)[0], gauged(x)[0], atol=1e-10, rtol=1e-10)


def test_covariance_objective_breaks_compensated_equivalence():
    torch.manual_seed(65)
    x = torch.randn(80, 24, dtype=torch.float64)
    y = torch.arange(80)%6
    reference = Net(6).double()
    gauged = apply_gauge(reference, 'heterogeneous')
    for i in range(5):
        step(reference, x, y, lr=.05, mode='cov',compensated=False,gauge='identity')
        step(gauged, x, y, lr=.05, mode='cov',compensated=True,gauge='heterogeneous')
    difference = float((reference(x)[0] - gauged(x)[0]).abs().max().detach())
    assert difference > 1e-7, difference
