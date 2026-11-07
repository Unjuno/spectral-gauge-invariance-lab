import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from experiment import Net, apply_gauge, train_step, spectral_metrics

def test_predictions_unchanged_under_positive_diagonal_gauge():
    torch.manual_seed(15)
    base=Net(6)
    x=torch.randn(60,24,dtype=torch.float64)
    base=base.double()
    ref=base(x)[0]
    scaled=apply_gauge(base,'heterogeneous')
    torch.testing.assert_close(ref,scaled(x)[0],rtol=1e-12,atol=1e-12)

def test_compensated_sgd_recovers_canonical_one_step():
    torch.manual_seed(31)
    x=torch.randn(50,24,dtype=torch.float64)
    y=torch.arange(50)%6
    base=Net(6).double()
    transformed=apply_gauge(base,'heterogeneous')
    train_step(base,x,y,0.1,compensated=False,scales=None)
    train_step(transformed,x,y,0.1,compensated=True,scales='heterogeneous')
    torch.testing.assert_close(base(x)[0],transformed(x)[0],rtol=1e-11,atol=1e-11)

def test_spectrum_changes_but_is_finite():
    torch.manual_seed(9)
    net=Net(6).double();x=torch.randn(100,24,dtype=torch.float64)
    m0=spectral_metrics(net(x)[1])
    m1=spectral_metrics(apply_gauge(net,'heterogeneous')(x)[1])
    assert m0['trace']>0 and m1['trace']>0
    assert abs(m0['effective_rank']-m1['effective_rank'])>0.05
    assert 0<=m0['top_fraction']<=1
