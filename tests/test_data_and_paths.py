import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from experiment import Net, dataset, train_step, apply_gauge

def test_splits_independent_and_complete():
    for name in ('classification6','moons'):
        d=dataset(name,include_test=True)
        a,b,c=map(set,(d['train_indices'],d['val_indices'],d['test_indices']))
        assert len(a)==800 and len(b)==400 and len(c)==2400
        assert not (a&b or a&c or b&c)
        assert len(a|b|c)==3600
        assert d['train'][0].dtype==torch.float64

def test_function_preservation_after_twenty_compensated_updates():
    torch.manual_seed(88)
    x=torch.randn(90,24,dtype=torch.float64)
    y=torch.arange(90)%6
    net=Net(6).double()
    alt=apply_gauge(net,'shrink')
    for _ in range(20):
        train_step(net,x,y,.025,False,None)
        train_step(alt,x,y,.025,True,'shrink')
    torch.testing.assert_close(net(x)[0],alt(x)[0],rtol=1e-9,atol=1e-9)
