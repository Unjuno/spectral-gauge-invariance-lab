import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from core import Net,gauge,stats,obj,update,dataset

torch.set_num_threads(1)


def test_holdout_is_optional_and_ood_is_distinct():
    for task in ('moons24','prototypes4'):
        tr=dataset(task,False)
        d=dataset(task,True)
        assert 'ood' not in tr and 'id' not in tr
        assert all(x[0].shape[1]==24 for x in (d['train'],d['val'],d['id'],d['ood']))
        assert len(d['train'][1])==640 and len(d['val'][1])==320
        assert len(d['id'][1])==len(d['ood'][1])==1200
        assert torch.equal(tr['train'][0],d['train'][0])
        assert not torch.allclose(d['id'][0].mean(0),d['ood'][0].mean(0),atol=0.10)


def test_initial_function_and_correlation_equal_both_architectures():
    tx,ty=dataset('moons24')['train']
    for arch in ('plain','residual'):
        torch.manual_seed(100)
        m=Net(2,arch).double()
        t=gauge(m,'heterogeneous')
        with torch.no_grad():
            l,h=m(tx)
            lp,hp=t(tx)
            assert torch.max(torch.abs(l-lp))<1e-12
            assert torch.max(torch.abs(stats(h)['corr']-stats(hp)['corr']))<1e-12
            assert abs(float(obj(l,h,ty,'corr'))-float(obj(lp,hp,ty,'corr')))<1e-12


def test_compensation_restores_ce_and_corr_paths():
    tx,ty=dataset('moons24')['train']
    for arch in ('plain','residual'):
      for kind in ('ce','corr'):
        torch.manual_seed(100)
        m=Net(2,arch).double()
        t=gauge(m,'heterogeneous')
        for _ in range(5):
            assert update(m,tx,ty,.07,kind,'sgd')
            assert update(t,tx,ty,.07,kind,'corrected')
            with torch.no_grad():
                assert (m(tx)[0]-t(tx)[0]).abs().max()<1e-10


def test_covariance_penalty_breaks_invariance():
    tx,ty=dataset('moons24')['train']
    torch.manual_seed(100)
    m=Net(2,'plain').double()
    t=gauge(m,'heterogeneous')
    with torch.no_grad():
        ml,mh=m(tx); tl,th=t(tx)
        assert abs(float(obj(ml,mh,ty,'cov'))-float(obj(tl,th,ty,'cov')))>1e-5


def test_residual_architecture_not_equivalent_to_plain():
    torch.manual_seed(100)
    m=Net(2,'plain').double()
    torch.manual_seed(100)
    t=Net(2,'residual').double()
    assert len(list(t.parameters()))>len(list(m.parameters()))
