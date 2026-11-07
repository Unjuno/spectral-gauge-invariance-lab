"""One-time evaluation of the preregistered, frozen experiment on independent holdout."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment import Net,dataset,metrics,GAUGES,OPTIMIZERS,LEARNING_RATES,SEEDS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'


def bootstrap(x,seed,n=5000):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all(): return (float('nan'),float('nan'))
    r=np.random.default_rng(seed)
    ids=r.integers(0,len(x),(n,len(x)))
    means=x[ids].mean(1)
    return tuple(np.quantile(means,[.025,.975]).tolist())


def main():
    manifest=pd.read_csv(OUT/'model_manifest.csv')
    expected=len(SEEDS)*len(LEARNING_RATES)*len(OPTIMIZERS)*len(GAUGES)*2
    assert len(manifest)==expected, (len(manifest),expected)
    assert len(pd.read_csv(OUT/'training.csv'))==expected
    all_rows=[]
    for dataname in ('classification6','moons'):
        data=dataset(dataname,include_test=True)
        x,y=data['test']
        for row in manifest[manifest.dataset==dataname].itertuples():
            net=Net(data['classes']).double()
            weight=torch.load(ROOT/'models'/row.filename,map_location='cpu',weights_only=True)
            net.load_state_dict(weight)
            score=metrics(net,x,y)
            all_rows.append(dict(dataset=row.dataset,seed=int(row.seed),lr=float(row.lr),
                                 optimizer=row.optimizer,gauge=row.gauge,
                                 test_ce=score['ce'],test_ba=score['balanced_accuracy'],
                                 test_acc=score['accuracy'],test_trace=score['trace'],
                                 test_effective_rank=score['effective_rank'],
                                 test_top_fraction=score['top_fraction']))
    df=pd.DataFrame(all_rows)
    df.to_csv(OUT/'heldout_test.csv',index=False)
    contrasts=[]
    for (dataname,lr),grp in df.groupby(['dataset','lr']):
        ref=grp[(grp.optimizer=='vanilla_sgd')&(grp.gauge=='identity')].set_index('seed')
        assert len(ref)==len(SEEDS)
        for method in OPTIMIZERS:
            for gauge in GAUGES:
                sub=grp[(grp.optimizer==method)&(grp.gauge==gauge)].set_index('seed').loc[ref.index]
                for metric in ('test_ce','test_ba','test_acc'):
                    delta=(sub[metric]-ref[metric]).to_numpy(float)
                    lower,upper=bootstrap(delta,seed=20261112)
                    contrasts.append(dict(dataset=dataname,lr=lr,optimizer=method,gauge=gauge,
                                          metric=metric,mean_diff=float(delta.mean()),
                                          median_diff=float(np.median(delta)),
                                          min_diff=float(delta.min()),max_diff=float(delta.max()),
                                          ci95_low=lower,ci95_high=upper,
                                          seeds=len(delta)))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
    print('one-time holdout evaluated:',len(df),'models; contrasts:',len(contrasts))

if __name__=='__main__':main()
