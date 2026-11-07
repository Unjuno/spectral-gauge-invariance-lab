"""Data-only secondary analysis. Reads already produced holdout, never tunes runs."""
from pathlib import Path
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent
RESULTS=R/'results'

def main():
    data=pd.read_csv(RESULTS/'holdout.csv')
    train=pd.read_csv(RESULTS/'training.csv')
    assert len(data)==len(train)==432
    keys=['task','arch','seed','lr','objective']
    assert not data.duplicated(keys+['gauge','optimizer']).any()
    output=[]
    for (task,arch,lr), grp in data.groupby(['task','arch','lr'],sort=True):
        def ids(kind):
            return grp[(grp.objective==kind)&(grp.gauge=='identity')&(grp.optimizer=='sgd')].set_index('seed').sort_index()
        base=ids('ce')
        for objective in ('corr','cov'):
            other=ids(objective)
            d=(other['ood_ce']-base['ood_ce']).to_numpy()
            d_id=(other['id_ce']-base['id_ce']).to_numpy()
            d_ba=(other['ood_balanced_acc']-base['ood_balanced_acc']).to_numpy()
            assert len(d)==6
            rng=np.random.default_rng(6400+{'corr':1,'cov':2}[objective]+int(1000*lr)+(0 if task=='moons24' else 100)+(0 if arch=='plain' else 200))
            bootstrap=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
            output.append(dict(task=task,arch=arch,lr=lr,objective=objective,n_seeds=6,
                ood_ce_diff_mean=float(d.mean()),ood_ce_diff_median=float(np.median(d)),
                ci95_low=float(np.quantile(bootstrap,.025)),ci95_high=float(np.quantile(bootstrap,.975)),
                id_ce_diff_mean=float(d_id.mean()),ood_ba_diff_pp=float(d_ba.mean()*100),
                ood_ce_reference_mean=float(base['ood_ce'].mean()),
                ood_ce_objective_mean=float(other['ood_ce'].mean()),
                corr_rank_delta=float((other['ood_corr_rank']-base['ood_corr_rank']).mean()),
                corr_penalty_delta=float((other['ood_corr_pen']-base['ood_corr_pen']).mean())))
    pd.DataFrame(output).to_csv(RESULTS/'paired_effects.csv',index=False)
    s=[]
    for (obj,g,opt),grp in train.groupby(['objective','gauge','optimizer']):
        s.append(dict(objective=obj,gauge=g,optimizer=opt,n=len(grp),
            path_max=float(grp.max_path_diff.max()),path_median=float(grp.max_path_diff.median()),
            finite=bool(grp.finite.all()),near_guard_max=int(grp.val_near_guard.max())))
    pd.DataFrame(s).to_csv(RESULTS/'path_summary.csv',index=False)
    print(pd.read_csv(RESULTS/'paired_effects.csv').to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('\nPath maxima:\n',pd.read_csv(RESULTS/'path_summary.csv').to_string(index=False))
    p=pd.read_csv(RESULTS/'paired_effects.csv')
    valid=p[p.objective=='corr']
    print('CORR OOD CE favorable cells:',int((valid.ood_ce_diff_mean<0).sum()),'of',len(valid))

if __name__=='__main__':main()
