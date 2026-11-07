"""Statistics and static plots using locked training/holdout results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'
assert (RES/'FREEZE_BEFORE_HOLDOUT.json').exists()
tr=pd.read_csv(RES/'training.csv')
te=pd.read_csv(RES/'heldout_test.csv')
initial=pd.read_csv(RES/'initial_invariance.csv')
assert len(tr)==len(te)==len(initial)==432
joined=tr.merge(te,on=['dataset','seed','lr','objective','gauge','optimizer'],validate='one_to_one')
assert len(joined)==432
print('all finite',tr.finite.all(),'; train models',len(tr))
print('initial prediction diff max:',initial.initial_prediction_diff.max())
print('initial corr diff max:',initial.initial_corr_diff.max())
print('initial gauge-dependent covariance regularizer diff median:',initial[(initial.objective=='cov')&(initial.gauge=='heterogeneous')].initial_regularizer_difference.abs().median())

rows=[]
metrics=['test_ce','test_balanced_accuracy','train_corr_penalty','train_corr_rank','train_active_channels','train_cov_rank','val_ce']
for task in ('classification6','moons22'):
  for lr in (0.08,0.25):
    df=joined[(joined.dataset==task)&(joined.lr==lr)]
    def pivot_for(x,y):
       a=df[(df.objective==x[0])&(df.gauge==x[1])&(df.optimizer==x[2])].sort_values('seed')
       b=df[(df.objective==y[0])&(df.gauge==y[1])&(df.optimizer==y[2])].sort_values('seed')
       assert list(a.seed)==list(b.seed)==list(range(7200,7212))
       return a,b
    contrasts={
       'corr_vs_ce':(('corr','identity','sgd'),('ce','identity','sgd')),
       'cov_vs_ce':(('cov','identity','sgd'),('ce','identity','sgd')),
       'corr_gauge_corrected_vs_identity':(('corr','heterogeneous','corrected'),('corr','identity','sgd')),
       'corr_gauge_plain_vs_identity':(('corr','heterogeneous','sgd'),('corr','identity','sgd')),
       'cov_gauge_corrected_vs_identity':(('cov','heterogeneous','corrected'),('cov','identity','sgd')),
       'ce_gauge_corrected_vs_identity':(('ce','heterogeneous','corrected'),('ce','identity','sgd')),
    }
    for c,(v1,v0) in contrasts.items():
       a,b=pivot_for(v1,v0)
       for metric in metrics:
           dif=a[metric].to_numpy()-b[metric].to_numpy()
           rng=np.random.default_rng(9102026+list(contrasts).index(c)*11+metrics.index(metric))
           boot=dif[rng.integers(0,12,size=(5000,12))].mean(axis=1)
           lo,hi=np.quantile(boot,[.025,.975])
           rows.append(dict(dataset=task,lr=lr,contrast=c,metric=metric,mean_difference=float(dif.mean()),
              median_difference=float(np.median(dif)),ci_low=float(lo),ci_high=float(hi),min_difference=float(dif.min()),max_difference=float(dif.max()),n_seeds=12))
compare=pd.DataFrame(rows)
compare.to_csv(RES/'paired_comparisons.csv',index=False)

# A selection-neutral summary; no post-hoc model tuning.
summary=joined.groupby(['dataset','lr','objective','gauge','optimizer'],as_index=False).agg(
    test_ce=('test_ce','mean'),test_balanced_accuracy=('test_balanced_accuracy','mean'),
    corr_penalty=('train_corr_penalty','mean'),train_corr_rank=('train_corr_rank','mean'),
    train_active_channels=('train_active_channels','mean'),
    train_ce=('train_ce','mean'),val_ce=('val_ce','mean'),
    max_path_error=('max_path_diff','max'),failure=('finite',lambda v:int((~v).sum())))
summary.to_csv(RES/'summary.csv',index=False)
print('\nCANONICAL MAIN EFFECTS:')
print(summary[(summary.gauge=='identity')&(summary.optimizer=='sgd')][['dataset','lr','objective','test_ce','test_balanced_accuracy','corr_penalty','train_corr_rank','train_active_channels']].to_string(index=False))
print('\nMAIN CONTRASTS:')
print(compare[(compare.metric=='test_ce')&(compare.contrast.isin(('corr_vs_ce','cov_vs_ce','corr_gauge_corrected_vs_identity','corr_gauge_plain_vs_identity','cov_gauge_corrected_vs_identity')))][['dataset','lr','contrast','mean_difference','ci_low','ci_high']].to_string(index=False))
print('\nCOMPENSATED PATH max:')
for obj in ('ce','corr','cov'):
 sub=tr[(tr.objective==obj)&(tr.gauge=='heterogeneous')&(tr.optimizer=='corrected')]
 print(obj,'max',sub.max_path_diff.max(),'median',sub.max_path_diff.median())

f,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,(task,sub) in zip(axs,compare[(compare.metric=='test_ce')&(compare.contrast.isin(('corr_vs_ce','cov_vs_ce')))].groupby('dataset')):
  labels=[f'{obj} / LR {lr}' for lr,obj in zip(sub.lr,sub.contrast)]
  y=np.arange(len(sub))
  errlo=sub.mean_difference.values-sub.ci_low.values
  errhi=sub.ci_high.values-sub.mean_difference.values
  ax.errorbar(sub.mean_difference.values,y,xerr=np.vstack((errlo,errhi)),fmt='o',capsize=4)
  ax.axvline(0,color='grey',linestyle='--')
  ax.set_yticks(y,labels)
  ax.set_title(task)
  ax.set_xlabel('Test cross entropy difference vs CE (negative = improvement)')
  ax.grid(axis='x',alpha=.2)
f.suptitle('Gauge-aware feature regularization: held-out paired effects')
f.tight_layout()
f.savefig(ROOT/'figures'/'regularizer_holdout_comparison.png',dpi=170)

f,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,(task,sub) in zip(axs,compare[(compare.metric=='test_ce')&(compare.contrast.isin(('corr_gauge_plain_vs_identity','corr_gauge_corrected_vs_identity','cov_gauge_corrected_vs_identity')))].groupby('dataset')):
  labels=[f'{obj} / LR {lr}' for lr,obj in zip(sub.lr,sub.contrast)]
  y=np.arange(len(sub))
  ax.errorbar(sub.mean_difference.values,y,xerr=np.vstack((sub.mean_difference-sub.ci_low,sub.ci_high-sub.mean_difference)),fmt='o',capsize=4)
  ax.axvline(0,color='grey',linestyle='--')
  ax.set_yticks(y,labels);ax.set_title(task);ax.set_xlabel('Test CE change from identity-coordinate objective')
  ax.grid(axis='x',alpha=.2)
f.suptitle('Equivalent coordinates vs objective and optimizer effects')
f.tight_layout()
f.savefig(ROOT/'figures'/'gauge_effects.png',dpi=170)

assert not (~tr.finite).any()
assert initial.initial_prediction_diff.max()<1e-9
assert initial.initial_corr_diff.max()<1e-9
assert tr[(tr.objective=='corr')&(tr.gauge=='heterogeneous')&(tr.optimizer=='corrected')].max_path_diff.max()<1e-7
assert tr[(tr.objective=='ce')&(tr.gauge=='heterogeneous')&(tr.optimizer=='corrected')].max_path_diff.max()<1e-7
assert tr[(tr.objective=='cov')&(tr.gauge=='heterogeneous')&(tr.optimizer=='corrected')].max_path_diff.max()>1e-7
print('AUDIT: invariant objectives and non-invariant control verified')
