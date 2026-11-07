"""Recompute summary receipts and figures from committed CSV; no model selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)

def run():
    a=pd.read_csv(R/'training.csv')
    g=pd.read_csv(R/'initial_geometry.csv')
    t=pd.read_csv(R/'heldout_test.csv')
    c=pd.read_csv(R/'paired_contrasts.csv')
    # crosschecks against the actual paired outcomes
    assert len(a)==len(t)==len(g)==384 and not a.failed.any()
    assert g.prediction_diff.max()<=1e-10
    assert a[a.optimizer=='gauge_compensated_sgd'].max_probe_logit_difference.max()<=1e-8
    assert (g.groupby('gauge').trace.median()>0).all()
    # effect-size chart for all predefined gauges across datasets / learning rates
    sub=c[(c.metric=='test_ce')&(c.optimizer=='vanilla_sgd')&(c.gauge!='identity')]
    fig,ax=plt.subplots(figsize=(9.8,6.8))
    ticks=[];labels=[]
    for i,(v,row) in enumerate(sub.sort_values(['dataset','lr','gauge']).reset_index(drop=True).iterrows()):
        lo=row['mean_diff']-row['ci95_low'];hi=row['ci95_high']-row['mean_diff']
        ax.errorbar(row['mean_diff'],i,xerr=[[lo],[hi]],fmt='o',capsize=3)
        ticks.append(i);labels.append(f"{row.dataset}, lr={row.lr:.2f}, {row.gauge}")
    ax.set_yticks(ticks,labels,fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0,color='0.4',linewidth=1,linestyle='--')
    ax.set_xlabel('Difference in independent-test cross-entropy vs identity SGD (negative = better)')
    ax.set_title('Parameterization-sensitive SGD: benefits are task / learning-rate dependent')
    fig.tight_layout()
    fig.savefig(F/'sgd_test_loss_contrasts.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    # rank ratio by dataset
    gm=g[g.optimizer=='vanilla_sgd'].copy()
    gm['rank_ratio']=gm.effective_rank/gm.ref_rank
    agg=gm.groupby(['dataset','gauge'])[['trace','rank_ratio','prediction_diff']].agg(['median','min','max'])
    agg.to_csv(R/'geometry_summary.csv')
    return dict(count=len(a),corrected_max_diff=float(a[a.optimizer=='gauge_compensated_sgd'].max_probe_logit_difference.max()),
                failed=int(a.failed.sum()),initial_max_diff=float(g.prediction_diff.max()))
if __name__=='__main__':print(run())
