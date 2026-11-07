from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
r=pd.read_csv(R/'results/paired_effects.csv')
r=r[r.objective=='corr'].sort_values(['task','arch','lr']).reset_index(drop=True)
labels=[f'{row.task}, {row.arch}, lr={row.lr:.2f}' for row in r.itertuples()]
fig,ax=plt.subplots(figsize=(10.6,5.4))
for i,v in r.iterrows():
  mean=v.ood_ce_diff_mean
  ax.errorbar(mean,i,xerr=[[mean-v.ci95_low],[v.ci95_high-mean]],fmt='o',capsize=4)
ax.axvline(0,color='gray',linestyle='--',linewidth=1)
ax.set_yticks(list(range(len(labels))),labels)
ax.invert_yaxis()
ax.set_xlabel('OOD cross-entropy difference: correlation regularization − CE (lower is better)')
ax.set_title('Gauge-consistent correlation control: no uniform OOD improvement')
ax.grid(axis='x',alpha=.20)
fig.text(.15,.015,'2 generated tasks × 2 architectures × 2 learning rates. Six paired seeds per cell; percentile 95% seed-bootstrap intervals, unadjusted.',fontsize=9)
fig.subplots_adjust(left=.34,right=.97,bottom=.17,top=.89)
(R/'figures').mkdir(parents=True,exist_ok=True)
fig.savefig(R/'figures/ood_ce_differences.png',dpi=175)
fig.savefig(R/'figures/ood_ce_differences.svg')
plt.close(fig)
print('figure saved',R/'figures/ood_ce_differences.png')
