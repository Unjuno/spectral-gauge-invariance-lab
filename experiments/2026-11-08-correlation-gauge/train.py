"""Training-only script. NEVER touches heldout test inputs; run evaluate.py separately."""
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import Net, apply_gauge, covariance_stats, model_metrics, objective, step, synthetic_data, LAMBDA, VAR_EPS

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'
MOD=ROOT/'models'
RES.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
TASKS=['classification6','moons22']
SEEDS=list(range(7200,7212))
LRS=[0.08,0.25]
MODES=['ce','corr','cov']
VARIANTS=[('identity','sgd'),('heterogeneous','sgd'),('heterogeneous','corrected')]
STEPS=120


def logit_probe(model,x):
    with torch.no_grad():return model(x)[0].detach().clone()


def train():
    began=time.monotonic()
    rows=[];traj=[];init=[]
    for task in TASKS:
        ds=synthetic_data(task,include_test=False)
        tx,ty=ds['train']; vx,vy=ds['val']
        probe=vx[:48]
        for seed in SEEDS:
            torch.manual_seed(seed)
            warm=Net(ds['classes']).double()
            for i in range(10):
                s=step(warm,tx,ty,0.03,'ce',False,'identity')
                assert s['finite']
            warm_probe=logit_probe(warm,probe)
            with torch.no_grad():
                warm_feature=warm(tx)[1]
                warm_corr=covariance_stats(warm_feature,include_ranks=False)['corr']
            for lr in LRS:
                for mode in MODES:
                    ref=copy.deepcopy(warm)
                    probes=[logit_probe(ref,probe)]
                    for j in range(STEPS):
                        _=step(ref,tx,ty,lr,mode,False,'identity')
                        probes.append(logit_probe(ref,probe))
                    for gauge,optimizer in VARIANTS:
                        m=apply_gauge(warm,gauge)
                        prefix=dict(dataset=task,seed=seed,lr=lr,objective=mode,gauge=gauge,optimizer=optimizer)
                        with torch.no_grad():
                            first_logits,first_h=m(probe)
                            h_train=m(tx)[1]
                            first_corr=covariance_stats(h_train,include_ranks=False)['corr']
                            pred_diff=float((first_logits-warm_probe).abs().max())
                            corr_diff=float((first_corr-warm_corr).abs().max())
                            loss_r=float(objective(m(tx)[0],h_train,ty,mode)[1])
                            loss_r_ref=float(objective(warm(tx)[0],warm_feature,ty,mode)[1])
                        init.append(dict(**prefix,initial_prediction_diff=pred_diff,
                            initial_corr_diff=corr_diff,initial_regularizer=loss_r,
                            baseline_regularizer=loss_r_ref,
                            initial_regularizer_difference=loss_r-loss_r_ref))
                        max_path=pred_diff
                        finite=True
                        updates_done=0
                        for j in range(STEPS):
                            info=step(m,tx,ty,lr,mode,compensated=(optimizer=='corrected'),gauge=gauge)
                            if not info['finite']:
                                finite=False
                                break
                            updates_done+=1
                            pdiff=float((logit_probe(m,probe)-probes[j+1]).abs().max())
                            max_path=max(max_path,pdiff)
                            if j in (0,9,29,59,89,119):
                                tr=model_metrics(m,tx,ty,'train_')
                                val=model_metrics(m,vx,vy,'val_')
                                traj.append(dict(**prefix,step=j+1,prediction_path_diff=pdiff,**tr,**val))
                        tr=model_metrics(m,tx,ty,'train_')
                        val=model_metrics(m,vx,vy,'val_')
                        filename=f'{task}_s{seed}_lr{str(lr).replace(".","p")}_{mode}_{gauge}_{optimizer}.pt'
                        torch.save(m.state_dict(),MOD/filename)
                        rows.append(dict(**prefix,filename=filename,updates_done=updates_done,
                            finite=finite,max_path_diff=max_path,**tr,**val))
            print(f'training task={task} seed={seed} complete, count={len(rows)}, secs={time.monotonic()-began:.1f}',flush=True)
            pd.DataFrame(rows).to_csv(RES/'training.csv',index=False)
            pd.DataFrame(traj).to_csv(RES/'trajectories.csv',index=False)
            pd.DataFrame(init).to_csv(RES/'initial_invariance.csv',index=False)
    env={'torch':str(torch.__version__),'numpy':np.__version__, 'platform':platform.platform(),
        'precision':'float64','threads':torch.get_num_threads(),'GPU':False,'batch_size':800,
        'seeds':len(SEEDS),'steps':STEPS,'warmup':10,'learning_rates':LRS,
        'lambda':LAMBDA,'variance_guard':VAR_EPS, 'n_training_runs':len(rows),
        'train_seconds':time.monotonic()-began}
    (RES/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2))
    assert len(rows)==432
    print('TRAINING COMPLETE: 432 conditions; heldout has NOT been read',flush=True)

if __name__=='__main__':
    train()
