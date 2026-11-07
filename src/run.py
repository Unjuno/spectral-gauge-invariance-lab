"""Run training without returning or inspecting any test-set examples."""
import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment import (Net, apply_gauge, dataset, train_step, metrics, spectral_metrics,
                        GAUGES, OPTIMIZERS, LEARNING_RATES, SEEDS, STEPS, WARMUP_STEPS)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
(ROOT/'models').mkdir(exist_ok=True)

@torch.no_grad()
def probe(net,x):
    return net(x)[0].detach().clone()


def run_training():
    started=time.monotonic()
    env=dict(platform=platform.platform(),processor=platform.processor(),
             torch=torch.__version__,numpy=np.__version__,num_threads=torch.get_num_threads(),
             cuda=torch.cuda.is_available(),precision='float64',train_batch=800,
             n_seeds=len(SEEDS),steps=STEPS,warmup_steps=WARMUP_STEPS)
    (OUT/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    rows=[]; pathrows=[]; geometry=[];probes=[]
    for dataset_name in ('classification6','moons'):
        d=dataset(dataset_name,include_test=False)
        tx,ty=d['train'];vx,vy=d['validation']
        px=vx[:64]
        for seed in SEEDS:
            torch.manual_seed(seed)
            original=Net(d['classes']).double()
            for t in range(WARMUP_STEPS):
                train_step(original,tx,ty,.03,False,None)
            warm_reference=probe(original,px)
            warm_geometry=spectral_metrics(original(tx)[1].detach())
            for lr in LEARNING_RATES:
                # Baseline computed first for valid paired path comparison.
                ref = copy.deepcopy(original)
                ref_probes=[probe(ref,px)]
                for step in range(STEPS):
                    train_step(ref,tx,ty,lr,False,None)
                    ref_probes.append(probe(ref,px))
                for mode in OPTIMIZERS:
                    for gauge in GAUGES:
                        net=apply_gauge(original,gauge)
                        with torch.no_grad():
                            current_pred=probe(net,px)
                            pred_diff=float((current_pred-warm_reference).abs().max())
                            initial_geom=spectral_metrics(net(tx)[1])
                        assert pred_diff<1e-10, (dataset_name,seed,lr,mode,gauge,pred_diff)
                        prefix=dict(dataset=dataset_name,seed=seed,lr=lr,optimizer=mode,gauge=gauge)
                        geometry.append(dict(**prefix,prediction_diff=pred_diff,**initial_geom,
                                             ref_trace=warm_geometry['trace'],
                                             ref_rank=warm_geometry['effective_rank']))
                        max_path_diff=pred_diff
                        reason=''; train_losses=[]
                        for step in range(STEPS):
                            z=train_step(net,tx,ty,lr,compensated=(mode=='gauge_compensated_sgd'),scales=gauge)
                            if not np.isfinite(z):
                                reason='nonfinite gradient/loss';break
                            train_losses.append(z)
                            with torch.no_grad():
                                difference=float((probe(net,px)-ref_probes[step+1]).abs().max())
                                if np.isfinite(difference):max_path_diff=max(max_path_diff,difference)
                            if step in (0,9,29,59,99):
                                ss=metrics(net,tx,ty)
                                probes.append(dict(**prefix,step=step+1,train_ce=ss['ce'],
                                                   trace=ss['trace'],effective_rank=ss['effective_rank'],
                                                   max_logit_difference_to_identity=difference))
                            if not np.isfinite(difference) or difference>1e100:
                                reason='nonfinite/huge prediction';break
                        tr=metrics(net,tx,ty)
                        vl=metrics(net,vx,vy)
                        row=dict(**prefix,steps_done=len(train_losses),failed=bool(reason),
                                 failure_reason=reason,max_probe_logit_difference=max_path_diff,
                                 train_ce=tr['ce'],train_ba=tr['balanced_accuracy'],
                                 val_ce=vl['ce'],val_ba=vl['balanced_accuracy'],
                                 val_acc=vl['accuracy'],final_trace=tr['trace'],
                                 final_effective_rank=tr['effective_rank'],
                                 final_top_fraction=tr['top_fraction'])
                        rows.append(row)
                        filename=f'{dataset_name}_{seed}_lr{str(lr).replace(".","p")}_{mode}_{gauge}.pt'
                        torch.save(net.state_dict(),ROOT/'models'/filename)
                        pathrows.append(dict(**prefix,filename=filename))
            pd.DataFrame(rows).to_csv(OUT/'training.csv',index=False)
            pd.DataFrame(probes).to_csv(OUT/'training_trajectories.csv',index=False)
            pd.DataFrame(geometry).to_csv(OUT/'initial_geometry.csv',index=False)
            pd.DataFrame(pathrows).to_csv(OUT/'model_manifest.csv',index=False)
            print('training',dataset_name,seed,'runs',len(rows),'elapsed',round(time.monotonic()-started,1),flush=True)
    env['wall_seconds']=time.monotonic()-started
    (OUT/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    print('COMPLETED',len(rows),'runs',round(env['wall_seconds'],1),'seconds',flush=True)

if __name__=='__main__':
    run_training()
