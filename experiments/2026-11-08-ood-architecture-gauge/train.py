"""Train only; never creates an OOD sample or accesses ID/OOD holdout tensors."""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from core import Net,update,dataset,gauge,stats,evaluate_model,obj

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
MODELS=ROOT/'models'
TASKS=('moons24','prototypes4')
ARCHS=('plain','residual')
SEEDS=tuple(range(8200,8206))
LRS=(.08,.25)
OBJS=('ce','corr','cov')
VARIANTS=(('identity','sgd'),('heterogeneous','sgd'),('heterogeneous','corrected'))
STEPS=90
WARMUP=8

def ck_name(t,a,s,lr,k,g,opt):
    return f'{t}_{a}_{s}_{str(lr).replace(".","p")}_{k}_{g}_{opt}.pt'

def pprobe(m,x):
    with torch.no_grad():return m(x)[0].detach().clone()

def main():
    start=time.monotonic()
    assert not (RESULTS/'holdout.csv').exists(), 'Remove previous results and rerun in fresh directory'
    RESULTS.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records=[]; paths=[]; initials=[]
    for task in TASKS:
        ds=dataset(task,False);tx,ty=ds['train'];vx,vy=ds['val'];probe=vx[:48]
        for architecture in ARCHS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                warm=Net(ds['classes'],architecture).double()
                for step in range(WARMUP):
                    assert update(warm,tx,ty,.03,'ce','sgd'), (task,architecture,seed,step)
                warm_logit=pprobe(warm,probe)
                with torch.no_grad():
                    ws=stats(warm(tx)[1])
                    initial_corr=ws['corr']
                    initial_obj={kind:float(obj(*warm(tx),ty,kind)) for kind in OBJS}
                for lr in LRS:
                    for kind in OBJS:
                        r=copy.deepcopy(warm)
                        ref_path=[warm_logit]
                        for step in range(STEPS):
                            assert update(r,tx,ty,lr,kind,'sgd'), (task,architecture,seed,lr,kind,step)
                            ref_path.append(pprobe(r,probe))
                        for g,opt in VARIANTS:
                            m=copy.deepcopy(r) if g=='identity' else gauge(warm,g)
                            init_model=warm if g=='identity' else m
                            with torch.no_grad():
                                plog,ph=init_model(probe)
                                rh=init_model(tx)[1]
                                pscore=float(obj(*init_model(tx),ty,kind))
                                corrdiff=float((stats(rh)['corr']-initial_corr).abs().max())
                                pdiff=float((plog-warm_logit).abs().max())
                            key=dict(task=task,arch=architecture,seed=seed,lr=lr,objective=kind,gauge=g,optimizer=opt)
                            initials.append(dict(**key,initial_logit_diff=pdiff,initial_corr_diff=corrdiff,
                                            initial_objective_diff=pscore-initial_obj[kind]))
                            max_diff=0. if g=='identity' else pdiff
                            if g!='identity':
                                for step in range(STEPS):
                                    assert update(m,tx,ty,lr,kind,'corrected' if opt=='corrected' else 'sgd'),key
                                    diff=float((pprobe(m,probe)-ref_path[step+1]).abs().max())
                                    max_diff=max(max_diff,diff)
                                    if step in (0,14,44,89):
                                        paths.append(dict(**key,step=step+1,path_diff=diff))
                            else:
                                for step in (0,14,44,89):
                                    paths.append(dict(**key,step=step+1,path_diff=0.))
                            tr=evaluate_model(m,tx,ty)
                            val=evaluate_model(m,vx,vy)
                            name=ck_name(task,architecture,seed,lr,kind,g,opt)
                            torch.save(m.state_dict(),MODELS/name)
                            records.append(dict(**key,model=name,steps=STEPS,finite=True,max_path_diff=max_diff,
                                                **{'train_'+k:v for k,v in tr.items()},
                                                **{'val_'+k:v for k,v in val.items()}))
                print(f'TRAINING {task} {architecture} seed={seed} done; {len(records)} total; elapsed={time.monotonic()-start:.1f}s',flush=True)
                pd.DataFrame(records).to_csv(RESULTS/'training.csv',index=False)
                pd.DataFrame(initials).to_csv(RESULTS/'initial.csv',index=False)
                pd.DataFrame(paths).to_csv(RESULTS/'trajectories.csv',index=False)
    assert len(records)==432 and len(initials)==432 and len(paths)==1728
    if max(row['initial_logit_diff'] for row in initials)>1e-9:
        raise AssertionError('Initial prediction invariance broken')
    if max(row['initial_corr_diff'] for row in initials)>1e-9:
        raise AssertionError('Initial correlation invariance broken')
    if max(row['max_path_diff'] for row in records if row['objective'] in ('ce','corr') and row['optimizer']=='corrected')>1e-7:
        raise AssertionError('Coordinate-corrected path invariance broken')
    env={'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version(),
         'platform':platform.platform(),'num_threads':torch.get_num_threads(),'precision':'float64',
         'device':'CPU','batch_size':640,'lr':LRS,'seeds':SEEDS,'warmup':WARMUP,'steps':STEPS,
         'training_seconds':round(time.monotonic()-start,3)}
    (RESULTS/'environment.json').write_text(json.dumps(env,indent=2))
    print('TRAINING DONE: all 432 conditions, no ID/OOD holdout read.',flush=True)

if __name__=='__main__':main()
