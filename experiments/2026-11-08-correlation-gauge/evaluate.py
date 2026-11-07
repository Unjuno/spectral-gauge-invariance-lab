"""Evaluate untouched holdout exactly once after freezing training results."""
from pathlib import Path
import json
import pandas as pd
import torch
from core import Net, synthetic_data, model_metrics

ROOT=Path(__file__).resolve().parent
FREEZE=ROOT/'results'/'FREEZE_BEFORE_HOLDOUT.json'
assert FREEZE.exists(),'Must run freeze.py before holdout evaluation.'
assert not (ROOT/'results'/'heldout_test.csv').exists(),'Holdout already evaluated; refuse second evaluation.'
manifest=json.loads(FREEZE.read_text())
assert manifest['n_models']==432
tr=pd.read_csv(ROOT/'results'/'training.csv')
assert len(tr)==432
rows=[]
for task in ('classification6','moons22'):
    d=synthetic_data(task,include_test=True)
    x,y=d['test']
    for r in tr[tr.dataset==task].to_dict('records'):
        model=Net(d['classes']).double()
        model.load_state_dict(torch.load(ROOT/'models'/r['filename'],weights_only=True,map_location='cpu'))
        m=model_metrics(model,x,y,prefix='test_')
        rows.append({k:r[k] for k in ('dataset','seed','lr','objective','gauge','optimizer')}|m)
    print('holdout evaluation finished',task, 'rows',len(rows),flush=True)
result=pd.DataFrame(rows)
assert len(result)==432
result.to_csv(ROOT/'results'/'heldout_test.csv',index=False)
print('HOLDOUT ONCE: COMPLETE 432 models')
