"""Evaluate untouched ID/OOD split once after source/model freeze."""
import json
from pathlib import Path
import pandas as pd
import torch
from core import Net, dataset, evaluate_model
from train import RESULTS, MODELS
R=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    assert (RESULTS/'freeze.json').exists(), 'Must freeze before holdout'
    assert not (RESULTS/'holdout.csv').exists(), 'Holdout has already been evaluated'
    manifest=json.loads((RESULTS/'freeze.json').read_text())
    import hashlib
    sha=lambda data:hashlib.sha256(data).hexdigest()
    for p,digest in manifest.items():
        if p in ('model_files','model_digest'):continue
        assert sha((R/p).read_bytes())==digest, f'frozen file changed: {p}'
    paths=sorted(MODELS.glob('*.pt'))
    assert len(paths)==manifest['model_files']
    assert sha(''.join(sha(p.read_bytes()) for p in paths).encode())==manifest['model_digest']
    models=pd.read_csv(RESULTS/'training.csv')
    assert len(models)==432
    rows=[]
    for task in ('moons24','prototypes4'):
        ds=dataset(task,True)
        reference_logits={}
        for r in models.to_dict(orient='records'):
            if r['task']!=task:continue
            m=Net(ds['classes'],r['arch']).double()
            m.load_state_dict(torch.load(MODELS/r['model'],map_location='cpu',weights_only=True))
            metrics={}
            key=(r['arch'],r['seed'],r['lr'],r['objective'])
            for split in ('id','ood'):
                v=evaluate_model(m,*ds[split])
                metrics.update({split+'_'+k:x for k,x in v.items()})
                with torch.no_grad():
                    logits=m(ds[split][0])[0]
                    if r['gauge']=='identity' and r['optimizer']=='sgd':
                        reference_logits[(key,split)]=logits.detach().clone()
                    else:
                        metrics[split+'_max_logit_diff_from_identity']=float((logits-reference_logits[(key,split)]).abs().max())
            if r['gauge']=='identity':
                metrics.update({'id_max_logit_diff_from_identity':0.,'ood_max_logit_diff_from_identity':0.})
            rows.append({k:r[k] for k in ('task','arch','seed','lr','objective','gauge','optimizer','model')}
                        | metrics)
        print(f'HOLDOUT EVALUATED task={task} rows so far={len(rows)}',flush=True)
    assert len(rows)==432
    pd.DataFrame(rows).to_csv(RESULTS/'holdout.csv',index=False)
    print('INDEPENDENT ID AND OOD HOLDOUT COMPLETE, 432 conditions',flush=True)

if __name__=='__main__':main()
