"""Independent read-only verifier for the published correlation-gauge experiment.

Requires only the Python standard library. Results may be extracted from full_data.zip
or read from results/. Frozen files are never rewritten by this script.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
ARCHIVE=RESULTS/'full_data.zip'


def result_bytes(path):
    p=ROOT/path
    if p.exists():
        return p.read_bytes()
    with zipfile.ZipFile(ARCHIVE) as z:
        return z.read(path)


def rows(path):
    return list(csv.DictReader(io.StringIO(result_bytes(path).decode('utf-8'))))


def check():
    manifest=json.loads(result_bytes('results/FREEZE_BEFORE_HOLDOUT.json'))
    for path,digest in manifest.items():
        if path in ('n_models','models_digest'):
            continue
        actual=hashlib.sha256(result_bytes(path)).hexdigest()
        assert actual==digest, f'frozen source or results modified: {path}'
    model_paths=sorted((ROOT/'models').glob('*.pt')) if (ROOT/'models').exists() else []
    if model_paths:
        assert len(model_paths)==manifest['n_models'], 'incomplete model checkpoint set'
        joined=''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths)
        assert hashlib.sha256(joined.encode()).hexdigest()==manifest['models_digest'], 'models hash mismatch'
        models='PASS'
    else:
        models='UNVERIFIED (checkpoints intentionally omitted)'
    keys=['dataset','seed','lr','objective','gauge','optimizer']
    training=rows('results/training.csv')
    test=rows('results/heldout_test.csv')
    initial=rows('results/initial_invariance.csv')
    assert len(training)==len(test)==len(initial)==432
    def make_keys(src): return [tuple(r[k] for k in keys) for r in src]
    assert len(set(make_keys(training)))==432
    assert len(set(make_keys(test)))==432
    assert len(set(make_keys(initial)))==432
    assert set(make_keys(training))==set(make_keys(test))==set(make_keys(initial))
    assert all(x['finite']=='True' and int(x['updates_done'])==120 for x in training)
    assert max(float(r['initial_prediction_diff']) for r in initial)<1e-9
    assert max(float(r['initial_corr_diff']) for r in initial)<1e-9
    paths=defaultdict(list)
    for r in training:
        if r['gauge']=='heterogeneous' and r['optimizer']=='corrected':
            paths[r['objective']].append(float(r['max_path_diff']))
    assert len(paths['ce'])==len(paths['corr'])==len(paths['cov'])==48
    assert max(paths['ce'])<1e-7 and max(paths['corr'])<1e-7
    assert max(paths['cov'])>1e-7
    lookup={tuple(r[k] for k in keys):r for r in test}
    for task in ('classification6','moons22'):
        for lr in ('0.08','0.25'):
            dif=[]
            for seed in range(7200,7212):
                common=(task,str(seed),lr)
                ce=lookup[(*common,'ce','identity','sgd')]
                co=lookup[(*common,'corr','identity','sgd')]
                dif.append(float(co['test_ce'])-float(ce['test_ce']))
            calculated=sum(dif)/len(dif)
            expected={('classification6','0.08'):0.04208386251546734,('classification6','0.25'):0.0014832185978453,('moons22','0.08'):0.006652511937681562,('moons22','0.25'):-0.006755860357096531}[(task,lr)]
            assert abs(calculated-expected)<1e-10, (task,lr,calculated,expected)
            print(f'{task} lr={lr}: correlation vs CE test loss delta={calculated:+.6f}')
    print('Frozen manifest: PASS; training/test uniqueness: PASS; 432 runs: PASS; paths: PASS')
    print('Model checkpoint digest:',models)

if __name__=='__main__':
    check()
