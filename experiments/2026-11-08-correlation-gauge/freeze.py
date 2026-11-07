"""Lock model manifest/code/protocol before touching holdout data."""
import hashlib, json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
files=['PROTOCOL.md','core.py','train.py','evaluate.py','freeze.py','results/training.csv','results/trajectories.csv','results/initial_invariance.csv']
tr=pd.read_csv(ROOT/'results/training.csv'); init=pd.read_csv(ROOT/'results/initial_invariance.csv')
assert len(tr)==432 and len(init)==432, (len(tr),len(init))
assert tr.finite.all() and (tr.updates_done==120).all()
model_paths=[ROOT/'models'/p for p in tr.filename]
assert len(model_paths)==432 and all(p.exists() for p in model_paths)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest={f:sha(ROOT/f) for f in files}
manifest['models_digest']=hashlib.sha256(''.join(sha(p) for p in sorted(model_paths)).encode()).hexdigest()
manifest['n_models']=len(model_paths)
(ROOT/'results'/'FREEZE_BEFORE_HOLDOUT.json').write_text(json.dumps(manifest,indent=2))
print('LOCKED',len(model_paths),'models',manifest['models_digest'])
