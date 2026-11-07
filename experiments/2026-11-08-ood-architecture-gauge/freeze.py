"""Freeze training and implementation to hashes before touching ID/OOD holdouts."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent
FILES=['PROTOCOL.md','core.py','train.py','evaluate.py','analysis.py','tests/test_invariance.py',
       'results/training.csv','results/trajectories.csv','results/initial.csv']
assert not (R/'results/holdout.csv').exists()
paths=sorted((R/'models').glob('*.pt'))
assert len(paths)==432,(len(paths))
sha=lambda data:hashlib.sha256(data).hexdigest()
manifest={str(p):sha((R/p).read_bytes()) for p in FILES}
manifest['model_files']=len(paths)
manifest['model_digest']=sha(''.join(sha(p.read_bytes()) for p in paths).encode())
(R/'results/freeze.json').write_text(json.dumps(manifest,indent=2))
print('FROZEN', len(paths), 'models',manifest['model_digest'])
