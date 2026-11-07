"""Read-only independent structural and hash audit for published experiment."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    freeze=json.loads((R/'freeze.json').read_text())
    for rel,digest in freeze.items():
        if rel in ('model_files','model_digest'):continue
        assert file_sha(ROOT/rel)==digest,rel
    models=sorted((ROOT/'models').glob('*.pt'))
    if models:
        assert len(models)==freeze['model_files']==432
        combined=''.join(file_sha(p) for p in models)
        assert hashlib.sha256(combined.encode()).hexdigest()==freeze['model_digest']
        print('checkpoints: 432/432 digests PASS')
    else:
        print('checkpoints not included: UNVERIFIED (regenerate by train.py)')
    train=pd.read_csv(R/'training.csv')
    start=pd.read_csv(R/'initial.csv')
    traj=pd.read_csv(R/'trajectories.csv')
    hold=pd.read_csv(R/'holdout.csv')
    contrast=pd.read_csv(R/'paired_effects.csv')
    assert (len(train),len(start),len(traj),len(hold),len(contrast))==(432,432,1728,432,16)
    keys=['task','arch','seed','lr','objective','gauge','optimizer']
    tkeys=set(map(tuple,train[keys].itertuples(index=False,name=None)))
    assert len(tkeys)==432
    assert tkeys==set(map(tuple,hold[keys].itertuples(index=False,name=None)))
    assert tkeys==set(map(tuple,start[keys].itertuples(index=False,name=None)))
    assert train.finite.all() and (train.steps==90).all()
    assert float(start.initial_logit_diff.max())<1e-9 and float(start.initial_corr_diff.max())<1e-9
    correct=train[(train.optimizer=='corrected')&(train.objective.isin(('ce','corr')))]
    assert len(correct)==96 and float(correct.max_path_diff.max())<1e-7
    correct_held=hold[(hold.optimizer=='corrected')&(hold.objective.isin(('ce','corr')))]
    assert float(correct_held.id_max_logit_diff_from_identity.max())<1e-7
    assert float(correct_held.ood_max_logit_diff_from_identity.max())<1e-7
    negative=train[(train.optimizer=='corrected')&(train.objective=='cov')]
    assert len(negative)==48 and float(negative.max_path_diff.max())>1e-7
    assert not train.select_dtypes(include='number').isna().any().any()
    assert not hold.select_dtypes(include='number').isna().any().any()
    # The paired OOD effect is independently recalculated from the raw holdout table.
    for row in contrast.itertuples():
        grp=hold[(hold.task==row.task)&(hold.arch==row.arch)&(hold.lr==row.lr)]
        ce=grp[(grp.objective=='ce')&(grp.gauge=='identity')&(grp.optimizer=='sgd')].set_index('seed').sort_index()
        co=grp[(grp.objective==row.objective)&(grp.gauge=='identity')&(grp.optimizer=='sgd')].set_index('seed').sort_index()
        actual=(co.ood_ce-ce.ood_ce).mean()
        assert abs(actual-row.ood_ce_diff_mean)<1e-12
        actual_id=(co.id_ce-ce.id_ce).mean()
        assert abs(actual_id-row.id_ce_diff_mean)<1e-12
    print('frozen files, key uniqueness, 432 runs, 96 corrected paths, 16 paired comparisons: PASS')
    print('all corrected CE/corr ID and OOD predictions coincide: PASS')
    print('OOD correlation-improved cells:', int((contrast[contrast.objective=='corr'].ood_ce_diff_mean<0).sum()),'/8')

if __name__=='__main__':run()
