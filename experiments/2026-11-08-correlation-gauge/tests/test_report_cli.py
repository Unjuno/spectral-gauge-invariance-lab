"""Regression: the published report CLI must work without a pre-created figures/ folder."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results' / 'full_data.zip'


@pytest.mark.skipif(not ARCHIVE.is_file(), reason='requires published full_data.zip audit archive')
def test_report_recreates_missing_figures_directory(tmp_path):
    shutil.copy2(ROOT / 'report.py', tmp_path / 'report.py')
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in ('results/FREEZE_BEFORE_HOLDOUT.json',
                     'results/training.csv',
                     'results/heldout_test.csv',
                     'results/initial_invariance.csv'):
            dest = tmp_path / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
    assert not (tmp_path / 'figures').exists()
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = str(tmp_path / 'mplconfig')
    run = subprocess.run(
        [sys.executable, 'report.py'], cwd=tmp_path, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert run.returncode == 0, run.stdout[-1000:] + run.stderr[-1700:]
    assert (tmp_path / 'figures' / 'regularizer_holdout_comparison.png').is_file()
    assert (tmp_path / 'figures' / 'gauge_effects.png').is_file()
