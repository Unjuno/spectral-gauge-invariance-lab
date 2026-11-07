# Fresh-run reproduction checklist

Environment: Linux/CPU, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, pandas 2.2.3, `float64`, one torch thread, OMP/BLAS threads=1. Other backends may not reproduce bitwise.

From a **clean copy of the source directory without any `results/holdout.csv`**:

```bash
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q
python train.py       # only train/validation split; 432 runs
python freeze.py      # frozen source/training checksums, 432 model weights
python evaluate.py    # first and only ID/OOD holdout access
python analysis.py    # 16 paired effect rows; 8 correlation cells
python verify.py      # hash and paired-analysis audit
python plot.py
```

The holdout script refuses a repeat run when `results/holdout.csv` exists. This is a workflow guard, not an assurance against a researcher manually viewing a split in a different script.

- `results/freeze.json`: SHA-256 of pre-holdout protocol, implementation and training tables, plus aggregate SHA-256 over 432 checkpoints.
- `results/training.csv`, `initial.csv`, `trajectories.csv`, `holdout.csv`: complete per-condition outcomes, no omitted failures.
- `results/paired_effects.csv` and `path_summary.csv`: compact final comparisons.
- `figures/ood_ce_differences.png`: OOD paired CE differences.

The 432 `models/*.pt` files are excluded from GitHub to avoid binary bulk. Regeneration is required to validate checkpoint hashes; `verify.py` prints UNVERIFIED when models are absent. The original frozen digest is present in `results/freeze.json`.

The first completed training run stopped at a mistaken post-run assertion `216` rather than the correct grid size `432`. The count-only correction happened before independent test data were accessed, and the complete run was repeated. The corrected protocol includes a dated erratum; this is not a hyperparameter change.
