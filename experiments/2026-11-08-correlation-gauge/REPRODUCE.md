# Reproduce and audit

This is the **2026-11-08 correlation-gauge follow-up**, a frozen small-scale experiment. Source `.py` files, `PROTOCOL.md`, and the files listed in `FREEZE_BEFORE_HOLDOUT.json` are byte-for-byte copies of the pre-holdout frozen files; do not alter them when checking the original hashes.

## Requirements

- Python with pinned packages in `requirements.txt`
- CPU, PyTorch 2.10.0, `float64`, one PyTorch thread, no momentum or weight decay
- Full-batch 800 training examples; 10 warmup + 120 evaluation-study updates per condition
- Two synthetic tasks and 12 matched initialization seeds; no pretrained weights or private data

## Inspect without retraining

Unpack `results/full_data.zip` **into this experiment folder**, preserving the `results/` prefix; then run:

```bash
python verify_publication.py
python -m pytest -q tests/
python report.py
```

`verify_publication.py` checks the frozen hashes, the entire 432-row training/test structure, and independent paired main-effect calculations. If model checkpoints are absent, it reports the checkpoint digest as **unverified**, not as passed. Model weights are deliberately omitted from GitHub to avoid large binary history.

## Reproduce training and holdout in a fresh working directory

Copy the experiment **source files and tests only** to a fresh directory; do not bring previous results or models. Then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python train.py
python freeze.py
python evaluate.py
python report.py
```

`evaluate.py` refuses to run if `results/heldout_test.csv` already exists. The independent test split is generated deterministically only when `include_test=True`, which is called by `evaluate.py`. This safeguards the declared evaluation protocol; it does not replace external pre-registration or independent replication.

## What is not provided

- 432 checkpoint binaries (`models/*.pt`), which can be regenerated from the scripts.
- A claim that current source is interoperable across CUDA/TPU or a different torch/BLAS ABI.
- Distribution-shift/OOD evaluation, confidence calibration, convergence proof, or universal gain.

## Important integrity nuance

The frozen manifest contains a SHA-256 aggregate digest of all 432 checkpoint files. The publication includes the manifest but *not* the checkpoints; checking that digest requires running training or obtaining the original archival weights. The original `evaluate.py` checks that the freeze manifest exists but does not itself independently verify all hashes; `verify_publication.py` provides the missing post-hoc audit check without rewriting frozen evaluation code.
