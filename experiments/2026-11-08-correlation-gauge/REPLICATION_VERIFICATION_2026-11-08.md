# Independent reproduction check and report regression fix (2026-11-08)

## Why this note exists

The published CPU experiment was independently rerun from the source-only package in a newly created directory. This is a same-environment deterministic reproduction, **not** external cross-hardware replication or evidence of better language-model training.

## Fresh rerun

- Environment: Linux container, Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0; float64, one PyTorch thread; full batch size 800; 10 warmup updates then 120 updates per condition.
- Recreated 432 trained models. The original and new aggregated model SHA-256 digests matched exactly: `13356e6e4316ce56db73a7c7b2eec3a75685e0c63297d88f599dd78043d7212b`.
- Frozen SHA-256 manifest entries (protocol, implementation, pre-holdout training tables and original model digest) matched.
- Independent CSV-by-CSV comparison (numeric values read as float64, other columns by exact text equality): `training.csv` **432** rows, `initial_invariance.csv` **432**, `trajectories.csv` **2,592**, `heldout_test.csv` **432**, `paired_comparisons.csv` **168**, `summary.csv` **36**. Maximum observed numeric difference was **0** in all six tables.
- The independent holdout evaluation was run only after `freeze.py`, consistent with the published protocol.

## Fix discovered by fresh execution

An original *source-only* directory has no `figures/` folder. `report.py` previously assumed that folder already existed, raising `FileNotFoundError` during `savefig` after computing all statistics.

A minimal change now creates `figures/` before saving the images. A new integration regression test (`tests/test_report_cli.py`) runs the report with results extracted into a clean temporary directory that has **no** pre-existing figures folder. The test failed before the fix and passed after it. The updated local suite reported **6 passed** when the compact audit archive was available; from source-only archives, the new integration test skips until the published `results/full_data.zip` is present.

`report.py` is not among the original `FREEZE_BEFORE_HOLDOUT.json` source files, so this post-holdout report-only fix does **not** change the original frozen training/evaluation inputs. The generated result values are unchanged.

## Limitations

These are same-stack deterministic results, conditioned on two synthetic tasks, one model architecture and fixed data splits. They do not establish performance gains across distributions, general convergence, or cross-platform bitwise reproducibility. Published checkpoint binaries remain excluded and can be regenerated using [REPRODUCE.md](REPRODUCE.md).
