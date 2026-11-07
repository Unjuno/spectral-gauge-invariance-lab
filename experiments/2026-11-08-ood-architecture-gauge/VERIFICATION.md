# Same-stack independent reproduction, 2026-11-08

The evaluation was first run on 432 models trained under the frozen protocol. The experiment was then reproduced **from source files only** in a separate directory without reference checkpoints or old held-out data.

## Verification findings

- Unit tests: 5 passed.
- 432 model files regenerated in the separate run; aggregate SHA-256 matches the original.
- `results/training.csv`: 432 rows, identical bytes.
- `results/initial.csv`: 432 rows, identical bytes.
- `results/trajectories.csv`: 1,728 rows, identical bytes.
- `results/holdout.csv`: 432 rows, identical bytes.
- `results/paired_effects.csv`: 16 rows, identical bytes.
- `results/path_summary.csv`: 9 rows, identical bytes.
- `results/freeze.json`: identical bytes.
- Both ID and OOD predictions at final checkpoint match the identity-SGD model for all 96 CE/correlation-corrected runs (recorded max logit deviation 0.0 in FP64).
- Negative control: covariance penalty has path divergence (max ~6.317 on validation probe); final OOD logit divergence max ~9.347.
- Correlation objective improved mean OOD CE in 2/8 prespecified task/architecture/LR cells, otherwise worsened.

## Pre-holdout mechanical correction

The initial protocol's factorial count was arithmetically wrong (`216` instead of `432`, with `36` corrected cases/objective instead of `48`). The first run finished all 432 trainings but failed the final count assertion. That failed-run log is retained in the local audit archive, not in GitHub. All code and protocol count assertions were corrected **before evaluating held-out data**, the entire training rerun unchanged, and a SHA-256 freeze made before holdout evaluation. The results reported here come only from that rerun.

## Boundary of the verification

This is a same-CPU-stack replication; it is neither independent external replication nor cross-hardware bitwise reproducibility. The OOD test is a **synthetic label-preserving shift** defined in source code. No real-world OOD generalization, convergence theorem, or universal performance improvement is claimed. The public Git tree omits model binaries; the frozen model digest can be verified after retraining.
