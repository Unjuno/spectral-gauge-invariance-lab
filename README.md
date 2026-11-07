# Spectral gauge invariance and optimizer sensitivity

**Status: reproducible small-scale empirical and algebraic result, not a neural-network convergence theorem.**

The practical question is whether manipulating neural-network feature spectra improves learning *because of the representation*, or because an optimizer responds differently to an equivalent parameterization. Our controlled experiment finds the latter mechanism suffices to explain a large class of observed gains and failures.

## Reproducible findings

1. **The same predictor can have different feature spectra.** Positive diagonal scaling of a ReLU penultimate layer, with an inverse scaling in the output layer, gives unchanged predictions. Uniform 1/4 scaling reduces covariance trace by 16x; uniform 4x scaling increases it 16x; heterogeneous scaling changes effective rank to about 0.50–0.56x the original, while predictions are unchanged. This is an exact structural symmetry; numerical differences at initialization were zero at double precision for the chosen power-of-two scales.
2. **Uncompensated SGD reacts to parameterization.** In the held-out 6-class task, the mean test CE after 100 updates at learning rate 0.08 was **1.365 (identity) vs 0.880 (1/4 scaling)**, mean paired difference −0.485, approximate seed-bootstrap 95% CI [−0.515, −0.457]. Yet in the independent nonlinear moons task at LR 0.25, the same scaling **worsened** test CE by +0.043, CI [+0.038,+0.047]. Scaling is therefore not a generally beneficial regularizer.
3. **Coordinate-corrected SGD exactly recovers the reference training path.** For all 2 datasets × 12 random seeds × 2 learning rates × 4 gauges, max prediction-path divergence was recorded as 0 in float64 for the gauge-corrected conditions. The identity conditions are included; all 384 training runs were finite. This is a numerical check of the algebra below.

**Useful implication:** feature-spectrum statistics without a function-based diagnostic can be misleading. Any apparent gain from rescaling features should be compared with an equivalent optimizer update before attributing the gain to representation compression or expansion. The rank/variance behavior of a representation is coordinate-dependent.

## Method and auditability

- Two independent synthetic benchmark *splits*, each with n=3600, 24-dimensional inputs: `classification6` (`make_classification`, 6 classes) and `moons` (`make_moons`, 2 classes + 22 nuisance features). Training 800, validation 400, independent holdout 2400; stratified nonoverlapping indices and train-only standardization. The holdout data were not accessed during training, and no hyperparameters were selected with it.
- Model 24→64→24→C with ReLU activations; transformed penultimate 24 channels. Full-batch plain SGD, 15 warmup steps at LR 0.03, then 100 fixed steps at LR 0.08 or 0.25. 12 matched seeds (6100–6111). Four transformations: identity; uniform 1/4; uniform 4; cyclic diagonal (1/4,1,4). Paired comparisons on the same seed, 5,000 seed-bootstrap repetitions. Cross-entropy and balanced accuracy on the untouched test split.
- Environment: PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, float64, deterministic CPU, one PyTorch thread, full batch 800, no GPU. CPU frequency was not fixed; no throughput benchmark is claimed. `results/environment.json` records platform and measured runtime.
- Primary definitions and conditions frozen in `PROTOCOL.md` before holdout evaluation. Results in `results/heldout_test.csv`, all training outcomes in `results/training.csv`, raw paired effects with intervals in `results/paired_contrasts.csv`, per-step traces in `results/training_trajectories.csv`, and reproducible geometry summaries.

![Paired held-out cross-entropy differences](figures/sgd_test_loss_contrasts.png)

## Mathematical statement and exact scope

Let the model have intermediate preactivation `z = W₂a + b₂`, ReLU features `h = ReLU(z)`, and logits `f = Vh + c`. For a positive diagonal invertible matrix `D`, define `W₂' = DW₂`, `b₂' = Db₂`, and `V' = VD⁻¹`. Since `ReLU(Dz) = D ReLU(z)`, logits are identical for every input. Feature covariance transforms as `Σ' = DΣD`; spectral scale, normalized eigenvalues and effective rank need not be invariant.

With ordinary SGD `θ ← θ − η ∇θL`, the Euclidean gradient depends on coordinates. Under this reparameterization, `∇W₂' L = D⁻¹ ∇W₂L`, and `∇V' L = (∇VL)D`. The compensating update is therefore **multiply `∇W₂'` and `∇b₂'` by `D²`, and multiply the columns of `∇V'` by `D⁻²`**, then apply the same learning rate; other layers use unchanged SGD. This gives precisely the same *function-space training path* as untransformed ordinary SGD in exact arithmetic, without momentum or weight decay. The experiment numerically confirms it for these power-of-two transforms.

**Variable table (SI units are 1 = dimensionless):**

| Symbol | Japanese meaning / definition | SI | Type / domain |
|---|---|---|---|
| `a` | 前段の活性値 | 1 | real input vector |
| `z,h` | 変換対象層の活性化前・ReLU後特徴 | 1 | real 24-vectors |
| `W₂,b₂` | 変換対象層の重み・バイアス | 1 | real 24×64 matrix; real 24-vector |
| `V,c` | 出力層の重み・バイアス | 1 | real C×24 matrix; real C-vector |
| `D` | 正の対角変換 | 1 | invertible diagonal 24×24 matrix, positive diagonals |
| `Σ` | 特徴の共分散 | 1 | symmetric PSD 24×24 matrix |
| `η` | 学習率 | 1 | positive scalar |
| `L` | 交差エントロピー損失 | 1 | nonnegative real scalar |
| `C` | クラス数 | 1 | positive integer, 2 or 6 |

**Dimensional check:** `W₂a+b₂`, `Vh+c`, and the transformed versions are all dimensionless vectors of matching sizes. Both covariance matrices are dimensionless squared, hence dimensionless. Gradient multipliers `D²` and `D⁻²` are dimensionless.

## Run from scratch

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/run.py
# Train/validation results are now fixed. Evaluate the untouched holdout once:
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/evaluate.py
python src/report.py
```

Using the script regenerates datasets from fixed random seeds. Models are generated in `models/` and deliberately not tracked in Git. Random seeds and calculations are fully specified in `src/experiment.py`.

## Limitations and falsifiability

- This is **not** evidence of asymptotic optimization convergence, better language models, or broad generalization. Data are synthetic, all runs use a fixed split and one architecture, and the holdout is not an out-of-distribution benchmark.
- The 12 seeds quantify initialization sensitivity *conditional on these fixed datasets*, not uncertainty across all datasets. The approximate percentile bootstrap intervals are unadjusted for multiple comparisons. Test data are touched exactly once by the evaluation script; no after-the-fact tuning is performed.
- The compensated update presumes a fixed positive diagonal gauge, ReLU, plain full-batch SGD, no momentum and no weight decay. It does not directly apply to arbitrary nonlinear representation interventions or AdamW without modification.
- All metrics are reported, including negative results and possible failures (none in this run). Function-preserving reparameterization is not the same as erasing nuisance information.

### Three useful connections

- **Differential geometry / parameter manifolds:** equivalent coordinates describe the same predictive function but have different Euclidean gradient directions.
- **Numerical optimization:** diagonal preconditioning changes effective step sizes, so apparent compression gains can be optimization artifacts.
- **High-dimensional statistics / representation learning:** covariance eigenspectra and effective ranks are not invariant to arbitrary invertible changes of feature coordinates; predictive sufficiency requires separate tests.

### Verification

`python -m pytest -q` must pass. `src/report.py` checks 384 paired rows, zero training failures, prediction-invariance tolerance and compensated path tolerance. The outputs are inspectable in the versioned CSVs. Main finding is a falsification of treating feature-spectrum shrinkage alone as a sufficient explanation for performance gains.

**Next falsifiable question:** Does any residual benefit remain after using an optimizer that is coordinate-covariant with the entire representation transform, instead of merely the diagonal transformation studied here?