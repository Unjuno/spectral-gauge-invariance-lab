# Gauge-invariant feature-correlation control: a deeper test

**Status (2026-11-08): completed exploratory CPU experiment. This is not a generalization or convergence theorem.**

This experiment asks a more discriminating question than “does shrinking a spectrum help?”: **after removing the parameterization dependence of SGD, can a feature regularizer still produce a genuine change in training dynamics and generalization?** We distinguish *objective invariance* from *optimizer covariance* using paired tests.

The exact preregistered experiment is recorded in [PROTOCOL.md](PROTOCOL.md), written before the holdout was evaluated. The original published project is [Unjuno/spectral-gauge-invariance-lab](https://github.com/Unjuno/spectral-gauge-invariance-lab); this is a new standalone study, not an overwrite of its data.

## 1. Findings / 発見

1. **A feature-channel correlation matrix is unchanged by positive diagonal rescaling.** Predictions also remain unchanged if the outgoing weights are rescaled inversely. The covariance eigenvalues and covariance-based effective rank generally change. On all 432 paired configurations, the maximum initial prediction discrepancy and channel-correlation discrepancy were **0** in float64 for powers-of-two scales.
2. **The correlation objective, unlike normalized covariance off-diagonal energy, is invariant under the same feature gauge.** With *coordinate-corrected SGD*, the validation-probe predictions over 120 steps were identical to the canonical path in all 48 corrected correlation trajectories (2 datasets × 12 seeds × 2 learning rates), maximum recorded error **0**. The same held for CE only. With covariance regularization, corrected paths differed markedly (maximum per-path discrepancy 2.165, median 0.813 across 48 paths).
3. **Invariant does not mean useless or predictive improvement.** Correlation redundancy reduction substantially changes the correlation-spectrum geometry under canonical training: e.g., mean correlation effective rank increases from **6.03 to 10.77** for six-class LR=0.08, and from **4.84 to 15.73** for moons LR=0.25. This is a real change relative to the same coordinate system and same optimizer, not an artifact of choosing a feature scale. Yet out-of-sample prediction effects depend on task and learning rate, and are often negative.
4. **The optimizer and objective cause different kinds of gauge dependence.** Ordinary SGD produces different paths even with an invariant correlation objective; coordinate-corrected SGD removes this. In contrast, coordinate-corrected SGD does *not* remove the difference when the regularizer itself depends on channel scales. This is the specific experimental mechanism demonstrated here.

### Independent holdout results

All effects below are **correlation regularizer minus CE-only**, with both trained in the canonical coordinate system using plain SGD. Negative test cross entropy differences mean better performance. Values are paired averages over 12 seeds with approximate percentile 95% intervals from 5,000 seed bootstraps; not multiple-comparison adjusted.

| Task | Learning rate | CE-only test CE | Corr-reg test CE | Difference [95% CI] | Balanced accuracy change |
|---|---:|---:|---:|---:|---:|
| classification6 | 0.08 | 1.2314 | 1.2735 | **+0.0421 [+0.0315,+0.0523]** | −1.02 percentage points |
| classification6 | 0.25 | 0.8758 | 0.8773 | +0.0015 [−0.0016,+0.0045] | +0.07 points |
| moons22 | 0.08 | 0.3646 | 0.3712 | +0.0067 [+0.0039,+0.0097] | −0.10 points |
| moons22 | 0.25 | 0.4288 | 0.4221 | **−0.0068 [−0.0086,−0.0050]** | −0.18 points |

**Decision:** Improvement was **NOT replicated consistently across tasks.** The favorable cross-entropy result in moons at LR=0.25 coexists with slightly worse balanced accuracy; it is not sufficient to call the regularizer generally useful for prediction. The observed change in geometry is more robust than the predictive change.

### Geometric changes under the canonical optimizer

| Task / LR | Mean corr effective rank CE → corr-reg | Mean off-diagonal squared correlation CE → corr-reg | Active feature channels CE → corr-reg |
|---|---|---|---|
| classification6 / 0.08 | 6.03 → **10.77** | 0.1344 → **0.0550** | 24 → 24 |
| classification6 / 0.25 | 6.37 → **11.68** | 0.1222 → **0.0460** | 24 → 23.92 |
| moons22 / 0.08 | 5.48 → **11.41** | 0.1506 → **0.0490** | 24 → 24 |
| moons22 / 0.25 | 4.84 → **15.73** | 0.1774 → **0.0233** | 24 → 24 |

A high correlation effective rank does **not** prove greater semantic information, and does not by itself establish out-of-distribution generalization.

## 2. Exact mechanism, assumptions, and derivation

**Variables and units (all neural activations and loss values are modeled as SI dimensionless quantities, `1`):**

| Symbol | Japanese meaning / definition | SI | Type, domain, assumptions |
|---|---|---|---|
| `n` | 標本数 | 1 | integer, `n >= 2`; training n=800 |
| `d` | 内部特徴次元数 | 1 | integer, d=24 |
| `h` | 1標本の特徴 | 1 | real d-vector, ReLU output |
| `H` | 各行に1標本の特徴を並べた行列 | 1 | real n×d matrix |
| `Z` | 列平均を引いた特徴行列 | 1 | real n×d matrix |
| `Σ` | 特徴の共分散 `ZᵀZ/n` | 1 | real d×d positive semidefinite matrix |
| `S` | 各チャネルの標準偏差の対角行列 | 1 | diagonal d×d, strictly positive in formal proof |
| `C` | 相関行列 `S⁻¹ Σ S⁻¹` | 1 | real d×d, symmetric PSD |
| `D` | 正の対角ゲージ変換 | 1 | invertible diagonal d×d, entries >0 |
| `R_corr` | 異なるチャネル間の相関の二乗平均 | 1 | scalar [0,1], see equation |
| `R_cov` | 平均分散で正規化した共分散の非対角二乗平均 | 1 | nonnegative scalar |
| `λ` | 正則化強度 | 1 | scalar, fixed 0.5 |
| `W,b,V` | 変換対象ReLU層と出力層の重み | 1 | W:24×64, b:24-vector, V:C_classes×24 |
| `η` | SGDの学習率 | 1 | positive scalar, 0.08 or 0.25 |
| `L_CE, L` | 交差エントロピーと全損失 | 1 | nonnegative real scalars |
| `r_C` | 相関行列の参加率による実効ランク | 1 | real scalar in [1,d] when C nonzero |

**(a) Gauge invariance of channel correlations.** Let centered features be `Z`. Under a fixed positive diagonal feature scaling `D`, rows become `h' = Dh`, hence `Z' = ZD`, and

\[
\Sigma' = \frac{(ZD)^T(ZD)}{n} = D\Sigma D.
\]

Because each standard deviation scales by the corresponding positive diagonal entry, `S' = DS`. Thus

\[
C'=(S')^{-1}\Sigma'(S')^{-1}=(DS)^{-1}(D\Sigma D)(DS)^{-1}=C.
\]

The normalized *correlation* eigenvalues and correlation effective rank are consequently unchanged by this transformation. The *covariance* eigenvalues need not be unchanged. The exact proof presumes every standard deviation is positive. Code assigns zero correlation to channels whose variance is below `1e−16`; recorded threshold-near channels were absent. Constant channels require separately checking the practical numerical convention.

**(b) Invariant and noninvariant penalties.** We used

\[
R_{\mathrm{corr}} = \frac{1}{d(d-1)}\sum_{i\ne j} C_{ij}^2,\quad
R_{\mathrm{cov}} = \frac{1}{d(d-1)}\sum_{i\ne j}\left(\frac{\Sigma_{ij}}{\mathrm{tr}(\Sigma)/d}\right)^2,\quad
L=L_{\mathrm{CE}}+\lambda R.
\]

`R_corr` is invariant under positive diagonal `D` (within the numerical guard). `R_cov` is invariant under *uniform* scaling but generally **not** under nonuniform channel scaling. In this experiment, median absolute initial `R_cov` difference was **0.01018** under the heterogeneous gauge, whereas `R_corr` differences were recorded as 0.

The correlation participation ratio is

\[
r_C=\frac{[\mathrm{tr}(C)]^2}{\mathrm{tr}(C^2)}.
\]

It is a scale-invariant diagnostic for this positive diagonal group; it is **not** invariant to an arbitrary invertible mixing of feature coordinates.

**(c) Gauge-covariant optimizer.** A positive diagonal gauge applied to `W,b,V` through `(W',b',V')=(DW,Db,VD⁻¹)` does not change the ReLU model's logits, because `ReLU(Dz)=D ReLU(z)`. For a gauge-invariant loss, the gradient in transformed coordinates obeys the chain rule. An ordinary update in the original coordinates becomes the transformed update exactly if the transformed gradients are multiplied by `D²` on the rows of `W'`, by `D²` on `b'`, and by `D⁻²` on the columns of `V'`. Other layers use the unmodified update.

To see this explicitly for `W`, with `W'=DW`,

\[
\nabla_{W'}L'=D^{-1}\nabla_W L,\qquad
W'_{new}=DW-\eta D^2\nabla_{W'}L'=D(W-\eta\nabla_W L).
\]

Similarly, `V'=VD⁻¹` implies `∇_{V'}L'=(∇_VL)D`, so multiplying the transformed gradient by `D⁻²` restores the output-layer update. Identical initialization and subsequent identical update steps imply the same entire *function-space trajectory* by induction, not merely the first step. **The result relies on the regularizer itself being gauge-invariant.** It does not hold for `R_cov`.

**Dimension check:** `Z` has size n×d; `Σ` and `C` are d×d; `DΣD` matches; the penalty is a dimensionless sum of squared dimensionless correlations; `L_CE + λR` is therefore dimensionally valid under the stated convention.

## 3. Methodology and reproducibility

- Brand-new independent synthetic data generated with seeds `20261121–20261125`, not reused from the preceding gauge experiment. 3,600 samples per task: train 800, validation 400, untouched holdout 2,400; all disjoint, stratified. Train-only standardization. These are same-distribution holdouts, **not** OOD benchmarks.
- Architecture 24→64→24→classes; ReLU; FP64 CPU. Ten full-batch warmup updates LR 0.03, followed by 120 full-batch updates, LR 0.08 or 0.25. Two tasks × 12 initializations × 2 LRs × 3 losses × 3 gauge/optimizer conditions = **432 full training runs**.
- All methods and λ=0.5 were locked before holdout. No hyperparameter selection based on test. The `results/FREEZE_BEFORE_HOLDOUT.json` manifest captures training/code/model hashes before holdout evaluation. The evaluator refuses to repeat when a test result already exists.
- Hardware: AMD EPYC 9V74 (container allocates 5 virtual CPUs; PyTorch uses 1 thread), float64, full-batch size 800, PyTorch 2.10.0+cpu, no GPU, CPU clock uncontrolled. Training wall time is recorded in `results/environment.json`; *no fair speed benchmark is claimed*.
- 5 of 5 numerical/invariance/data tests passed. All 432 training runs completed 120 updates without nonfinite values. Initial prediction and correlation errors: maximum recorded 0. `max_path_diff` on the first 48 validation examples over 120 updates: corrected CE and corrected correlation = 0; corrected covariance max 2.165, median 0.813; ordinary SGD typically nonzero.
- CIs condition on the two fixed datasets and 12 initialization seeds. The heldout observations are *not* treated as independent experimental repetitions. CI ranges are exploratory, unadjusted for comparisons; no universal inferential claim.

### Run from scratch

For a **fresh source-only folder** (do not reuse results that already contain `heldout_test.csv`):

```bash
python -m pip install -r requirements.txt
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py
python freeze.py
python evaluate.py
python report.py
```

The provided full-audit archive contains trained model checkpoints and the completed holdout results. Use the separate source-only package when repeating the holdout workflow with a fresh file system.

## 4. Interpretation, alternatives, and further tests

- **PASS (M1):** channel correlations and predictions did not change under the designated positive diagonal gauge.
- **PASS (M2):** corrected SGD recovered the canonical path for both CE and gauge-invariant correlation penalty in all applicable conditions.
- **PASS (M3):** normalized covariance penalty broke objective-level invariance, and the compensated SGD paths diverged.
- **FAIL as a universal gain:** gauge-invariant correlation-spectrum optimization did not consistently improve heldout prediction. It achieved much larger correlation ranks with both improvements and deterioration of loss in different settings.
- **UNCERTAIN beyond scope:** larger architectures, language models, OOD distribution changes, learning-rate adaptivity, nonlinear feature transforms, AdamW, and global asymptotic convergence.

**Competing explanation:** Some apparent gains from raw feature-spectrum control can be entirely explained by optimizer preconditioning, whereas an explicitly gauge-invariant regularizer has a genuine representational objective effect but may still hurt generalization.

**Uncertainty:** data-generation choice, single architecture and split, initial seeds, nonzero-variance threshold, controlled learning-rate choices, repeated exploration across research rounds. Reported CI is seeded percentile bootstrap and not a full combined standard uncertainty `u_c` across architectures/distributions; `k=1.96` would be a normal approximation and is *not* used to compute these intervals.

**Novelty limits:** A correlation-based redundancy penalty is related to established self-supervised approaches. **Barlow Twins** ([Zbontar et al., ICML 2021](https://proceedings.mlr.press/v139/zbontar21a.html), [arXiv:2103.03230](https://arxiv.org/abs/2103.03230)) uses *cross-view* correlations, not the single-representation autocorrelation penalty used here; **VICReg** ([Bardes et al., ICLR 2022](https://arxiv.org/abs/2105.04906)) couples variance preservation with covariance regularization. This study claims a *specific controlled causal separation and replication*, not the invention of correlation-based redundancy reduction itself.

**Three connected disciplines:** numerical optimization (preconditioning and coordinate dependence), high-dimensional statistics (correlation versus covariance spectra and effective rank), differential geometry (equivalent parameterizations of the same function). A further connection is feedback control for stabilized learning, but convergence was not established here.

**Next falsifiable question:** Can a broader invariance group or explicitly supervised sufficient-information constraint produce consistent heldout gains across genuinely independent data distributions, not merely higher correlation rank?
