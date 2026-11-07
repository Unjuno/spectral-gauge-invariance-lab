# Locked experimental protocol — 2026-11-07

## Narrow objective
Identify experimentally useful and falsifiable properties of feature covariance-spectrum manipulation under exact function-preserving parameter transformations. Distinguish representational geometry from SGD coordinate dependence. Do not claim causal gains or asymptotic convergence from finite runs.

## Primary mathematically identified contrasts
1. Positive diagonal rescaling of penultimate ReLU features and reciprocal rescaling of the output weights must leave the entire network output unchanged to numerical tolerance, while changing feature covariance spectrum.
2. Ordinary SGD may yield different training trajectories under equivalent reparameterizations.
3. Gauge-compensated SGD must recover the reference trajectory, apart from floating-point error. This is a numerical confirmation of an algebraic result, *not* proof that all other spectrum manipulations lack value.

## Data and architecture
Two seeded, disjoint-split datasets: synthetic 6-class nonlinear classification (make_classification, n=3,600) and nonlinear binary moons (make_moons with 22 independent nuisance features, n=3,600). Deterministic stratified train 800, validation 400, test 2,400. Standardization fit on training only. The holdout test split is never accessed before lock of all methods, settings and evaluation code. Model: 24->64->24->C, ReLU, 15 full-batch warmup SGD steps with lr 0.03. Second layer (24 channels) is transformed. Float64, CPU, torch deterministic algorithms, one thread. No momentum, clipping or weight decay.

## Conditions
Matched initializations: 12 seeds 6100..6111. Fixed stress learning rates 0.08 and 0.25. 100 post-warmup full-batch steps.
Gauges: identity, uniform shrink 0.25, uniform expand 4.0, heterogeneous fixed channel scales (repeat [0.25, 1, 4], 8x); all exactly positive.
Optimizers: (a) vanilla SGD, (b) fixed diagonal gauge-compensated SGD with per-parameter learning-rate multipliers D^2 on layer2 rows and D^-2 on output columns; all other parameters multiplier 1. Identical updates in canonical function coordinates in exact arithmetic.

## Metrics and decisions
Record: exact initial function discrepancy; train/test CE and balanced accuracy; pre/post trace, covariance effective rank, top eigenvalue fraction, normalized spectrum entropy; best and last train CE; gradient finite and failures. For compensated SGD, compare *per-step* logits to identity trajectory on heldout validation only, and compare final predictions on test after training is frozen. Report median (range) across 12 matched seeds, paired mean deltas and bootstrap 95% CI resampling seeds (5000, fixed independent RNG). Do not treat 2400 test items as independent model replicates. Report GPU none, hardware, threads, precision, batch size, optimizer.

Primary expected PASS (invariance): max initial pred diff <= 1e-10 for all, max compensated trajectory diff <= 1e-8 for valid finite paths.
Primary exploratory hypothesis: vanilla SGD's parameterization sensitivity is larger than compensated SGD's; meaningful test CE contrast if mean signed difference CI excludes 0, with multiple-comparison warning. Secondary: gauge may change effective rank without changing prediction; demonstrate paired invariance and geometric change.

## Disclosure / scope
Do not tune on test; no hyperparameter search using heldout. No new claim of LLM performance, general optimizer invariance, or guaranteed training convergence. All failures included. Code, tests, protocol, raw CSV and figures published openly; synthetic data regeneration avoids redistributing private inputs.