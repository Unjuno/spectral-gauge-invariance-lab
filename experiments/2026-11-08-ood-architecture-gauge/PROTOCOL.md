# Pre-holdout protocol — 2026-11-08

## Research question
Do the diagonal-gauge equivariance of feature-correlation regularization and the compensated-SGD trajectory equality persist when the DATA DISTRIBUTION and UPSTREAM NETWORK ARCHITECTURE change? Does the correlation regularizer deliver a consistent performance gain under a prespecified label-preserving covariate shift?

## Frozen design (before training and holdout evaluation)

- Two label-generating synthetic tasks: `moons24`, `prototypes4` (a four-class fixed-prototype model). Input dimension 24, all features dimensionless.
- Two architectures: `plain` 24->48(ReLU)->24(ReLU)->C, `residual` 24->48(ReLU)->residual block with GELU branch->24(ReLU)->C. Fixed positive diagonal reparameterization acts ONLY on the final 24-channel ReLU feature layer and its immediately following linear head.
- 6 matched seeds 8200–8205; 2 learning rates 0.08 and 0.25; 3 objectives CE, CE+0.5*mean(off-diagonal correlation squared), CE+0.5*mean(off-diagonal normalized-covariance squared); 3 variants identity+SGD, heterogeneous positive gauge (.25,1,4 repeating)+SGD, heterogeneous+coordinate-corrected SGD; total 2×2×6×2×3×3 = 432 runs.
- 8 warmup full-batch SGD iterations LR=0.03; 90 additional SGD iterations per experiment, no momentum or weight decay, float64, one CPU thread, full-batch train 640. Identity/SGD serves as matched reference for each task/arch/seed/lr/objective. 48 within-objective corrected paths per objective (2×2×6×2).
- Train 640, validation 320, independent ID holdout 1200, independent covariate-shift OOD holdout 1200. All train/validation/ID generated from one source, with fixed disjoint indices; OOD independently generated with the same label rule, increasing causal feature noise and shifting nuisance channels. Train-only scaler. Holdouts must not be used in training or parameter selection. No selection among regularizer strengths (fixed 0.5).
- Every trained model is stored (not necessarily published). Evaluate ID and OOD exactly once after source, training metrics, and checkpoint checksums are frozen.
- Primary algebraic validation thresholds: initial predictions <1e-9, initial per-channel correlations <1e-9, entire corrected SGD path max deviation <1e-7 (using fixed 48 validation probe points). Covariance objective is a negative control: at least one corrected path is >1e-7. Record any failure, pathological near-zero channel variance, or nonfinite result without omission.
- Functional performance hypothesis H4: correlation regularization improves *mean OOD cross-entropy* vs CE reference in all 4 task×architecture cells, both tested learning rates; exploratory effect-size summary with paired seeds. H4 PASS only if mean paired delta <0 in all 8 cells; otherwise FAIL. OOD balanced accuracy and ID CE are guardrails. No claims of universal significance based on 6 seeds and fixed source distributions.
- Primary OOD comparisons: identity SGD corr-vs-CE within task×arch×lr, paired by seed, using 5,000 percentile bootstrap resamples of 6 paired seeds. Do not interpret multiple positive results as multiplicity-corrected confirmatory tests.
- Environment/uncertainty: `torch==2.10.0+cpu`, numpy/sklearn versions logged, CPU hardware and clock unconstrained, float64, BLAS single-thread. With 6 seeds there is substantial seed uncertainty; we do not estimate across datasets or platforms. Different hardware BLAS may not be bitwise identical. Finite-horizon trajectory equality is not a convergence theorem. OOD shift is synthetic, not real-world out-of-distribution generalization.

## H/T/D/C/U
H: exact gauge-path equality for CE and correlation, not generally for normalized covariance; mixed OOD regularizer performance.
T: 432 fixed-run paired grid, 90 updates, no retuning, train/holdout split, separate OOD generator.
D: thresholds above; reject universal OOD gain if any of eight comparisons is nonnegative; classify otherwise descriptive.
C: channel variance near numerical guard, different activations at the gauge target, optimizer hyperparameters, or dataset difficulty alter observed paths. OOD effects may reflect added regularization not gauge symmetry.
U: seeds (n=6), fixed synthetic labels, shift specification, BLAS/precision, multiple contrasts, empirical variance estimation and degeneracy.

## Prespecified arithmetic erratum (before test evaluation)
The first draft mistakenly wrote `216` for the 2×2×6×2×3×3 grid and `36` corrected paths per objective. Correct totals are **432** and **48**. A first training run executed all 432 models but stopped at the incorrect post-run assertion; the revised code was rerun with unchanged model/data/loss settings before accessing ID/OOD tests. No metric or hyperparameter was selected as part of this correction.
