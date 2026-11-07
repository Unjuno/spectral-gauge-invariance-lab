# Locked protocol — 2026-11-08 (before experiments)

Question: After accounting for SGD coordinate dependence, does a scale-invariant feature-correlation-spectrum regularizer exert a distinguishable learning effect, and does it improve held-out prediction?

## Three mechanistic claims (primary)
- M1: Given positive diagonal D and fixed trained ReLU model with transformed adjacent parameters, predictions and feature channel-correlation matrix remain unchanged (numerical atol 1e-9); covariance eigenspectrum need not.
- M2: For CE alone or CE + correlation redundancy penalty, *gauge-corrected* full-batch SGD preserves the function training path over 120 steps for identity vs heterogeneous positive diagonal reparameterization (max held-out validation-logit error < 1e-7). Plain SGD need not.
- M3: A trace-normalized covariance off-diagonal penalty is NOT diagonal-gauge invariant. Therefore correcting optimizer alone will not recover baseline path with this noninvariant loss. This contrast distinguishes optimizer-coordinate effects from objective-coordinate effects.

## Predictive and geometric effect (secondary, direction not presupposed)
Correlation penalty (CE + 0.5 × average squared off-diagonal correlation) vs CE alone: estimate held-out loss/accuracy and active channel count, covariance-spectrum and correlation-spectrum effective ranks, correlation redundancy. Genuine regularizer impact exists if *under the same canonical optimizer*, train correlation objective and path differ (they may yield better, worse, or unchanged test performance). Undercompensated gauge-varying objectives are not interpreted as improvements. Do not optimize settings on holdout.

## Design
- Use brand-new independent synthetic datasets with deterministic generator seeds distinct from previous experiment: classification6 (6 balanced-ish classes, informative/redundant features) and moons22 (2 moons + 22 noise features). Each dataset size 3,600; stratified disjoint train 800, validation 400, holdout 2,400. StandardScaler fitted exclusively on train. Fixed dataset seeds 20261121/20261122; split 20261123/20261124; nuisance seed 20261125. Same distribution for independent holdout: NOT OOD.
- Net 24→64→24→C ReLU. 10 full-batch warmup steps at learning rate 0.03. Then 120 full-batch plain SGD steps at lr 0.08 and 0.25. No momentum/weight decay. dtype float64, CPU, torch 1 thread, deterministic.
- 12 independent initialization seeds 7200..7211. Conditions: objectives CE; CE + 0.5 correlation redundancy; CE + 0.5 normalized covariance redundancy. Each evaluated with identity+plain SGD, hetero gauge+plain SGD, hetero gauge+compensated SGD; 2×12×2×3×3 = 432 training runs. Hetero D repeating [0.25,1,4] in 24 dimensions.
- Denominator for normalized covariance penalty is mean feature variance squared; regularizer is uniform-scale invariant but not channel-scale invariant. Correlation has zero-variance guard; if a channel variance below 1e-16, inverse SD set to 0. This introduces approximate rather than universally exact gauge invariance near threshold; verify whether threshold affected any observed nonzero channels and note if it does.
- Monitor training loss, validation loss, all finite failures, each regularizer, correlation rank, dead channels, trace-normalized covariance offdiag; per-step probe logits on first 48 *validation* items only.

## Evaluation and decisions
- Run all 432 training configurations without reading test data, write immutable manifests/checkpoints and training metrics. The only selection is fixed conditions above. Freeze file hashes before running final evaluation script.
- Evaluate holdout once, report 12-seed paired mean effects with seed bootstrap (5000 repetitions) per task / lr; CIs descriptive and not adjusted for the 4 secondary comparisons. Primary invariance comparisons are mathematical/threshold tests, not significance tests.
- PASS M1 if output and C errors <=1e-9, PASS M2 if max probe logit difference<=1e-7, PASS M3 if demonstrated nonzero loss difference and nonzero path difference. For predictive improvement, only call supported if seed-bootstrap 95% interval on test cross entropy is strictly below 0 for *both independent tasks at a prespecified lr*; otherwise report negative or inconclusive, no universal claim.
- Numerical precision, split/seed sensitivity, correlation near constant channels, no GPU, one architecture, two synthetic datasets and fixed splitting are limitations. No asymptotic convergence claim.

## Prospective causal caution
- Changing non-invariant objectives while correcting only SGD cannot isolate information-theoretic effects; the transformed objective itself changes.
- Even a gauge-invariant correlation penalty is not automatically information preserving: dead features reduce correlation penalties. Monitor active channels, class metrics, and calibration.
