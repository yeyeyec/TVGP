# TVGP: Bayesian treed Vecchia Gaussian process emulation via sequential Monte Carlo

Code for C. Ye and P. Ma, *Bayesian Treed Vecchia Gaussian Process Emulation via Sequential
Monte Carlo* (submitted to *Bayesian Analysis*).

TVGP partitions the input space with a Bayesian CART prior and fits a Gaussian process with a
Vecchia likelihood in each region. Leaf means and variances are integrated analytically under a
normal–inverse-gamma prior, covariance parameters by a Laplace approximation with analytic
derivatives, and trees are sampled by sequential Monte Carlo with a one-step lookahead proposal.

## Layout

```
tvgp/                  the method
  tvgp_smc.R           leaf model, Laplace approximation, SMC sampler, prediction
  tvgp_core.cpp        Vecchia likelihood with analytic gradient and Hessian, Vecchia prediction
  tvgp_dense.cpp       full-covariance leaf likelihood (ablation of Section 4.2)
  dense_backend.R      switches tvgp_smc.R to the full-covariance likelihood
R/
  datasets.R           data-generating models and training/test designs of all experiments
  fit_tvgp.R           TVGP fit and prediction for one dataset, with the settings of the paper
  competitors.R        TGP, DGP VEC, SVecchia, SPGP, JGP, laGP
  metrics.R            RMSE, CRPS, coverage and width of 95% intervals
experiments/run.R      one method on one replication of one experiment
sampler_comparison/    SMC and RJ-MCMC on a common tree space (Section 4.1); official tgp chains
storm_surge/           SVecchia, laGP, JGP, and DGP VEC for the storm-surge application (Section 5)
vendor/                third-party code: JumpGaussianProcess (JGP), SPGP, scaled Vecchia
```

## Requirements

- R (4.2 or later) with `Rcpp`, `RcppArmadillo`, `RcppParallel`, `digest`, and `lhs`; the C++
  core is compiled by `Rcpp::sourceCpp` the first time `tvgp/tvgp_smc.R` is sourced.
- Competitors: `tgp` 2.4-23, `deepgp` 1.2.3, `GpGp`, `GPvecchia`, `FNN`, `laGP`, `posterior`;
  SPGP also needs `igraph`, `fields`, `genlasso`, `Rfast`, `deldir`, `RANN`, `Matrix`, and
  `ramcmc`, and its BRISC sources in `vendor/SPGP/BRISC` compiled to `BRISC.so`.
- JGP: Python 3.9–3.12 with the packages in `vendor/jumpgp_requirements.txt`; set
  `TVGP_BASELINE_PYTHON` (simulations) or `PYTHON_BIN` (storm surge) to that interpreter.

## Quick start

```r
source("tvgp/tvgp_smc.R")
source("R/datasets.R")
d <- make_dataset("sim1", rep = 1, size = c(1000, 500))   # 2-dimensional example, reduced size
fit <- TVGP_SMC(d$y, d$X, n_particles = 200, nu = 2.5, m_vecchia = 30, min_leaf_size = 30,
                cov_prior = "jointly_robust",
                cov_prior_control = list(jr_a = 0.2, jr_b = 1, g_min = 1e-6),
                leaf_prior_control = list(m0 = 0, V0 = 100, a0 = 2, b0 = 1))
pred <- TVGP_predict(fit, d$X, d$y, d$Xt, nu = 2.5, m_vecchia = 30)
sqrt(mean((pred$mean - d$truth)^2))
```

`TVGP_predict` returns pointwise predictive means and variances of the latent function; set
`prediction_mode = "joint"` for joint predictive draws.

## Reproducing the paper

Each command fits one method to one replication (`<rep>` = 1, …, 10) and writes
`results/<experiment>/<method>/rep<rep>/metrics.csv`. The paper used 16 CPU cores per fit
(`--cores`). Add `--size 400,100` to any `experiments/run.R` command for a quick check.

| Paper | Command |
|---|---|
| Section 4.1, Figure 1 | `Rscript sampler_comparison/run.R fit <sim1\|ns10_axis> <run> <smc\|rjmcmc>`, then the same with `evaluate` |
| Supplement C.6, TGP chains | `Rscript sampler_comparison/official_tgp.R fit <sim1\|ns10_axis> <chain>`, then `summarize <example>` |
| Section 4.2, Figure 2 | `Rscript experiments/run.R ablation_n<500\|1000\|2000> <tvgp\|treed_dense\|global_vecchia\|global_dense> <rep>`; at n = 10,000, `sim1 tvgp` and `sim1 global_vecchia` |
| Section 4.3, Figure 3 | `Rscript experiments/run.R sim1 <tvgp\|tgp\|deepgp\|scaled_vecchia\|spgp\|jumpgp> <rep>` |
| Section 4.4, Figure 4 | `Rscript experiments/run.R ns10_axis <tvgp\|tgp\|deepgp\|scaled_vecchia\|jumpgp> <rep>` |
| Supplement C.2 | `Rscript experiments/run.R sim1 tvgp <seed> --data-rep 1 --particles <N>` |
| Supplement C.3 | `Rscript experiments/run.R smooth2d <method> <rep>` |
| Supplement C.4 | `Rscript experiments/run.R ns10_rotated <method> <rep>` |
| Supplement C.5 | `Rscript experiments/run.R sim1 tvgp <rep> --jr <a>,<b>,<g_min>` |
| Supplement C.8, exact zeros | `Rscript experiments/run.R <zero30\|zero70\|noise70> <tvgp\|tgp> <rep>` |
| Section 5, Table 2 | TVGP: `Rscript experiments/run.R <storm_n10000\|storm_full> tvgp <rep>`; competitors: `Rscript storm_surge/run_svecchia.R`, `Rscript storm_surge/run_lagp.R`, `bash storm_surge/run_jgp_all.sh`, `Rscript storm_surge/run_deepgp.R 1` |
| Supplement C.8, grouped holdout | TVGP: `storm_grouped_n10000` and `storm_grouped_full`; competitors: as above with `STORM_HOLDOUT=grouped` |

The storm-surge settings of TVGP (inverse-gamma covariance priors, at most twelve split rules
per node, parent warm starts, and the optimizer tolerances of Supplementary Section B.1) are set
in `tvgp_settings()` in `R/fit_tvgp.R`; storm-surge fits use C++ threads instead of forked
processes (`options(tvgp.parallel_backend = "threads")`).

## Storm-surge data

The 56,277 ADCIRC simulations of Ma et al. (2022) are not distributed with this code. With
access to the archive, set `STORMSURGE_DATA_PATH` to an `.RData` file containing

- `dat`: runs × locations matrix of peak surge elevation (m),
- `inputs`: runs × 7 matrix (ΔP, R_p, V_f, θ, B, landfall longitude, landfall latitude),
- `lonlat.sub`: locations × 2 matrix of longitude and latitude.

The paper uses location 500. `make_dataset()` builds both holdout designs from these objects.

## Notes

- The data-generating functions reproduce the paper's designs exactly; responses agree up to
  floating-point rounding across platforms.
- `tvgp/tvgp_smc.R` merges the versions used for the different experiments. The fits of
  Sections 4.2–4.4 and Supplementary Sections C.2–C.4 used an earlier version that normalized
  the jointly robust prior on g > 0 and optimized log g; this version normalizes the prior on
  g ≥ g_min and optimizes log(g − g_min), as described in the paper. With g_min = 1e-6, leaf
  log marginal likelihoods from the two versions differed by less than 1e-4 in our checks.
- Third-party code in `vendor/` keeps its own license and provenance files.
