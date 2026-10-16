# SMC versus RJ-MCMC on the 2- and 10-dimensional examples (Section 4.1, Supplement C.6).
#
#   Rscript sampler_comparison/run.R fit      <sim1|ns10_axis> <run 1-10> <smc|rjmcmc> [--cores C] [--out DIR]
#   Rscript sampler_comparison/run.R evaluate <sim1|ns10_axis> <run 1-10> <smc|rjmcmc> [--out DIR]
#
# Each example uses one dataset (5,000 training and 2,000 test inputs) for all runs, so
# differences among runs come from the samplers alone. `fit` runs SMC with N = 5,000 particles
# or one RJ-MCMC chain of 50,000 iterations (burn-in 20,000) and saves populations at
# checkpoints; `evaluate` scores every checkpoint on the test set (RMSE, CRPS, coverage),
# which gives the accuracy-versus-time curves of Figure 1. Both samplers use the tree space and
# leaf model of samplers.R: ten quantile cuts per input, depth at most 3, minimum leaf size 30,
# 30 neighbors, the truncated JR prior, and screened starting values for each leaf fit.
# --toy shrinks the problem (300 training and 100 test inputs) for a quick check.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) stop("Usage: Rscript sampler_comparison/run.R <fit|evaluate> <example> <run> <smc|rjmcmc>")
command <- args[1]; example <- args[2]; run <- as.integer(args[3]); method <- args[4]
option <- function(name, default) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) default else args[i + 1L]
}
toy <- "--toy" %in% args
root <- normalizePath(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
source(file.path(root, "R/datasets.R"))
source(file.path(root, "sampler_comparison/samplers.R"))
source(file.path(root, "sampler_comparison/trajectory.R"))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

if (!example %in% c("sim1", "ns10_axis")) stop("example must be sim1 or ns10_axis")
d <- make_dataset(paste0("sampler_", example), 1L, if (toy) c(300L, 100L) else NULL)
data <- list(X = d$X, y = d$y, X_test = d$Xt, y_test = d$truth, experiment = example,
             function_name = if (example == "sim1") "2-dimensional example" else "10-dimensional example",
             seed = 1L, response_center = 0, response_scale = 1, deterministic = FALSE)
out <- option("out", file.path(root, "results", "sampler_comparison", example,
                               sprintf("run%03d", run), method))
if (command == "fit") {
  run_trajectory_fit(.tvgp_samplers, data, method, run, out, toy = toy,
                     total = if (toy) 600L else 50000L, burn = if (toy) 200L else 20000L,
                     cores = as.integer(option("cores", 16L)))
} else if (command == "evaluate") {
  evaluate_trajectory(.tvgp_samplers, data, out)
} else stop("command must be fit or evaluate")
