# Shared helpers for the storm-surge comparators (Section 5.2 and Supplement C.8).
# STORM_HOLDOUT selects the holdout design: "random" (default; Table 2) or "grouped"
# (all landfall realizations of a storm design held out together). Results are written to
# results/storm_surge/<holdout>/ unless STORM_OUTPUT is set. The data are not distributed;
# STORMSURGE_DATA_PATH must point to the storm-surge .RData file (see README.md).

experiment_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(file_arg)) stop("Run this script with Rscript.")
  dirname(dirname(normalizePath(sub("^--file=", "", file_arg[1L]), mustWork = TRUE)))
}
source(file.path(experiment_root(), "R", "datasets.R"))

holdout <- function() {
  h <- Sys.getenv("STORM_HOLDOUT", unset = "random")
  if (!h %in% c("random", "grouped")) stop("STORM_HOLDOUT must be random or grouped.")
  h
}

run_root <- function() {
  path <- Sys.getenv("STORM_OUTPUT", unset = file.path(experiment_root(), "results", "storm_surge", holdout()))
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, mustWork = TRUE)
}

data_grid <- function() {
  expand.grid(data_version = c("n10000", "full"), rep = seq_len(10L), stringsAsFactors = FALSE)
}

# data_version "n10000" (about 10,000 training runs) or "full" (about 50,650).
load_experiment_data <- function(data_version, rep) {
  prefix <- if (holdout() == "grouped") "storm_grouped_" else "storm_"
  data <- make_dataset(paste0(prefix, data_version), as.integer(rep))
  # Row-level design groups; the tuning splits of laGP and JGP keep groups intact.
  data$source_ids <- list(train_row_group_id = data$train_group_id)
  data$data_version <- data_version
  data$rep <- as.integer(rep)
  data
}

# The splits are regenerated deterministically from the source data.
verify_fixed_data <- function() invisible(TRUE)

# Training rows held out to tune laGP and JGP (n_validation rows; whole design groups under
# the grouped holdout). Test responses are never used for tuning.
tuning_split <- function(data, n_validation, seed) {
  set.seed(seed)
  row_groups <- data$source_ids$train_row_group_id
  if (is.null(row_groups)) {
    validation <- sample.int(nrow(data$X), n_validation)
  } else {
    ordered_groups <- sample(unique(row_groups))
    cumulative <- cumsum(tabulate(match(row_groups, ordered_groups), nbins = length(ordered_groups)))
    validation_groups <- ordered_groups[seq_len(which(cumulative >= n_validation)[1L])]
    validation <- which(row_groups %in% validation_groups)
  }
  list(training = setdiff(seq_len(nrow(data$X)), validation), validation = validation)
}

normal_crps <- function(truth, mean, sd) {
  truth <- as.numeric(truth); mean <- as.numeric(mean); sd <- as.numeric(sd)
  out <- abs(truth - mean)
  positive <- is.finite(sd) & sd > 0
  z <- (truth[positive] - mean[positive]) / sd[positive]
  out[positive] <- sd[positive] * (z * (2 * stats::pnorm(z) - 1) + 2 * stats::dnorm(z) - 1 / sqrt(pi))
  out
}

sample_crps <- function(truth, draws) {
  draws <- as.matrix(draws)
  ordered <- t(apply(draws, 1L, sort))
  k <- ncol(ordered)
  rowMeans(abs(draws - truth)) - as.vector(ordered %*% (2 * seq_len(k) - k - 1)) / k^2
}

score_draws <- function(truth, mean, variance, draws) {
  if (length(mean) != length(truth) || length(variance) != length(truth) ||
      !identical(nrow(draws), length(truth)) || any(!is.finite(mean)) ||
      any(!is.finite(variance)) || any(variance < 0) || any(!is.finite(draws))) {
    stop("Invalid predictive moments or draws.")
  }
  intervals <- t(apply(draws, 1L, stats::quantile, probs = c(0.025, 0.975), names = FALSE))
  sd <- sqrt(variance)
  analytic_lower <- mean - stats::qnorm(0.975) * sd
  analytic_upper <- mean + stats::qnorm(0.975) * sd
  data.frame(
    RMSE = sqrt(mean((mean - truth)^2)),
    CRPS = mean(sample_crps(truth, draws)),
    Cover95 = mean(truth >= intervals[, 1L] & truth <= intervals[, 2L]),
    MeanCI95 = mean(intervals[, 2L] - intervals[, 1L]),
    AnalyticGaussianCRPS = mean(normal_crps(truth, mean, sd)),
    AnalyticCover95 = mean(truth >= analytic_lower & truth <= analytic_upper),
    AnalyticMeanCI95 = mean(analytic_upper - analytic_lower),
    stringsAsFactors = FALSE
  )
}

gaussian_draws <- function(mean, variance, seed, n_draws = 300L) {
  set.seed(seed)
  mean + sqrt(variance) * matrix(stats::rnorm(length(mean) * n_draws), nrow = length(mean), ncol = n_draws)
}

write_csv_atomic <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp_", Sys.getpid())
  utils::write.csv(x, temporary, row.names = FALSE)
  if (!file.rename(temporary, path)) stop("Could not finalize ", path)
  invisible(path)
}
write_csv <- write_csv_atomic

atomic_rds <- function(x, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp_", Sys.getpid())
  saveRDS(x, temporary, compress = compress)
  if (!file.rename(temporary, path)) stop("Could not finalize ", path)
  invisible(path)
}

method_output <- function(method, data_version = NULL, rep = NULL) {
  path <- file.path(run_root(), method)
  if (!is.null(data_version)) path <- file.path(path, data_version)
  if (!is.null(rep)) path <- file.path(path, sprintf("rep%03d", as.integer(rep)))
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  normalizePath(path, mustWork = TRUE)
}

scale_response <- function(y) {
  center <- mean(y)
  scale <- stats::sd(y)
  if (!is.finite(scale) || scale <= 1e-12) scale <- 1
  list(y = (y - center) / scale, center = center, scale = scale)
}
