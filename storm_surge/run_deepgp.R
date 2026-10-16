#!/usr/bin/env Rscript

source(file.path(dirname(normalizePath(sub("^--file=", "", grep(
  "^--file=", commandArgs(FALSE), value = TRUE)[1L]))), "common.R"))
if (!requireNamespace("deepgp", quietly = TRUE)) stop("Missing R package: deepgp")
if (utils::packageVersion("deepgp") != "1.2.3") {
  stop("DGP VEC is audited for deepgp 1.2.3; found ",
       as.character(utils::packageVersion("deepgp")), ".")
}
verify_fixed_data()

fit_cores <- min(4L, as.integer(Sys.getenv("CORES", "16")))
if (!is.finite(fit_cores) || fit_cores < 1L) fit_cores <- 1L
if (getFromNamespace("check_cores", "deepgp")(fit_cores) != fit_cores) {
  stop("deepgp cannot use the requested fitting cores on this host.")
}

deepgp_noise <- function(fit, fixed_g) {
  tau <- fit[["tau2_y"]]
  if (is.null(tau)) tau <- fit[["tau2"]]
  if (is.null(tau)) stop("DGP VEC fit does not contain the outer variance trace.")
  g <- fit[["g"]]
  if (is.null(g)) g <- fixed_g
  tau <- as.numeric(tau)
  g <- rep(as.numeric(g), length.out = length(tau))
  value <- mean(tau * g)
  if (!is.finite(value) || value < 0) stop("Invalid DGP VEC nugget variance.")
  value
}

predict_moments <- function(fit, Xt, response_scale, response_center,
                            standardized_noise, cores, m = 70L) {
  chunks <- split(seq_len(nrow(Xt)), ceiling(seq_len(nrow(Xt)) / 200L))
  mean <- response_variance <- numeric(nrow(Xt))
  for (indices in chunks) {
    prediction <- stats::predict(
      fit, x_new = Xt[indices, , drop = FALSE], m = m,
      lite = TRUE, mean_map = TRUE, return_all = FALSE, cores = cores
    )
    mu <- as.numeric(prediction$mean)
    variance <- as.numeric(prediction$s2)
    if (length(mu) != length(indices) || length(variance) != length(indices) ||
        any(!is.finite(mu)) || any(!is.finite(variance))) {
      stop("DGP VEC returned invalid predictive moments.")
    }
    mean[indices] <- response_center + response_scale * mu
    response_variance[indices] <- response_scale^2 * variance
    rm(prediction)
    invisible(gc(FALSE))
  }
  latent_variance <- response_variance - response_scale^2 * standardized_noise
  tolerance <- 2e-6 * max(1, abs(response_variance))
  if (any(!is.finite(latent_variance)) || any(latent_variance < -tolerance)) {
    stop("Invalid DGP VEC latent variance after removing the outer nugget.")
  }
  list(
    mean = mean, latent_variance = pmax(latent_variance, 0),
    response_variance = response_variance
  )
}

run_one <- function(rep) {
  out <- method_output("dgp_vec", "n10000", rep)
  if (file.exists(file.path(out, "COMPLETE"))) return(read.csv(file.path(out, "metrics.csv")))
  data <- load_experiment_data("n10000", rep)
  scaled <- scale_response(data$y)
  seed_fit <- 50000L + rep
  seed_predict <- 21000L + rep
  fixed_g <- 1e-6
  set.seed(seed_fit)
  fit_s <- system.time({
    fit <- deepgp::fit_two_layer(
      x = data$X, y = scaled$y, nmcmc = 2000L,
      D = ncol(data$X), pmx = TRUE, g_0 = 0.01, true_g = fixed_g,
      v = 2.5, cov = "matern", vecchia = TRUE, m = 70L,
      cores = fit_cores, verb = TRUE
    )
    fit <- deepgp::trim(fit, burn = 1000L, thin = 2L)
    standardized_noise <- deepgp_noise(fit, fixed_g)
  })[["elapsed"]]
  set.seed(seed_predict)
  pred_s <- system.time({
    prediction <- predict_moments(
      fit, data$Xt, scaled$scale, scaled$center,
      standardized_noise, min(4L, fit_cores), 70L
    )
    draws <- gaussian_draws(
      prediction$mean, prediction$latent_variance, seed_predict, 300L
    )
  })[["elapsed"]]
  metrics <- score_draws(
    data$truth, prediction$mean, prediction$latent_variance, draws
  )
  row <- cbind(data.frame(
    method = "DGP VEC", data_version = "n10000", rep = rep,
    n_train = nrow(data$X), n_test = nrow(data$Xt),
    package = "deepgp", package_version = "1.2.3",
    nmcmc = 2000L, burn = 1000L, thin = 2L,
    retained_states = fit$nmcmc, m_fit = 70L, m_predict = 70L,
    allocated_cores = as.integer(Sys.getenv("CORES", "16")),
    fit_cores = fit_cores, fixed_outer_nugget = fixed_g,
    fit_s = unname(fit_s), pred_s = unname(pred_s),
    Time_s = unname(fit_s + pred_s), stringsAsFactors = FALSE
  ), metrics)
  write_csv_atomic(row, file.path(out, "metrics.csv"))
  atomic_rds(data.frame(
    row_id = data$source_ids$test_id, truth = data$truth,
    mean = prediction$mean,
    latent_variance = prediction$latent_variance,
    response_variance = prediction$response_variance
  ), file.path(out, "predictions.rds"))
  atomic_rds(list(
    tau2_y = fit$tau2_y, g = fit$g, ll = fit$ll,
    standardized_noise_variance = standardized_noise,
    response_center = scaled$center, response_scale = scaled$scale
  ), file.path(out, "retained_trace_summary.rds"))
  writeLines("complete", file.path(out, "COMPLETE"))
  rm(fit, prediction, draws)
  invisible(gc(FALSE))
  row
}

results <- vector("list", 10L)
# Replications to fit, e.g. "1" (the single fit of Table 2) or "1:10" (default).
reps_arg <- commandArgs(trailingOnly = TRUE)
reps <- if (length(reps_arg)) eval(parse(text = reps_arg[1L])) else seq_len(10L)
for (rep in reps) {
  results[[rep]] <- tryCatch(run_one(rep), error = function(error) {
    out <- method_output("dgp_vec", "n10000", rep)
    failure <- data.frame(
      method = "DGP VEC", data_version = "n10000", rep = rep,
      error = conditionMessage(error), stringsAsFactors = FALSE
    )
    write_csv_atomic(failure, file.path(out, "failure.csv"))
    failure
  })
}
failed <- vapply(results, function(x) "error" %in% names(x), logical(1))
if (any(failed)) stop(sum(failed), " DGP VEC runs failed; inspect failure.csv files.")
