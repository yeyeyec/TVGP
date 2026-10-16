#!/usr/bin/env Rscript

source(file.path(dirname(normalizePath(sub("^--file=", "", grep(
  "^--file=", commandArgs(FALSE), value = TRUE)[1L]))), "common.R"))

required <- c("GpGp", "GPvecchia", "FNN")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
verify_fixed_data()

source(file.path(experiment_root(), "vendor", "vecchia_scaled.R"), local = .GlobalEnv)
if (!all(vapply(c("fit_scaled", "fit_vcf", "predictions_scaled"), exists,
                logical(1), mode = "function"))) stop("Incomplete SVecchia source.")

gpgp_noise <- function(fit) {
  no_nugget <- fit$covparms
  no_nugget[length(no_nugget)] <- 0
  covariance <- get(fit$covfun_name, envir = asNamespace("GpGp"))
  x <- fit$locs[1L, , drop = FALSE]
  noise <- as.numeric(covariance(fit$covparms, x) - covariance(no_nugget, x))
  if (length(noise) != 1L || !is.finite(noise) || noise < 0) {
    stop("Unable to recover the fitted nugget variance.")
  }
  noise
}

run_one <- function(data_version, rep) {
  out <- method_output("svecchia", data_version, rep)
  log_file <- file.path(out, "fit.log")
  result <- tryCatch({
    data <- load_experiment_data(data_version, rep)
    fit_seed <- 50000L + rep + if (data_version == "full") 1000L else 0L
    set.seed(fit_seed)
    fit_args <- list(
      y = data$y, inputs = data$X, ms = 200L, trend = "pre", nu = 2.5,
      nug = NULL, scale = "parms", select = Inf, print.level = 1L,
      max.it = 32L, tol.dec = 4, find.vcf = FALSE
    )
    # The n=10,000 fit deliberately uses the upstream default n.est=min(5000,n).
    if (data_version == "full") fit_args$n.est <- 10000L
    sink(log_file, split = FALSE)
    on.exit(sink(), add = TRUE)
    fit_s <- system.time(fit <- do.call(fit_scaled, fit_args))[["elapsed"]]
    if (!isTRUE(fit$conv)) stop("SVecchia covariance optimization did not converge.")

    set.seed(fit_seed + 20000L)
    vcf_s <- system.time({
      fit$vcf <- fit_vcf(fit, m.pred = 200L, scale = "parms", scorefun = ls)
    })[["elapsed"]]
    pred_s <- system.time({
      prediction <- predictions_scaled(
        fit, locs_pred = data$Xt, m = 200L, joint = FALSE,
        predvar = TRUE, scale = "parms"
      )
    })[["elapsed"]]
    mean <- as.numeric(prediction$means)
    response_variance <- pmax(as.numeric(prediction$vars), 0)
    noise_variance <- fit$vcf * gpgp_noise(fit)
    latent_variance <- response_variance - noise_variance
    tolerance <- 2e-6 * max(1, abs(response_variance))
    if (any(!is.finite(latent_variance)) || any(latent_variance < -tolerance)) {
      stop("Invalid latent variance after nugget removal.")
    }
    latent_variance <- pmax(latent_variance, 0)
    draws <- gaussian_draws(mean, latent_variance, 21000L + rep, 300L)
    metrics <- score_draws(data$truth, mean, latent_variance, draws)
    n_est <- if (data_version == "full") 10000L else min(5000L, nrow(data$X))
    row <- cbind(data.frame(
      method = "SVecchia", data_version = data_version, rep = rep,
      n_train = nrow(data$X), n_test = nrow(data$Xt),
      m_fit = 200L, m_vcf = 200L, m_pred = 200L, n_est = n_est,
      n_est_rule = if (data_version == "full") "fixed_10000" else "upstream_default",
      fit_s = unname(fit_s), vcf_s = unname(vcf_s), pred_s = unname(pred_s),
      Time_s = unname(fit_s + vcf_s + pred_s), convergence = TRUE,
      stringsAsFactors = FALSE
    ), metrics)
    write_csv_atomic(row, file.path(out, "metrics.csv"))
    saveRDS(data.frame(
      row_id = data$source_ids$test_id, truth = data$truth, mean = mean,
      latent_variance = latent_variance,
      response_variance = response_variance
    ), file.path(out, "predictions.rds"), compress = FALSE)
    saveRDS(list(
      covfun_name = fit$covfun_name, covparms = fit$covparms,
      convergence = fit$conv, vcf = fit$vcf,
      fitted_noise_variance = noise_variance
    ), file.path(out, "fit_summary.rds"), compress = FALSE)
    row
  }, error = function(error) {
    failure <- data.frame(
      method = "SVecchia", data_version = data_version, rep = rep,
      error = conditionMessage(error), stringsAsFactors = FALSE
    )
    write_csv_atomic(failure, file.path(out, "failure.csv"))
    failure
  })
  result
}

grid <- data_grid()
workers <- min(as.integer(Sys.getenv("SVEC_REP_WORKERS", "4")), nrow(grid))
if (!is.finite(workers) || workers < 1L) workers <- 1L
runner <- function(i) run_one(grid$data_version[i], grid$rep[i])
results <- if (.Platform$OS.type == "unix" && workers > 1L) {
  parallel::mclapply(seq_len(nrow(grid)), runner, mc.cores = workers,
                     mc.preschedule = FALSE)
} else lapply(seq_len(nrow(grid)), runner)
failed <- vapply(results, function(x) "error" %in% names(x), logical(1))
if (any(failed)) stop(sum(failed), " SVecchia runs failed; inspect per-run failure.csv files.")
