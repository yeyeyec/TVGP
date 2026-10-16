# Fit TVGP (or the global and dense ablation arms) to one dataset and predict the test set.
# Requires tvgp/tvgp_smc.R, and tvgp/dense_backend.R for the dense arms.

write_csv <- function(x, path) utils::write.csv(x, path, row.names = FALSE, na = "")

# Settings used in the paper. Storm surge uses the inverse-gamma covariance prior and the
# optimizer settings of Supplementary Section B.1; all other experiments use the truncated
# jointly robust prior (a = 0.2, b = 1, g_min = 1e-6).
tvgp_settings <- function(experiment, rep, m = 30L, n_particles = 5000L, cores = 16L) {
  s <- list(n_particles = as.integer(n_particles), m = as.integer(m), nu = 2.5,
    cores = as.integer(cores), n_draws = 300L,
    seed_fit = 50000L + rep, seed_predict = 21000L + rep,
    n_candidates_per_axis = 10L, root_maxit = 600L, full_maxit = 400L,
    cov_prior = "jointly_robust",
    cov_prior_control = list(jr_a = 0.2, jr_b = 1, g_min = 1e-6,
                             jr_definition = "gu_2019", jr_C_method = "range_scaled"),
    leaf_prior_control = list(m0 = 0, V0 = 100, a0 = 2, b0 = 1),
    smc_options = list(), pointwise_cores = cores, joint_batch_size = 4L)
  if (startsWith(experiment, "storm_")) {
    s$root_maxit <- 300L; s$full_maxit <- 200L
    s$cov_prior <- "inverse_gamma"
    s$cov_prior_control <- list(a_phi = 2.5, b_phi = 1, a_g = 2, b_g = 0.1)
    # At most twelve split rules per node, parent warm starts, a dedicated start for
    # constant-response leaves, and looser optimizer tolerances.
    s$smc_options <- list(candidate_budget = 12L, optimizer_gradient_tol = 1e-2,
      optimizer_factr = 1e6, optimizer_newton_maxit = 20L,
      parent_warm_start = TRUE, zero_fast_start = TRUE)
    s$pointwise_cores <- 1L; s$joint_batch_size <- 1L
  }
  s
}

fit_global_model <- function(data, s) {
  N <- nrow(data$X)
  ctx <- TVGP_prepare_context(data$X, s$m, ordering = "global", exact_nn_fallback = TRUE, verbose = FALSE)
  root <- TVGP_optimize_node_with_retries(data$y, data$X, seq_len(N), s$nu, s$m, ctx,
    cov_prior = s$cov_prior, cov_prior_control = s$cov_prior_control,
    leaf_prior_spec = TVGP_leaf_prior_spec(s$leaf_prior_control), maxit = s$root_maxit, hessian = TRUE)
  if (isTRUE(root$optimization_failed)) stop("Global GP optimization failed: ", root$optimization_failure_reason)
  p <- TVGP_init_particle(root, N, s$m); p$nodes[[1]]$terminal <- TRUE
  list(final_particles = list(p), history_weights = list(1), diagnostics = list(), root_fit_diagnostics = root)
}

# Keep one copy of each distinct particle, with its summed weight.
compact_population <- function(fit) {
  p <- fit$final_particles; w <- tail(fit$history_weights, 1)[[1]]
  keys <- vapply(p, TVGP_tree_signature, character(1), include_params = TRUE)
  groups <- split(seq_along(p), keys)
  fit$original_particle_count <- length(p)
  fit$particle_multiplicities <- vapply(groups, length, integer(1))
  fit$final_particles <- p[vapply(groups, `[`, integer(1), 1L)]
  fit$history_weights <- list(vapply(groups, function(i) sum(w[i]), numeric(1)))
  fit$history_particles <- NULL; fit$particle_diagnostics <- NULL; fit$context <- NULL
  fit
}

predict_population <- function(fit, data, s, mode = "pointwise", Xt = data$Xt, n_draws = s$n_draws) {
  if (mode == "pointwise") {
    chunks <- split(seq_len(nrow(Xt)), ceiling(seq_len(nrow(Xt)) / 250L))
    ans <- lapply(seq_along(chunks), function(k) TVGP_predict(fit, data$X, data$y,
      Xt[chunks[[k]], , drop = FALSE], nu = s$nu, m_vecchia = s$m, predict_obs = FALSE,
      n_draws = n_draws, draw_seed = s$seed_predict + k - 1L, parallel_cores = s$pointwise_cores,
      prediction_mode = "pointwise", return_covariance = FALSE))
    return(list(mean = unlist(lapply(ans, `[[`, "mean")), variance = unlist(lapply(ans, `[[`, "variance")),
                draws = if (n_draws > 0) do.call(rbind, lapply(ans, `[[`, "draws")) else NULL))
  }
  TVGP_predict(fit, data$X, data$y, Xt, nu = s$nu, m_vecchia = s$m, predict_obs = FALSE,
    n_draws = n_draws, draw_seed = s$seed_predict, parallel_cores = s$pointwise_cores,
    prediction_mode = "joint", joint_ordering = "maximin", return_covariance = FALSE,
    joint_batch_size = s$joint_batch_size)
}

# method: "tvgp", or the ablation arms "global_vecchia", "treed_dense", "global_dense".
run_tvgp <- function(method, data, s, outdir) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  dense <- method %in% c("treed_dense", "global_dense")
  if (dense) { restore_dense <- install_dense_backend(); on.exit(restore_dense(), add = TRUE) }
  set.seed(s$seed_fit)
  fit_s <- system.time({
    fit <- if (method %in% c("global_vecchia", "global_dense")) fit_global_model(data, s) else
      do.call(TVGP_SMC, c(list(data$y, data$X, n_particles = s$n_particles, nu = s$nu,
        m_vecchia = s$m, alpha = 0.95, beta = 0.5, n_candidates_per_axis = s$n_candidates_per_axis,
        min_leaf_size = s$m, root_maxit = s$root_maxit, full_maxit = s$full_maxit, hessian = TRUE,
        ordering = "global", exact_nn_fallback = TRUE, parallel_cores = s$cores,
        store_history_particles = FALSE, store_step_particle_diagnostics = FALSE,
        cov_prior = s$cov_prior, cov_prior_control = s$cov_prior_control,
        leaf_prior_control = s$leaf_prior_control, verbose = FALSE), s$smc_options))
  })[["elapsed"]]
  if (length(fit$diagnostics))
    write_csv(do.call(rbind, lapply(fit$diagnostics, as.data.frame)), file.path(outdir, "smc_diagnostics.csv"))
  fit <- compact_population(fit)
  saveRDS(fit, file.path(outdir, "fitted_population.rds"))
  write_csv(TVGP_tree_signature_summary(fit), file.path(outdir, "tree_probabilities.csv"))
  write_csv(TVGP_leaf_covariance_summary(fit, data$y), file.path(outdir, "leaf_parameters.csv"))
  predictions <- list()
  for (mode in if (dense) "pointwise" else c("pointwise", "joint")) {
    pred <- NULL
    pred_s <- system.time(tryCatch(
      pred <- if (dense) predict_dense_population(fit, data$X, data$y, data$Xt, s) else
        predict_population(fit, data, s, mode),
      error = function(e) {
        # Joint prediction can fail on nearly degenerate covariances; pointwise results are kept.
        if (mode == "pointwise") stop(e)
        warning("Joint prediction failed: ", conditionMessage(e))
      }))[["elapsed"]]
    if (!is.null(pred)) predictions[[mode]] <- c(pred, list(fit_s = fit_s, pred_s = pred_s))
  }
  list(predictions = predictions, fit_s = fit_s, settings = s)
}
