atomic_rds <- function(x, file) {
  tmp <- paste0(file, ".tmp")
  saveRDS(x, tmp, compress = FALSE)
  if (!file.rename(tmp, file)) stop("Cannot write ", file)
}

write_status <- function(out, phase, status, message = "") {
  write.csv(data.frame(phase = phase, status = status, message = message,
    time = format(Sys.time(), tz = "UTC", usetz = TRUE)),
    file.path(out, paste0(phase, "_status.csv")), row.names = FALSE)
}

checkpoint_iterations <- function(total, burn) {
  sort(unique(c(0L, c(1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 40000)[
      c(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
        10000, 15000, 20000, 25000, 30000, 40000) <= total], burn, burn + 1L, total)))
}

run_trajectory_fit <- function(s, data, method, rep_id, out, toy = FALSE,
                               total = 50000L, burn = 20000L, cores = 16L) {
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  if (file.exists(file.path(out, "run_config.rds"))) {
    stop("Fit output exists; use a new run directory or evaluate the saved checkpoints.")
  }
  if (!method %in% c("smc", "rjmcmc") || total <= burn || burn < 0 ||
      rep_id < 1L || rep_id > 10L || cores < 1L) stop("Invalid protocol.")
  if (!toy && (nrow(data$X) != 5000L || nrow(data$X_test) != 2000L ||
               total != 50000L || burn != 20000L)) stop("Invalid formal data/settings.")
  s$load_core()
  cfg <- s$config(data$X, cores)
  if (toy) {
    cfg$min_leaf_size <- 15L
    cfg$m_vecchia <- 15L
  }
  seed <- 940000L + 1000L * ncol(data$X) + rep_id
  run <- list(config = cfg, method = method, rep = rep_id, dimension = ncol(data$X),
    function_name = data$function_name, experiment = data$experiment,
    toy = toy, n_particles = 5000L,
    total = total, burn = burn, seed_fit = seed, seed_predict = seed + 100000L,
    n_draws = 1000L, prediction = "pointwise_latent", n_train = nrow(data$X),
    n_test = nrow(data$X_test), cores = cores, data_hash = digest::digest(data, "sha256"),
    source_md5 = tools::md5sum(file.path(s$bundle, c("tvgp/tvgp_core.cpp",
      "tvgp/tvgp_smc.R", "sampler_comparison/samplers.R", "sampler_comparison/rjmcmc_kernel.R"))))
  atomic_rds(run, file.path(out, "run_config.rds"))
  dir.create(file.path(out, "checkpoints"), showWarnings = FALSE)
  write_status(out, "fit", "RUNNING")
  io_seconds <- 0
  checkpoint <- function(step, elapsed, populations) {
    cp <- list(step = step, fit_wall_s = elapsed, checkpoint_io_s_before = io_seconds,
      fit_compute_s = elapsed - io_seconds, populations = populations)
    t0 <- proc.time()[["elapsed"]]
    atomic_rds(cp, file.path(out, "checkpoints", sprintf("step_%06d.rds", step)))
    io_seconds <<- io_seconds + proc.time()[["elapsed"]] - t0
  }
  tryCatch({
    set.seed(seed)
    started <- proc.time()[["elapsed"]]
    target <- s$new_target(data$y, data$X, cfg)
    target$checkpoint_dir <- out
    # A common root start makes the first available predictor explicit.
    initial <- s$SVR_make_root_tree(data$X)
    result <- if (method == "smc") s$run_smc(target, 5000L, started,
      checkpoint = checkpoint) else s$run_rjmcmc(target, initial, total, burn, started,
      checkpoint = checkpoint, checkpoint_iterations = checkpoint_iterations(total, burn))
    elapsed <- proc.time()[["elapsed"]] - started
    atomic_rds(result, file.path(out, "sampler_result.rds"))
    write.csv(result$trace, file.path(out, "trace.csv"), row.names = FALSE)
    write.csv(result$diagnostics, file.path(out, "diagnostics.csv"), row.names = FALSE)
    if (!is.null(result$move_diagnostics)) write.csv(result$move_diagnostics,
      file.path(out, "move_diagnostics.csv"), row.names = FALSE)
    write.csv(data.frame(method = method, dimension = ncol(data$X), rep = rep_id,
      n_particles = 5000, cores = cores, fit_wall_s = elapsed,
      checkpoint_io_s = io_seconds, leaf_evaluations = target$n_eval,
      cache_hits = target$n_hit, peak_cache_entries = target$peak_entries,
      unique_final_trees = nrow(result$tree_counts)),
      file.path(out, "fit_summary.csv"), row.names = FALSE)
    write_status(out, "fit", "COMPLETE")
  }, error = function(e) {
    atomic_rds(list(message = conditionMessage(e), class = class(e),
      subset_key = e$subset_key, indices = e$indices, attempts = e$attempts),
      file.path(out, "fit_failure.rds"))
    write_status(out, "fit", "FAILED", conditionMessage(e))
    stop(e)
  })
}

empirical_crps <- function(draws, truth) {
  draws <- as.matrix(draws)
  if (nrow(draws) != length(truth) || any(!is.finite(draws)) || ncol(draws) < 2L)
    stop("Invalid CRPS inputs.")
  m <- ncol(draws)
  second <- apply(draws, 1L, function(x) sum((2 * seq_len(m) - m - 1) * sort(x)) / m^2)
  rowMeans(abs(draws - truth)) - second
}

new_prediction_cache <- function(s, target, Xt, out) {
  cache <- new.env(parent = emptyenv())
  cache$tree_dir <- file.path(out, "prediction_cache", "trees")
  cache$leaf_dir <- file.path(out, "prediction_cache", "leaves")
  dir.create(cache$tree_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cache$leaf_dir, recursive = TRUE, showWarnings = FALSE)
  cache$moments <- function(compact, expected_logML) {
    key <- digest::digest(compact, "sha256")
    file <- file.path(cache$tree_dir, paste0(key, ".rds"))
    if (file.exists(file)) return(readRDS(file))
    tree <- s$expand_tree(compact, target)
    leaves <- list()
    visit <- function(node, test_idx) {
      if (s$SVR_is_leaf(node)) {
        leaves[[length(leaves) + 1L]] <<- list(node = node, test_idx = test_idx)
      } else {
        left <- Xt[test_idx, node$split_dim] <= node$split_val
        visit(node$left, test_idx[left])
        visit(node$right, test_idx[!left])
      }
    }
    visit(tree$root, seq_len(nrow(Xt)))
    pieces <- s$TVGP_parallel_lapply(leaves, function(z) {
      tryCatch({
        k <- digest::digest(list(sort(z$node$indices), z$test_idx), "sha256")
        lf <- file.path(cache$leaf_dir, paste0(k, ".rds"))
        if (file.exists(lf)) return(readRDS(lf))
        fit <- s$strict_fit(z$node, target)
        p <- if (length(z$test_idx)) s$vecchia_prediction_moments(
          target$y[fit$ord_indices], target$X[fit$ord_indices, , drop = FALSE],
          Xt[z$test_idx, , drop = FALSE], fit$theta_hat, fit$gamma_mean,
          fit$gamma_var_factor, fit$sigma2_shape, fit$sigma2_scale,
          target$config$nu, target$config$m_vecchia, predict_obs = FALSE) else NULL
        z <- list(index = z$test_idx, moments = p, logML = fit$log_evidence)
        atomic_rds(z, lf)
        z
      }, error = function(e) list(error = conditionMessage(e)))
    }, cores = target$config$parallel_cores, mc_preschedule = TRUE)
    if (any(vapply(pieces, function(z) inherits(z, "try-error") ||
      !is.list(z) || !is.null(z$error), logical(1))))
      stop("Prediction leaf failed: ", paste(vapply(pieces, function(z)
        if (is.list(z) && !is.null(z$error)) z$error else "", ""), collapse = "; "))
    actual <- sum(vapply(pieces, `[[`, numeric(1), "logML"))
    if (abs(actual - expected_logML) > 1e-8 * max(1, abs(expected_logML)))
      stop("Prediction refit changed a retained tree score.")
    p <- list(mu = numeric(nrow(Xt)), scale2 = numeric(nrow(Xt)), df = numeric(nrow(Xt)))
    for (z in pieces) if (length(z$index)) for (nm in names(p))
      p[[nm]][z$index] <- z$moments[[nm]]
    if (any(!is.finite(unlist(p))) || any(p$scale2 <= 0) || any(p$df <= 2))
      stop("Invalid Student-t predictive moments.")
    atomic_rds(p, file)
    p
  }
  cache
}

score_population <- function(population, cache, truth, n_draws, seed) {
  weights <- population$tree_counts$weight
  stopifnot(all(weights >= 0), abs(sum(weights) - 1) < 1e-10)
  n <- length(truth)
  set.seed(seed)
  # Common uniforms reduce Monte Carlo differences across checkpoints.
  cumulative <- c(0, cumsum(weights))
  cumulative[length(cumulative)] <- 1
  components <- findInterval(runif(n * n_draws), cumulative, rightmost.closed = TRUE)
  entries <- split(seq_along(components), factor(components, levels = seq_along(weights)))
  rm(components)
  u_t <- matrix(runif(n * n_draws), n)
  draws <- matrix(NA_real_, n, n_draws)
  pred_mean <- numeric(n)
  for (k in seq_along(weights)) {
    p <- cache$moments(population$tree_dict[[k]], population$tree_counts$logML[k])
    pred_mean <- pred_mean + weights[k] * p$mu
    idx <- entries[[k]]
    rows <- (idx - 1L) %% n + 1L
    if (length(idx)) draws[idx] <- p$mu[rows] +
      sqrt(p$scale2[rows]) * qt(u_t[idx], df = p$df[rows])
  }
  crps <- empirical_crps(draws, truth)
  half <- n_draws %/% 2L
  crps1 <- mean(empirical_crps(draws[, seq_len(half), drop = FALSE], truth))
  crps2 <- mean(empirical_crps(draws[, seq.int(half + 1L, n_draws), drop = FALSE], truth))
  ci <- t(apply(draws, 1L, quantile, probs = c(.025, .975), names = FALSE))
  list(summary = c(RMSE = sqrt(mean((pred_mean - truth)^2)), CRPS = mean(crps),
    Cover95 = mean(truth >= ci[, 1] & truth <= ci[, 2]),
    MeanCI95 = mean(ci[, 2] - ci[, 1]), CRPS_half_difference = abs(crps1 - crps2)),
    mean = pred_mean, crps = crps, interval = ci)
}

evaluate_trajectory <- function(s, data, out) {
  run <- readRDS(file.path(out, "run_config.rds"))
  stopifnot(identical(run$data_hash, digest::digest(data, "sha256")))
  current_md5 <- tools::md5sum(file.path(s$bundle, c("tvgp/tvgp_core.cpp",
    "tvgp/tvgp_smc.R", "sampler_comparison/samplers.R", "sampler_comparison/rjmcmc_kernel.R")))
  if (!identical(unname(run$source_md5), unname(current_md5)))
    stop("Sampler/core source hashes differ from the saved fit.")
  files <- sort(list.files(file.path(out, "checkpoints"), "\\.rds$", full.names = TRUE))
  if (!length(files)) stop("No checkpoints available.")
  write_status(out, "prediction", "RUNNING")
  dir.create(file.path(out, "checkpoint_metrics"), showWarnings = FALSE)
  tryCatch({
    target <- s$new_target(data$y, data$X, run$config)
    cache <- new_prediction_cache(s, target, data$X_test, out)
    rows <- list()
    for (f in files) {
      cp <- readRDS(f)
      for (type in names(cp$populations)) {
        name <- sprintf("%s_%06d", type, cp$step)
        saved <- file.path(out, "checkpoint_metrics", paste0(name, ".rds"))
        if (file.exists(saved)) {
          row <- readRDS(saved)$row
        } else {
          started <- proc.time()[["elapsed"]]
          z <- score_population(cp$populations[[type]], cache, data$y_test,
                                run$n_draws, run$seed_predict)
          row <- data.frame(experiment = run$experiment, method = run$method, dimension = run$dimension,
            rep = run$rep, estimator = type, step = cp$step,
            fit_wall_s = cp$fit_wall_s, fit_compute_s = cp$fit_compute_s,
            offline_prediction_s = proc.time()[["elapsed"]] - started,
            unique_trees = length(cp$populations[[type]]$tree_dict), t(z$summary),
            n_train = run$n_train, n_test = run$n_test, n_particles = run$n_particles,
            seed_fit = run$seed_fit, seed_predict = run$seed_predict, toy = run$toy)
          atomic_rds(list(row = row, mean = z$mean, crps = z$crps, interval = z$interval), saved)
        }
        rows[[length(rows) + 1L]] <- row
        write.csv(do.call(rbind, rows), file.path(out, "metrics.csv"), row.names = FALSE)
        cat(sprintf("%s %s step %d: RMSE %.5f CRPS %.5f at %.2fs\n",
          run$method, type, cp$step, row$RMSE, row$CRPS, row$fit_wall_s))
        flush.console()
      }
    }
    write_status(out, "prediction", "COMPLETE")
  }, error = function(e) {
    write_status(out, "prediction", "FAILED", conditionMessage(e))
    stop(e)
  })
}
