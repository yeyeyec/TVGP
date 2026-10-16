# Full-covariance (dense) leaf likelihood for the ablation of Section 4.2. install_dense_backend()
# replaces the Vecchia leaf likelihood of tvgp_smc.R by the dense one; the returned function
# restores it. Source tvgp_smc.R in the global environment first.
.dense_backend_core_file <- local({
  files <- vapply(sys.frames(), function(f) if (is.null(f$ofile)) "" else f$ofile, "")
  files <- files[basename(files) == "dense_backend.R"]
  if (!length(files)) stop("Source tvgp/dense_backend.R with source().")
  file.path(dirname(normalizePath(tail(files, 1L), mustWork = TRUE)), "tvgp_dense.cpp")
})

.dense_backend_exports <- new.env(parent = globalenv())

.load_dense_backend <- function(core_file = .dense_backend_core_file, rebuild = FALSE) {
  core_file <- normalizePath(core_file, mustWork = TRUE)
  if (isTRUE(rebuild) || !identical(.dense_backend_exports$core_file, core_file)) {
    Rcpp::sourceCpp(core_file, env = .dense_backend_exports, rebuild = rebuild)
    .dense_backend_exports$core_file <- core_file
  }
  .dense_backend_exports
}

install_dense_backend <- function(core_file = .dense_backend_core_file, rebuild = FALSE) {
  names <- c("vecchia_nll_grad_counts", "vecchia_nll_grad_hessian_counts")
  if (!all(vapply(names, exists, logical(1), envir = globalenv(), inherits = FALSE))) {
    stop("Source tvgp/tvgp_smc.R in the global environment before installing the dense backend.")
  }
  original <- mget(names, envir = globalenv(), inherits = FALSE)
  if (!all(vapply(original, is.function, logical(1)))) stop("Core fitting wrappers must be functions.")
  backend <- .load_dense_backend(core_file, rebuild)
  wrappers <- list(
    function(theta, y, X, nn_mat, nn_count, nu, m, cov_prior_code,
             prior_params, leaf_prior_params, calc_grad) {
      backend$dense_nig_nll_grad(theta, y, X, nu, cov_prior_code,
                                 prior_params, leaf_prior_params, calc_grad)
    },
    function(theta, y, X, nn_mat, nn_count, nu, m, cov_prior_code,
             prior_params, leaf_prior_params) {
      backend$dense_nig_nll_grad_hessian(theta, y, X, nu, cov_prior_code,
                                         prior_params, leaf_prior_params)
    }
  )
  names(wrappers) <- names
  for (i in seq_along(names)) {
    if (!identical(names(formals(wrappers[[i]])), names(formals(original[[i]])))) {
      stop("The core fitting API changed; refusing to install a mismatched dense wrapper.")
    }
    formals(wrappers[[i]]) <- formals(original[[i]])
  }
  list2env(wrappers, envir = globalenv())
  restored <- FALSE
  restore <- function() {
    if (restored) return(invisible(FALSE))
    current <- mget(names, envir = globalenv(), inherits = FALSE)
    if (!all(vapply(seq_along(names), function(i) identical(current[[i]], wrappers[[i]]), logical(1)))) {
      stop("Dense hooks changed since installation; restore nested installations in reverse order.")
    }
    list2env(original, envir = globalenv())
    restored <<- TRUE
    invisible(TRUE)
  }
  invisible(restore)
}

predict_dense_population <- function(fit, X, y, Xt, settings = list()) {
  if (!is.list(settings)) stop("settings must be a list.")
  get_setting <- function(name, default) {
    if (is.null(settings[[name]])) default else settings[[name]]
  }
  if (!identical(get_setting("prediction_mode", "pointwise"), "pointwise") ||
      isTRUE(get_setting("predict_obs", FALSE)) || isTRUE(get_setting("return_covariance", FALSE))) {
    stop("The DENSE ablation provides pointwise latent Student-t prediction only.")
  }
  integer_setting <- function(name, default, lower, upper) {
    value <- get_setting(name, default)
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
        value != floor(value) || value < lower || value > upper) {
      stop(name, " must be an integer in [", lower, ", ", upper, "].")
    }
    as.integer(value)
  }
  chunk_size <- integer_setting("dense_chunk_size", 128L, 1L, 2000L)
  n_draws <- integer_setting("n_draws", 0L, 0L, .Machine$integer.max)
  nu <- get_setting("nu", 1.5)
  X <- as.matrix(X)
  Xt <- as.matrix(Xt)
  y <- as.numeric(y)
  if (nrow(X) != length(y) || ncol(X) != ncol(Xt) ||
      any(!is.finite(X)) || any(!is.finite(Xt)) || any(!is.finite(y))) {
    stop("Dense prediction requires conformable finite X, y, and Xt.")
  }
  if (ncol(X) < 1L || ncol(X) > 2L) stop("DENSE ablation supports one or two dimensions.")
  particles <- fit$final_particles
  weights <- tail(fit$history_weights, 1L)[[1L]]
  if (!length(particles) || length(weights) != length(particles) ||
      any(!is.finite(weights)) || any(weights < 0) || sum(weights) <= 0) {
    stop("fit must contain final_particles and valid final history_weights.")
  }
  weights <- as.numeric(weights) / sum(weights)
  # Check every leaf before any factorization, including leaves with no test points.
  for (p in particles) {
    for (leaf in p$nodes) {
      if (!isTRUE(leaf$is_leaf)) next
      idx <- leaf$ord_indices
      if (length(idx) < 1L || length(idx) > 2000L || anyNA(idx) ||
          any(idx != as.integer(idx)) || any(idx < 1L | idx > nrow(X)) || anyDuplicated(idx)) {
        stop("Every dense leaf must contain 1 to 2000 distinct valid ord_indices.")
      }
    }
  }
  backend <- .load_dense_backend()
  n_test <- nrow(Xt)
  active <- which(weights > 0)
  # Exact duplicate particles are common after resampling. Do not merge rounded
  # hyperparameters or trees that merely share a topology.
  reps <- integer()
  group_weights <- numeric()
  for (i in active) {
    same <- if (isTRUE(get_setting("cache_identical_trees", TRUE))) {
      which(vapply(reps, function(j) identical(particles[[i]], particles[[j]]), logical(1)))
    } else integer()
    if (length(same)) {
      group_weights[same[1L]] <- group_weights[same[1L]] + weights[i]
    } else {
      reps <- c(reps, i)
      group_weights <- c(group_weights, weights[i])
    }
  }
  draws <- NULL
  draw_components <- NULL
  if (n_draws > 0L) {
    if (!is.null(settings$draw_seed)) set.seed(settings$draw_seed)
    draws <- matrix(NA_real_, nrow = n_test, ncol = n_draws)
    draw_components <- matrix(sample.int(length(reps), n_test * n_draws,
                                         replace = TRUE, prob = group_weights),
                               nrow = n_test, ncol = n_draws)
  }
  mean <- within <- between <- numeric(n_test)
  mass <- 0
  for (component in seq_along(reps)) {
    p <- particles[[reps[component]]]
    leaf_ids <- vapply(seq_len(n_test), function(k) find_leaf_id(Xt[k, ], p$nodes, 1L), integer(1))
    mu <- variance <- numeric(n_test)
    for (lid in unique(leaf_ids)) {
      test_idx <- which(leaf_ids == lid)
      leaf <- p$nodes[[lid]]
      train_idx <- leaf$ord_indices
      prediction <- backend$dense_nig_prediction_moments(
        y[train_idx], X[train_idx, , drop = FALSE], Xt[test_idx, , drop = FALSE],
        leaf$theta_hat, leaf$gamma_mean, leaf$gamma_var_factor,
        leaf$sigma2_shape, leaf$sigma2_scale, nu, chunk_size
      )
      mu[test_idx] <- prediction$mu
      variance[test_idx] <- prediction$variance
      if (n_draws > 0L) {
        selected <- which(draw_components[test_idx, , drop = FALSE] == component, arr.ind = TRUE)
        if (nrow(selected)) {
          local_rows <- selected[, 1L]
          draws[cbind(test_idx[local_rows], selected[, 2L])] <-
            prediction$mu[local_rows] + sqrt(prediction$scale2[local_rows]) *
            stats::rt(nrow(selected), df = prediction$df[local_rows])
        }
      }
    }
    weight <- group_weights[component]
    next_mass <- mass + weight
    difference <- mu - mean
    between <- between + weight * mass / next_mass * difference^2
    mean <- mean + weight / next_mass * difference
    within <- within + weight * variance
    mass <- next_mass
  }
  list(mean = mean, variance = within + between, draws = draws,
       n_prediction_trees = length(reps), n_particles = length(particles),
       predictive_distribution = "pointwise posterior mixture of univariate Student-t distributions",
       prediction_target = "latent_function", prediction_mode = "pointwise",
       likelihood_backend = "dense_proper_nig", dense_chunk_size = chunk_size)
}
