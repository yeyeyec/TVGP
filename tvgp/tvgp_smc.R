library(Rcpp)
library(RcppArmadillo)
loadNamespace("RcppParallel")

# Compile the C++ core that sits beside this file.
TVGP_source_dir <- local({
  files <- vapply(sys.frames(), function(f) if (is.null(f$ofile)) "" else f$ofile, "")
  files <- files[basename(files) == "tvgp_smc.R"]
  if (length(files)) dirname(normalizePath(tail(files, 1L), mustWork = TRUE)) else getwd()
})
TVGP_core_file <- file.path(TVGP_source_dir, "tvgp_core.cpp")
if (!file.exists(TVGP_core_file)) {
  TVGP_core_file <- file.path(getwd(), "tvgp_core.cpp")
}
if (!file.exists(TVGP_core_file)) {
  stop("tvgp_core.cpp was not found beside tvgp_smc.R or in the working directory.")
}
TVGP_cpp_cache <- Sys.getenv("TVGP_CPP_CACHE", unset = file.path(tempdir(), "tvgp_cpp"))
dir.create(TVGP_cpp_cache, recursive = TRUE, showWarnings = FALSE)
sourceCpp(normalizePath(TVGP_core_file, mustWork = TRUE),
          cacheDir = TVGP_cpp_cache, env = environment())

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
TVGP_logsumexp <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(-Inf)
  m <- max(x)
  m + log(sum(exp(x - m)))
}

TVGP_normalize_log_weights <- function(log_w) {
  log_w[!is.finite(log_w)] <- -1e100
  z <- TVGP_logsumexp(log_w)
  if (!is.finite(z)) return(rep(1 / length(log_w), length(log_w)))
  exp(log_w - z)
}

TVGP_parallel_lapply <- function(X, FUN, ..., cores = 1L, mc_preschedule = FALSE) {
  # Backend "fork" (default) forks over particles; backend "threads" leaves the
  # cores to the C++ row loops (TVGP_set_compute_threads) and runs serially here.
  cores <- as.integer(cores)
  if (identical(TVGP_parallel_backend(), "threads") || is.na(cores) || cores < 2L ||
      length(X) <= 1L || .Platform$OS.type == "windows") {
    return(lapply(X, FUN, ...))
  }
  parallel::mclapply(
    X, FUN, ...,
    mc.cores = min(cores, length(X)),
    mc.preschedule = mc_preschedule,
    mc.set.seed = FALSE
  )
}

TVGP_parallel_backend <- function() {
  backend <- getOption("tvgp.parallel_backend", "fork")
  if (!backend %in% c("fork", "threads")) stop("tvgp.parallel_backend must be 'fork' or 'threads'.")
  backend
}

TVGP_attach_fit_cache <- function(ctx, y, X, directory, max_bytes = 1024^3) {
  stopifnot(identical(ctx$ordering, "global"), is.finite(max_bytes), max_bytes > 0)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  domain <- digest::digest(list(y, X, ctx[setdiff(names(ctx), "fit_cache")],
    unname(tools::md5sum(c(TVGP_core_file, file.path(TVGP_source_dir,
      "tvgp_smc.R"))))), algo = "sha256")
  directory <- file.path(directory, domain)
  dir.create(directory, showWarnings = FALSE)
  cache <- new.env(parent = emptyenv())
  cache$directory <- directory
  cache$data_hash <- digest::digest(list(y, X), algo = "sha256")
  cache$hits <- cache$misses <- cache$writes <- cache$skipped_writes <- 0L
  cache$bytes <- sum(file.info(list.files(directory, full.names = TRUE))$size)
  cache$max_bytes <- max_bytes
  ctx$fit_cache <- cache
  ctx
}

TVGP_cache_stats <- function(ctx) {
  z <- ctx$fit_cache
  if (is.null(z)) return(data.frame(hits = 0L, misses = 0L, writes = 0L, bytes = 0))
  data.frame(hits = z$hits, misses = z$misses, writes = z$writes,
    skipped_writes = z$skipped_writes, bytes = z$bytes)
}

TVGP_cache_read <- function(cache, identity) {
  if (is.null(cache)) return(NULL)
  file <- file.path(cache$directory, paste0(digest::digest(identity, algo = "sha256"), ".rds"))
  entry <- if (file.exists(file)) tryCatch(readRDS(file), error = function(e) NULL) else NULL
  if (!is.null(entry) && identical(entry$identity, identity) &&
      isFALSE(entry$fit$optimization_failed) &&
      is.finite(entry$fit$log_evidence)) {
    cache$hits <- cache$hits + 1L
    return(entry$fit)
  }
  cache$misses <- cache$misses + 1L
  NULL
}

TVGP_cache_write <- function(cache, identity, fit) {
  if (is.null(cache) || isTRUE(fit$optimization_failed)) return(invisible(NULL))
  file <- file.path(cache$directory, paste0(digest::digest(identity, algo = "sha256"), ".rds"))
  if (file.exists(file)) return(invisible(NULL))
  tmp <- tempfile("fit_", tmpdir = cache$directory)
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(list(identity = identity, fit = fit), tmp, compress = FALSE)
  bytes <- file.info(tmp)$size
  if (cache$bytes + bytes > cache$max_bytes) {
    cache$skipped_writes <- cache$skipped_writes + 1L
    return(invisible(NULL))
  }
  if (!file.rename(tmp, file)) stop("Cannot finalize fitted-node cache entry.")
  cache$bytes <- cache$bytes + bytes
  cache$writes <- cache$writes + 1L
  invisible(NULL)
}

TVGP_get_safe_init_theta <- function(X, range_frac = 0.2, nugget = 0.05) {
  d_ranges <- apply(X, 2, function(x) diff(range(x)))
  d_ranges[!is.finite(d_ranges) | d_ranges <= 0] <- 1
  c(log(pmax(d_ranges * range_frac, 1e-8)), log(nugget))
}

TVGP_mean_pairwise_abs_distance <- function(x) {
  x <- sort(as.numeric(x))
  n <- length(x)
  if (n <= 1L) return(0)
  2 * sum((2 * seq_len(n) - n - 1) * x) / (n * (n - 1))
}

TVGP_jr_log_normalizer <- local({
  cache <- new.env(parent = emptyenv())
  function(a, b, C, g_min) {
    d <- length(C)
    if (any(!is.finite(C)) || any(C <= 0) || !is.finite(g_min) || g_min <= 0)
      stop("JR requires positive coordinate scales and a fixed positive g_min.")
    key <- paste(d, sprintf("%.17g", c(a, b, g_min)), collapse = ":")
    if (!exists(key, cache, inherits = FALSE)) {
      c <- b * g_min
      mode <- ((d + a - c) + sqrt((d + a - c)^2 + 4 * d * c)) / 2
      log_kernel <- function(x) d * log(x) + a * log(x + c) - x
      peak <- log_kernel(mode)
      integral <- integrate(function(x) exp(log_kernel(x) - peak),
                            0, Inf, rel.tol = 1e-11, subdivisions = 1000L)
      if (!identical(integral$message, "OK") || integral$value <= 0)
        stop("Truncated JR normalizing integral did not converge.")
      value <- lgamma(d + 1) + (a + d + 1) * log(b) + c -
        log(integral$value) - peak
      assign(key, value, cache)
    }
    get(key, cache, inherits = FALSE) + sum(log(C))
  }
})

TVGP_to_log_parameters <- function(psi, g_min = 0) {
  if (g_min == 0) return(psi)
  q <- length(psi)
  psi[q] <- TVGP_logsumexp(c(log(g_min), psi[q]))
  psi
}

TVGP_shifted_nugget_derivatives <- function(result, psi, g_min) {
  if (g_min == 0 || !isTRUE(result$valid)) return(result)
  q <- length(psi)
  log_g <- TVGP_to_log_parameters(psi, g_min)[q]
  s <- exp(psi[q] - log_g)
  old_gradient <- result$grad[q]
  result$nll <- result$nll - (psi[q] - log_g)
  result$grad[q] <- s * (old_gradient + 1) - 1
  if (!is.null(result$hessian)) {
    result$hessian[q, ] <- result$hessian[q, ] * s
    result$hessian[, q] <- result$hessian[, q] * s
    result$hessian[q, q] <- result$hessian[q, q] +
      s * (1 - s) * (old_gradient + 1)
  }
  result
}

TVGP_cov_prior_spec <- function(
    X,
    cov_prior = c("inverse_gamma", "jointly_robust"),
    cov_prior_control = NULL) {
  cov_prior <- match.arg(cov_prior)
  X <- as.matrix(X)
  n <- nrow(X)
  d <- ncol(X)
  if (n < 2L || d < 1L || any(!is.finite(X))) {
    stop("A covariance prior requires a finite design with at least two rows.")
  }
  if (is.null(cov_prior_control)) cov_prior_control <- list()
  if (!is.list(cov_prior_control)) stop("cov_prior_control must be a list or NULL.")
  allowed <- c(
    "a_phi", "b_phi", "a_g", "b_g",
    "jr_a", "jr_b", "jr_C", "jr_C_method", "jr_scale_floor",
    "jr_definition", "g_min"
  )
  unknown <- setdiff(names(cov_prior_control), allowed)
  if (length(unknown)) {
    stop("Unknown cov_prior_control field(s): ", paste(unknown, collapse = ", "))
  }

  get_control <- function(name, default) {
    value <- cov_prior_control[[name]]
    if (is.null(value)) default else value
  }

  if (cov_prior == "inverse_gamma") {
    raw_values <- list(
      a_phi = get_control("a_phi", 2.5),
      b_phi = get_control("b_phi", 1.0),
      a_g = get_control("a_g", 2.0),
      b_g = get_control("b_g", 0.1)
    )
    if (any(lengths(raw_values) != 1L)) {
      stop("Inverse-gamma covariance-prior parameters must be positive scalars.")
    }
    values <- vapply(raw_values, as.numeric, numeric(1))
    if (any(!is.finite(values)) || any(values <= 0)) {
      stop("Inverse-gamma covariance-prior parameters must be positive scalars.")
    }
    log_normalizer <- d *
      (values[["a_phi"]] * log(values[["b_phi"]]) - lgamma(values[["a_phi"]])) +
      values[["a_g"]] * log(values[["b_g"]]) - lgamma(values[["a_g"]])
    return(list(
      name = cov_prior,
      code = 0L,
      params = unname(values),
      log_normalizer = as.numeric(log_normalizer),
      details = c(list(n = n, d = d), as.list(values))
    ))
  }

  definition <- match.arg(
    as.character(get_control("jr_definition", "gu_2019")),
    c("gu_2019", "legacy_20260815")
  )
  a <- as.numeric(get_control("jr_a", 0.2))
  g_min <- as.numeric(get_control("g_min", 1e-6))
  if (length(g_min) != 1L || !is.finite(g_min) || g_min <= 0)
    stop("g_min must be one fixed positive value.")
  if (definition != "gu_2019") stop("Truncated JR uses the Gu (2019) definition.")
  b_default <- if (definition == "gu_2019") 1 else n^(-1 / d) * (a + d)
  b <- as.numeric(get_control("jr_b", b_default))
  C_method_default <- if (definition == "gu_2019") "range_scaled" else "mean_pairwise"
  C_method <- match.arg(
    as.character(get_control("jr_C_method", C_method_default)),
    c("mean_pairwise", "range_scaled")
  )
  scale_floor <- as.numeric(get_control("jr_scale_floor", 1e-12))
  C_override <- cov_prior_control[["jr_C"]]
  if (is.null(C_override)) {
    C <- if (C_method == "mean_pairwise") {
      apply(X, 2L, TVGP_mean_pairwise_abs_distance)
    } else {
      apply(X, 2L, function(z) diff(range(z))) / n^(1 / d)
    }
  } else {
    C <- as.numeric(C_override)
    C_method <- "user_supplied"
  }
  lower_a <- if (definition == "gu_2019") -(d + 1) else -d
  if (length(a) != 1L || !is.finite(a) || a <= lower_a) {
    stop("jr_a must be finite and greater than ", lower_a, ".")
  }
  if (length(b) != 1L || !is.finite(b) || b <= 0) {
    stop("jr_b must be a positive finite scalar.")
  }
  if (length(scale_floor) != 1L || !is.finite(scale_floor) || scale_floor <= 0) {
    stop("jr_scale_floor must be a positive finite scalar.")
  }
  if (length(C) != d || any(!is.finite(C)) || any(C < 0)) {
    stop("jr_C must contain one finite nonnegative scale for each input.")
  }
  coordinate_ranges <- apply(X, 2L, function(z) diff(range(z)))
  if (any(coordinate_ranges <= 0) || any(C <= 0))
    stop("Every modeled coordinate must have positive span in each candidate leaf.")
  adjusted <- rep(FALSE, d)

  if (definition == "gu_2019") {
    # Gu (2019), Eq. (3.2): (t + eta)^a exp[-b(t + eta)].
    log_normalizer <- TVGP_jr_log_normalizer(a, b, C, g_min)
    prior_code <- 1L
  } else {
    # Retained only to reproduce runs made before the Gu (2019) formula was
    # corrected in this codebase.
    log_normalizer <- lgamma(d) + (a + d + 1) * log(b) +
      sum(log(C)) - lgamma(a + d)
    prior_code <- 2L
  }
  list(
    name = cov_prior,
    code = prior_code,
    params = c(a, b, C),
    log_normalizer = as.numeric(log_normalizer),
    details = list(
      n = n, d = d, a = a, b = b, C = C, C_method = C_method,
      definition = definition, g_min = g_min,
      integration_coordinates = "log_phi_log_g_minus_g_min",
      C_floor_adjusted = which(adjusted), scale_floor = scale_floor
    )
  )
}

TVGP_leaf_prior_spec <- function(leaf_prior_control = NULL) {
  if (is.null(leaf_prior_control)) leaf_prior_control <- list()
  if (!is.list(leaf_prior_control)) {
    stop("leaf_prior_control must be a list or NULL.")
  }
  allowed <- c("m0", "V0", "a0", "b0")
  unknown <- setdiff(names(leaf_prior_control), allowed)
  if (length(unknown)) {
    stop("Unknown leaf_prior_control field(s): ", paste(unknown, collapse = ", "))
  }
  get_control <- function(name, default) {
    value <- leaf_prior_control[[name]]
    if (is.null(value)) default else value
  }
  raw_values <- list(
    m0 = get_control("m0", 0),
    V0 = get_control("V0", 100),
    a0 = get_control("a0", 2),
    b0 = get_control("b0", 1)
  )
  if (any(lengths(raw_values) != 1L)) {
    stop("The proper NIG leaf-prior parameters must be scalars.")
  }
  values <- vapply(raw_values, as.numeric, numeric(1))
  if (any(!is.finite(values)) ||
      values[["V0"]] <= 0 || values[["a0"]] <= 0 || values[["b0"]] <= 0) {
    stop("The proper NIG leaf prior requires finite m0 and positive V0, a0, and b0.")
  }
  list(
    name = "normal_inverse_gamma",
    params = unname(values),
    details = as.list(values)
  )
}

TVGP_log_iid_evidence <- function(y, leaf_prior_spec) {
  n <- length(y)
  if (n < 1L || any(!is.finite(y))) return(-Inf)
  pars <- leaf_prior_spec$params
  m0 <- pars[1L]
  P0 <- 1 / pars[2L]
  a0 <- pars[3L]
  b0 <- pars[4L]
  M <- P0 + n
  h <- P0 * m0 + sum(y)
  S0 <- sum(y * y) + P0 * m0^2 - h^2 / M
  tol <- 1e-10 * max(1, abs(sum(y * y)), abs(P0 * m0^2), abs(h^2 / M))
  if (!is.finite(S0) || S0 < -tol) return(-Inf)
  S0 <- max(S0, 0)
  a_n <- a0 + n / 2
  b_n <- b0 + S0 / 2
  -(n / 2) * log(2 * pi) + 0.5 * log(P0) - 0.5 * log(M) +
    a0 * log(b0) - lgamma(a0) + lgamma(a_n) - a_n * log(b_n)
}

TVGP_make_node <- function(id, indices, fit, depth, key, terminal = FALSE,
                           terminal_reason = NA_character_) {
  list(
    id = id, is_leaf = TRUE, terminal = terminal, indices = indices,
    ord_indices = fit$ord_indices,
    n_train = if (is.null(fit$n_train)) length(indices) else fit$n_train,
    y_mean = if (is.null(fit$y_mean)) NA_real_ else fit$y_mean,
    y_sd = if (is.null(fit$y_sd)) NA_real_ else fit$y_sd,
    y_zero_frac = if (is.null(fit$y_zero_frac)) NA_real_ else fit$y_zero_frac,
    theta_hat = fit$theta_hat,
    gamma_mean = fit$gamma_mean,
    gamma_var_factor = fit$gamma_var_factor,
    sigma2_shape = fit$sigma2_shape,
    sigma2_scale = fit$sigma2_scale,
    sigma2_mean = fit$sigma2_mean,
    sigma2_hat = fit$sigma2_mean,
    log_evidence = fit$log_evidence, depth = depth,
    split_dim = NA_integer_, split_val = NA_real_,
    left = NA_integer_, right = NA_integer_, key = key,
    terminal_reason = terminal_reason,
    optimization_failed = isTRUE(fit$optimization_failed),
    optimization_failure_reason = fit$optimization_failure_reason,
    optimizer_convergence = fit$convergence,
    optimizer_message = fit$optimizer_message,
    optimization_attempts = if (is.null(fit$optimization_attempts)) 1L else fit$optimization_attempts,
    optimization_attempt_label = if (is.null(fit$optimization_attempt_label)) NA_character_ else fit$optimization_attempt_label,
    previous_failure_reasons = if (is.null(fit$previous_failure_reasons)) NA_character_ else fit$previous_failure_reasons,
    gradient_norm = fit$gradient_norm,
    gradient_inf_norm = fit$gradient_inf_norm,
    minimum_bound_distance = fit$minimum_bound_distance,
    mode_on_boundary = fit$mode_on_boundary,
    initial_optimizer_convergence = fit$initial_optimizer_convergence,
    initial_optimizer_message = fit$initial_optimizer_message,
    newton_converged = fit$newton_converged,
    newton_iterations = fit$newton_iterations,
    newton_accepted_steps = fit$newton_accepted_steps,
    newton_damping_steps = fit$newton_damping_steps,
    newton_line_search_evaluations = fit$newton_line_search_evaluations,
    newton_failure_reason = fit$newton_failure_reason,
    optimizer_gradient_tol = fit$optimizer_gradient_tol,
    optimizer_boundary_tol = fit$optimizer_boundary_tol,
    hessian_status = fit$hessian_status,
    hessian_n_regularized = fit$hessian_n_regularized,
    hessian_min_eigenvalue = fit$hessian_min_eigenvalue,
    hessian_condition_number = fit$hessian_condition_number,
    hessian_near_singular = fit$hessian_near_singular,
    laplace_fallback = isTRUE(fit$laplace_fallback),
    leaf_prior = fit$leaf_prior,
    leaf_prior_details = fit$leaf_prior_details,
    cov_prior = fit$cov_prior,
    cov_prior_details = fit$cov_prior_details
  )
}

TVGP_node_n_train <- function(node) {
  if (is.null(node$n_train)) length(node$indices) else as.integer(node$n_train)
}

TVGP_tree_signature <- function(p, include_params = FALSE, digits = 10L) {
  node_sig <- vapply(p$nodes, function(z) {
    base <- paste(
      z$id, z$key, as.integer(isTRUE(z$is_leaf)), as.integer(isTRUE(z$terminal)),
      ifelse(is.na(z$split_dim), "NA", z$split_dim),
      ifelse(is.na(z$split_val), "NA", signif(z$split_val, digits)),
      ifelse(is.na(z$left), "NA", z$left),
      ifelse(is.na(z$right), "NA", z$right),
      sep = ":"
    )
    if (!isTRUE(include_params)) return(base)
    theta <- paste(signif(as.numeric(z$theta_hat), digits), collapse = ",")
    paste(
      base, theta, signif(z$gamma_mean, digits),
      signif(z$gamma_var_factor, digits), signif(z$sigma2_mean, digits),
      sep = ":"
    )
  }, character(1))
  paste(node_sig, collapse = "|")
}

TVGP_prepare_context <- function(X, m = 30, ordering = c("global", "input", "local"),
                                 nn_pool_factor = 8, exact_nn_fallback = FALSE,
                                 verbose = TRUE) {
  ordering <- match.arg(ordering)
  ctx <- list(
    ordering = ordering,
    exact_nn_fallback = exact_nn_fallback,
    design_hash = digest::digest(X, algo = "sha256"),
    m = m, nn_pool_factor = nn_pool_factor,
    N = nrow(X)
  )

  if (ordering %in% c("global", "input")) {
    pool_m <- min(max(m, ceiling(nn_pool_factor * m)), max(1, nrow(X) - 1))
    if (verbose) {
      label <- if (ordering == "global") "global maximin" else "input"
      cat(sprintf("Preparing %s order and NN pool (pool=%d)...\n", label, pool_m))
    }
    global_order0 <- if (ordering == "global") {
      as.integer(get_maximin_order(X))
    } else {
      as.integer(seq_len(nrow(X)) - 1L)
    }
    X_ord <- X[global_order0 + 1L, , drop = FALSE]
    global_pool_pos <- get_nn_matrix(X_ord, pool_m)
    global_pool_orig0 <- matrix(
      global_order0[as.integer(global_pool_pos) + 1L],
      nrow = nrow(global_pool_pos),
      ncol = ncol(global_pool_pos)
    )
    global_count <- as.integer(pmin(seq_len(nrow(X)) - 1L, pool_m))
    order_rank0 <- integer(nrow(X))
    order_rank0[global_order0 + 1L] <- seq_len(nrow(X)) - 1L

    ctx$global_order0 <- global_order0
    ctx$global_nn_pool_orig0 <- global_pool_orig0
    ctx$global_nn_count <- global_count
    ctx$order_rank0 <- order_rank0
    ctx$order_rank <- order_rank0 + 1L
    ctx$pool_m <- pool_m
  }

  ctx
}

TVGP_prepare_node_data <- function(y, X, indices, m, ctx = NULL) {
  if (!is.null(ctx) && ctx$ordering %in% c("global", "input")) {
    ord_indices <- indices[order(ctx$order_rank[indices])]
    nn <- build_local_nn_from_global_pool(
      X,
      as.integer(ord_indices - 1L),
      ctx$global_nn_pool_orig0,
      as.integer(ctx$global_nn_count),
      as.integer(ctx$order_rank0),
      as.integer(m),
      isTRUE(ctx$exact_nn_fallback)
    )
  } else {
    X_sub <- X[indices, , drop = FALSE]
    ord0 <- as.integer(get_maximin_order(X_sub))
    ord_indices <- indices[ord0 + 1L]
    nn <- list(
      nn_mat = get_nn_matrix(X[ord_indices, , drop = FALSE], m),
      nn_count = as.integer(pmin(seq_along(ord_indices) - 1L, m))
    )
  }

  list(
    y_ord = y[ord_indices],
    X_ord = X[ord_indices, , drop = FALSE],
    nn_mat = nn$nn_mat,
    nn_count = as.integer(nn$nn_count),
    ord_indices = ord_indices
  )
}

TVGP_hessian_logdet <- function(H, relative_eigen_tol = 1e-10) {
  eig <- eigen(H, symmetric = TRUE, only.values = TRUE)$values
  if (!length(eig) || any(!is.finite(eig))) {
    stop("non-finite Hessian eigenvalues")
  }
  if (min(eig) <= 0) {
    stop(sprintf(
      "Hessian is not positive definite (minimum eigenvalue %.6g)",
      min(eig)
    ))
  }
  eig_scale <- max(eig)
  list(
    log_det = sum(log(eig)), min_eigenvalue = min(eig),
    condition_number = eig_scale / min(eig),
    near_singular = min(eig) / max(1, eig_scale) < relative_eigen_tol
  )
}

TVGP_newton_refine <- function(theta, evaluate, lower, upper,
                               gradient_tol = 1e-3,
                               boundary_tol = 1e-6,
                               maxit = 15L,
                               max_backtracking = 24L,
                               trust_radius = 1.5,
                               armijo_constant = 1e-4, evaluate_light = NULL) {
  theta <- as.numeric(theta)
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  q <- length(theta)
  if (length(lower) != q || length(upper) != q || any(lower >= upper)) {
    stop("Invalid bounds supplied to TVGP_newton_refine().")
  }

  safe_evaluate <- function(z) {
    tryCatch(evaluate(z), error = function(e) NULL)
  }
  valid_result <- function(z) {
    !is.null(z) && isTRUE(z$valid) && is.finite(z$nll) &&
      length(z$grad) == q && all(is.finite(z$grad)) &&
      identical(dim(as.matrix(z$hessian)), c(q, q)) &&
      all(is.finite(z$hessian))
  }

  current <- safe_evaluate(theta)
  if (!valid_result(current)) {
    return(list(
      par = theta, result = current, converged = FALSE,
      iterations = 0L, accepted_steps = 0L, damping_steps = 0L,
      line_search_evaluations = 0L,
      failure_reason = "invalid_initial_analytic_evaluation"
    ))
  }

  accepted_steps <- 0L
  damping_steps <- 0L
  line_search_evaluations <- 0L
  failure_reason <- NA_character_

  try_direction <- function(direction, value, gradient, gradient_merit = FALSE) {
    directional_derivative <- sum(gradient * direction)
    if (!is.finite(directional_derivative) || directional_derivative >= 0) {
      return(NULL)
    }

    margin <- max(10 * boundary_tol, 1e-8)
    interior_lower <- lower + margin
    interior_upper <- upper - margin
    alpha_bound <- 1
    positive <- direction > 0
    negative <- direction < 0
    if (any(positive)) {
      alpha_bound <- min(
        alpha_bound,
        min((interior_upper[positive] - theta[positive]) / direction[positive])
      )
    }
    if (any(negative)) {
      alpha_bound <- min(
        alpha_bound,
        min((interior_lower[negative] - theta[negative]) / direction[negative])
      )
    }
    if (!is.finite(alpha_bound) || alpha_bound <= 0) return(NULL)

    alpha <- min(1, 0.995 * alpha_bound)
    for (line_search_step in seq_len(max_backtracking)) {
      candidate_theta <- theta + alpha * direction
      candidate <- if (is.null(evaluate_light)) safe_evaluate(candidate_theta) else
        tryCatch(evaluate_light(candidate_theta), error = function(e) NULL)
      line_search_evaluations <<- line_search_evaluations + 1L
      light_valid <- !is.null(candidate) && isTRUE(candidate$valid) &&
        is.finite(candidate$nll) && length(candidate$grad) == q &&
        all(is.finite(candidate$grad))
      accepted_value <- if (light_valid) {
        if (gradient_merit) {
          sum(candidate$grad^2) <= (1 - 2 * armijo_constant * alpha) * sum(gradient^2)
        } else {
          candidate$nll <= value + armijo_constant * alpha * directional_derivative
        }
      } else FALSE
      if (accepted_value) {
        if (!is.null(evaluate_light)) candidate <- safe_evaluate(candidate_theta)
        if (valid_result(candidate)) {
          return(list(par = candidate_theta, result = candidate))
        }
      }
      alpha <- alpha / 2
    }
    NULL
  }

  iterations_used <- 0L
  for (iteration in seq_len(as.integer(maxit))) {
    gradient <- as.numeric(current$grad)
    if (max(abs(gradient)) <= gradient_tol) break
    iterations_used <- iteration

    H <- 0.5 * (as.matrix(current$hessian) + t(as.matrix(current$hessian)))
    eigenvalues <- tryCatch(
      eigen(H, symmetric = TRUE, only.values = TRUE)$values,
      error = function(e) rep(NA_real_, q)
    )
    finite_eigenvalues <- eigenvalues[is.finite(eigenvalues)]
    curvature_scale <- if (length(finite_eigenvalues)) {
      max(1, max(abs(finite_eigenvalues)))
    } else {
      1
    }
    curvature_floor <- 1e-8 * curvature_scale
    minimum_eigenvalue <- if (length(finite_eigenvalues)) {
      min(finite_eigenvalues)
    } else {
      NA_real_
    }
    damping <- if (is.finite(minimum_eigenvalue)) {
      max(0, curvature_floor - minimum_eigenvalue)
    } else {
      curvature_scale
    }
    if (damping > 0) damping_steps <- damping_steps + 1L

    H_step <- H + diag(damping, q)
    direction <- tryCatch(
      -as.numeric(solve(H_step, gradient)),
      error = function(e) rep(NA_real_, q)
    )
    if (any(!is.finite(direction)) || sum(gradient * direction) >= 0) {
      direction <- -gradient
    }
    direction_size <- max(abs(direction))
    if (is.finite(direction_size) && direction_size > trust_radius) {
      direction <- direction * trust_radius / direction_size
    }

    accepted <- try_direction(direction, current$nll, gradient)
    if (is.null(accepted)) {
      gradient_direction <- -gradient / max(1, max(abs(gradient)))
      accepted <- try_direction(gradient_direction, current$nll, gradient)
    }
    if (is.null(accepted) && is.finite(minimum_eigenvalue) && minimum_eigenvalue > 0) {
      # Near a mode, objective differences can be below floating-point resolution.
      # Backtrack the Newton step on the squared gradient norm instead.
      accepted <- try_direction(direction, current$nll, gradient, gradient_merit = TRUE)
    }
    if (is.null(accepted)) {
      failure_reason <- "armijo_line_search_failed"
      break
    }

    theta <- accepted$par
    current <- accepted$result
    accepted_steps <- accepted_steps + 1L
  }

  gradient_inf_norm <- if (valid_result(current)) {
    max(abs(current$grad))
  } else {
    Inf
  }
  converged <- is.finite(gradient_inf_norm) && gradient_inf_norm <= gradient_tol
  if (!converged && is.na(failure_reason)) {
    failure_reason <- "newton_iteration_limit"
  }

  list(
    par = theta, result = current, converged = converged,
    iterations = iterations_used, accepted_steps = accepted_steps,
    damping_steps = damping_steps,
    line_search_evaluations = line_search_evaluations,
    gradient_inf_norm = gradient_inf_norm,
    failure_reason = failure_reason
  )
}

TVGP_log_evidence <- function(fit_value, theta_hat, analytic_hessian, n,
                              cov_log_prior_const, leaf_prior_spec,
                              hessian = TRUE,
                              hessian_eigen_tol = 1e-10) {
  d_theta <- length(theta_hat)
  leaf_pars <- leaf_prior_spec$params
  P0 <- 1 / leaf_pars[2L]
  a0 <- leaf_pars[3L]
  b0 <- leaf_pars[4L]
  a_n <- a0 + n / 2
  log_data_const <- -(n / 2) * log(2 * pi) + 0.5 * log(P0) +
    a0 * log(b0) - lgamma(a0) + lgamma(a_n)
  if (length(cov_log_prior_const) != 1L || !is.finite(cov_log_prior_const)) {
    stop("cov_log_prior_const must be one finite value.")
  }
  laplace <- 0
  hessian_status <- if (isTRUE(hessian)) "not_evaluated" else "not_requested"
  hessian_n_regularized <- 0L
  laplace_fallback <- FALSE
  hessian_min_eigenvalue <- NA_real_
  hessian_condition_number <- NA_real_
  hessian_near_singular <- NA

  if (isTRUE(hessian)) {
    hessian_error <- NULL
    hessian_result <- tryCatch({
      H <- as.matrix(analytic_hessian)
      if (!identical(dim(H), c(d_theta, d_theta))) {
        stop("analytic Hessian has the wrong dimension")
      }
      if (max(abs(H - t(H))) > 1e-7 * max(1, max(abs(H)))) {
        stop("analytic Hessian is not symmetric")
      }
      TVGP_hessian_logdet(
        0.5 * (H + t(H)), relative_eigen_tol = hessian_eigen_tol
      )
    }, error = function(e) {
      hessian_error <<- conditionMessage(e)
      NULL
    })

    if (!is.null(hessian_result)) {
      hessian_status <- "analytic"
      hessian_min_eigenvalue <- hessian_result$min_eigenvalue
      hessian_condition_number <- hessian_result$condition_number
      hessian_near_singular <- hessian_result$near_singular
      laplace <- (d_theta / 2) * log(2 * pi) -
        0.5 * hessian_result$log_det
    } else {
      hessian_status <- paste0("analytic_hessian_rejected: ", hessian_error)
      hessian_n_regularized <- NA_integer_
      laplace <- NA_real_
    }
  }

  out <- -fit_value + laplace + log_data_const + cov_log_prior_const
  if (!is.finite(out)) out <- -1e100
  list(
    log_evidence = out,
    hessian_status = hessian_status,
    hessian_n_regularized = hessian_n_regularized,
    hessian_min_eigenvalue = hessian_min_eigenvalue,
    hessian_condition_number = hessian_condition_number,
    hessian_near_singular = hessian_near_singular,
    laplace_fallback = laplace_fallback
  )
}

# ------------------------------------------------------------------------------
# 1. Node optimization / evidence
# ------------------------------------------------------------------------------
TVGP_optimize_node <- function(y, X, indices = seq_len(nrow(X)), nu = 1.5, m = 30,
                               ctx = NULL, init_theta = NULL,
                               cov_prior = c("inverse_gamma", "jointly_robust"),
                               cov_prior_control = NULL,
                               leaf_prior_spec = NULL,
                               leaf_prior_control = NULL,
                               maxit = 150, hessian = TRUE,
                               fixed_theta = NULL,
                               phi_bounds = c(1e-3, 10),
                               nugget_bounds = c(1e-8, 10),
                               optimizer_gradient_tol = 1e-3,
                               optimizer_boundary_tol = 1e-6,
                               newton_maxit = 20L, node_data = NULL) {
  cov_prior <- match.arg(cov_prior)
  if (length(optimizer_gradient_tol) != 1L ||
      !is.finite(optimizer_gradient_tol) || optimizer_gradient_tol <= 0) {
    stop("optimizer_gradient_tol must be one positive finite value.")
  }
  if (length(optimizer_boundary_tol) != 1L ||
      !is.finite(optimizer_boundary_tol) || optimizer_boundary_tol <= 0) {
    stop("optimizer_boundary_tol must be one positive finite value.")
  }
  if (is.null(node_data)) node_data <- TVGP_prepare_node_data(y, X, indices, m, ctx)
  y_ord <- node_data$y_ord
  X_ord <- node_data$X_ord
  response_summary <- list(
    n_train = length(y_ord),
    y_mean = mean(y_ord),
    y_sd = stats::sd(y_ord),
    y_zero_frac = mean(abs(y_ord) <= 1e-12)
  )
  D_dim <- ncol(X_ord)
  prior_spec <- TVGP_cov_prior_spec(X_ord, cov_prior, cov_prior_control)
  g_min <- if (cov_prior == "jointly_robust") prior_spec$details$g_min else 0
  if (is.null(leaf_prior_spec)) {
    leaf_prior_spec <- TVGP_leaf_prior_spec(leaf_prior_control)
  }

  if (is.null(init_theta)) init_theta <- TVGP_get_safe_init_theta(X_ord)
  if (length(init_theta) != D_dim + 1L) init_theta <- TVGP_get_safe_init_theta(X_ord)
  if (g_min > 0) init_theta[D_dim + 1L] <-
    log(max(exp(init_theta[D_dim + 1L]) - g_min, g_min))

  evaluation_cache <- new.env(parent = emptyenv())
  evaluation_cache$theta <- NULL
  evaluation_cache$result <- NULL
  evaluate_nll_grad <- function(theta) {
    if (!is.null(evaluation_cache$theta) &&
        identical(as.numeric(theta), evaluation_cache$theta)) {
      return(evaluation_cache$result)
    }
    result <- vecchia_nll_grad_counts(
      TVGP_to_log_parameters(theta, g_min), y_ord, X_ord, node_data$nn_mat, node_data$nn_count,
      nu, m, prior_spec$code, prior_spec$params,
      leaf_prior_spec$params, TRUE
    )
    result <- TVGP_shifted_nugget_derivatives(result, theta, g_min)
    evaluation_cache$theta <- as.numeric(theta)
    evaluation_cache$result <- result
    result
  }
  fn_nll <- function(theta) evaluate_nll_grad(theta)$nll
  gr_nll <- function(theta) evaluate_nll_grad(theta)$grad
  evaluate_nll_full <- function(theta) {
    result <- vecchia_nll_grad_hessian_counts(
      TVGP_to_log_parameters(theta, g_min), y_ord, X_ord, node_data$nn_mat, node_data$nn_count,
      nu, m, prior_spec$code, prior_spec$params, leaf_prior_spec$params
    )
    TVGP_shifted_nugget_derivatives(result, theta, g_min)
  }

  d_ranges <- apply(X_ord, 2, function(z) diff(range(z)))
  d_ranges[!is.finite(d_ranges) | d_ranges <= 0] <- 1
  lower_b <- c(log(pmax(d_ranges * phi_bounds[1], 1e-8)), log(nugget_bounds[1]))
  upper_b <- c(log(pmax(d_ranges * phi_bounds[2], d_ranges * phi_bounds[1] * 1.01)),
               log(nugget_bounds[2]))
  if (g_min > 0) {
    lower_b[D_dim + 1L] <- log(g_min) - 25
    upper_b[D_dim + 1L] <- log(max(nugget_bounds[2], 10 * g_min) - g_min)
  }
  init_theta <- pmin(pmax(init_theta, lower_b), upper_b)

  if (!is.null(fixed_theta)) {
    if (g_min > 0) stop("Fixed-theta screening is disabled for truncated JR.")
    theta_hat <- pmin(pmax(fixed_theta, lower_b), upper_b)
    final_res <- vecchia_nll_grad_counts(
      theta_hat, y_ord, X_ord, node_data$nn_mat, node_data$nn_count,
      nu, m, prior_spec$code, prior_spec$params,
      leaf_prior_spec$params, FALSE
    )
    evidence <- TVGP_log_evidence(final_res$nll, theta_hat, NULL,
                                  length(y_ord),
                                  cov_log_prior_const = prior_spec$log_normalizer,
                                  leaf_prior_spec = leaf_prior_spec,
                                  hessian = FALSE)
    return(list(
      theta_hat = theta_hat,
      n_train = response_summary$n_train,
      y_mean = response_summary$y_mean,
      y_sd = response_summary$y_sd,
      y_zero_frac = response_summary$y_zero_frac,
      gamma_mean = final_res$gamma_mean,
      gamma_var_factor = final_res$gamma_var_factor,
      sigma2_shape = final_res$sigma2_shape,
      sigma2_scale = final_res$sigma2_scale,
      sigma2_mean = final_res$sigma2_mean,
      sigma2_hat = final_res$sigma2_mean,
      log_evidence = evidence$log_evidence, ord_indices = node_data$ord_indices,
      convergence = NA_integer_, value = final_res$nll,
      optimization_failed = !is.finite(evidence$log_evidence) || evidence$log_evidence <= -1e90,
      optimization_failure_reason = if (evidence$log_evidence <= -1e90) "nonfinite_fixed_theta_evidence" else NA_character_,
      optimizer_message = "fixed_theta",
      function_evaluations = 1L,
      gradient_evaluations = 0L,
      gradient_norm = NA_real_,
      gradient_inf_norm = NA_real_,
      minimum_bound_distance = NA_real_,
      mode_on_boundary = NA,
      initial_optimizer_convergence = NA_integer_,
      initial_optimizer_message = "fixed_theta",
      newton_converged = NA,
      newton_iterations = 0L,
      newton_accepted_steps = 0L,
      newton_damping_steps = 0L,
      newton_line_search_evaluations = 0L,
      newton_failure_reason = NA_character_,
      lower_bounds = lower_b,
      upper_bounds = upper_b,
      optimizer_gradient_tol = optimizer_gradient_tol,
      optimizer_boundary_tol = optimizer_boundary_tol,
      hessian_status = evidence$hessian_status,
      hessian_n_regularized = evidence$hessian_n_regularized,
      hessian_min_eigenvalue = evidence$hessian_min_eigenvalue,
      hessian_condition_number = evidence$hessian_condition_number,
      hessian_near_singular = evidence$hessian_near_singular,
      laplace_fallback = evidence$laplace_fallback,
      leaf_prior = leaf_prior_spec$name,
      leaf_prior_details = leaf_prior_spec$details,
      cov_prior = cov_prior,
      cov_prior_details = prior_spec$details
    ))
  }

  optim_error <- NULL
  fit <- tryCatch({
    optim(
      init_theta, fn_nll, gr = gr_nll, method = "L-BFGS-B",
      lower = lower_b, upper = upper_b,
      control = list(
        maxit = maxit,
        factr = if (is.null(ctx$optimizer_factr)) 1e3 else ctx$optimizer_factr,
        pgtol = min(1e-4, optimizer_gradient_tol / 10),
        lmm = 10L
      )
    )
  }, error = function(e) {
    optim_error <<- conditionMessage(e)
    NULL
  })

  if (is.null(fit) || !is.finite(fit$value)) {
    leaf_pars <- leaf_prior_spec$params
    failure_shape <- leaf_pars[3L] + length(y_ord) / 2
    failure_sigma2_mean <- if (failure_shape > 1) {
      leaf_pars[4L] / (failure_shape - 1)
    } else {
      Inf
    }
    return(list(
      theta_hat = init_theta, log_evidence = -1e100,
      n_train = response_summary$n_train,
      y_mean = response_summary$y_mean,
      y_sd = response_summary$y_sd,
      y_zero_frac = response_summary$y_zero_frac,
      gamma_mean = leaf_pars[1L], gamma_var_factor = leaf_pars[2L],
      sigma2_shape = failure_shape, sigma2_scale = leaf_pars[4L],
      sigma2_mean = failure_sigma2_mean, sigma2_hat = failure_sigma2_mean,
      ord_indices = node_data$ord_indices, convergence = 99L, value = Inf,
      optimization_failed = TRUE,
      optimization_failure_reason = if (is.null(fit)) "optim_error" else "nonfinite_optim_value",
      optimizer_message = if (is.null(optim_error)) "non-finite optimizer value" else optim_error,
      function_evaluations = if (is.null(fit) || is.null(fit$counts)) NA_integer_ else as.integer(fit$counts[["function"]]),
      gradient_evaluations = if (is.null(fit) || is.null(fit$counts)) NA_integer_ else as.integer(fit$counts[["gradient"]]),
      gradient_norm = NA_real_,
      gradient_inf_norm = NA_real_,
      minimum_bound_distance = NA_real_,
      mode_on_boundary = NA,
      initial_optimizer_convergence = if (is.null(fit)) 99L else fit$convergence,
      initial_optimizer_message = if (is.null(optim_error)) NA_character_ else optim_error,
      newton_converged = FALSE,
      newton_iterations = 0L,
      newton_accepted_steps = 0L,
      newton_damping_steps = 0L,
      newton_line_search_evaluations = 0L,
      newton_failure_reason = "optimizer_did_not_return_a_finite_point",
      lower_bounds = lower_b,
      upper_bounds = upper_b,
      optimizer_gradient_tol = optimizer_gradient_tol,
      optimizer_boundary_tol = optimizer_boundary_tol,
      hessian_status = "not_evaluated",
      hessian_n_regularized = NA_integer_,
      hessian_min_eigenvalue = NA_real_,
      hessian_condition_number = NA_real_,
      hessian_near_singular = NA,
      laplace_fallback = FALSE,
      leaf_prior = leaf_prior_spec$name,
      leaf_prior_details = leaf_prior_spec$details,
      cov_prior = cov_prior,
      cov_prior_details = prior_spec$details
    ))
  }

  initial_optimizer_convergence <- fit$convergence
  initial_optimizer_message <- if (is.null(fit$message)) NA_character_ else as.character(fit$message)
  newton <- if (isTRUE(hessian)) {
    TVGP_newton_refine(
      fit$par, evaluate_nll_full, lower_b, upper_b,
      gradient_tol = optimizer_gradient_tol,
      boundary_tol = optimizer_boundary_tol,
      maxit = newton_maxit, evaluate_light = evaluate_nll_grad
    )
  } else {
    list(
      par = fit$par, result = NULL, converged = NA,
      iterations = 0L, accepted_steps = 0L, damping_steps = 0L,
      line_search_evaluations = 0L, failure_reason = NA_character_
    )
  }
  if (isTRUE(hessian) && isTRUE(newton$converged)) fit$convergence <- 0L
  theta_hat <- newton$par
  fit$par <- theta_hat
  final_res <- if (isTRUE(hessian) && !is.null(newton$result)) {
    newton$result
  } else if (isTRUE(hessian)) {
    evaluate_nll_full(theta_hat)
  } else {
    evaluate_nll_grad(theta_hat)
  }
  fit$value <- final_res$nll
  evidence <- TVGP_log_evidence(final_res$nll, theta_hat, final_res$hessian,
                                length(y_ord),
                                cov_log_prior_const = prior_spec$log_normalizer,
                                leaf_prior_spec = leaf_prior_spec,
                                hessian = hessian)
  final_gradient <- if (isTRUE(hessian)) {
    as.numeric(final_res$grad)
  } else {
    tryCatch(gr_nll(theta_hat), error = function(e) rep(NA_real_, length(theta_hat)))
  }
  gradient_norm <- if (all(is.finite(final_gradient))) sqrt(sum(final_gradient^2)) else NA_real_
  gradient_inf_norm <- if (all(is.finite(final_gradient))) max(abs(final_gradient)) else NA_real_
  bound_distance <- pmin(theta_hat - lower_b, upper_b - theta_hat)
  minimum_bound_distance <- if (all(is.finite(bound_distance))) {
    min(bound_distance)
  } else {
    NA_real_
  }
  mode_on_boundary <- !is.finite(minimum_bound_distance) ||
    minimum_bound_distance <= optimizer_boundary_tol
  optimizer_converged <- identical(as.integer(fit$convergence), 0L)
  gradient_accepted <- is.finite(gradient_inf_norm) &&
    gradient_inf_norm <= optimizer_gradient_tol
  evidence_failed <- !isTRUE(final_res$valid) || !optimizer_converged ||
    !gradient_accepted || isTRUE(mode_on_boundary) ||
    !is.finite(evidence$log_evidence) || evidence$log_evidence <= -1e90
  failure_reason <- if (!isTRUE(final_res$valid)) {
    paste0("covariance_evaluation: ", final_res$failure_reason)
  } else if (!optimizer_converged) {
    paste0("optimizer_nonconvergence_code_", fit$convergence)
  } else if (!is.finite(gradient_inf_norm)) {
    "nonfinite_gradient_at_mode"
  } else if (!gradient_accepted) {
    sprintf(
      "gradient_tolerance_not_met: inf_norm=%.6g > %.6g",
      gradient_inf_norm, optimizer_gradient_tol
    )
  } else if (isTRUE(mode_on_boundary)) {
    sprintf(
      "mode_on_optimizer_boundary: minimum_distance=%.6g <= %.6g",
      minimum_bound_distance, optimizer_boundary_tol
    )
  } else if (startsWith(evidence$hessian_status, "analytic_hessian_rejected:")) {
    evidence$hessian_status
  } else if (evidence_failed) {
    "nonfinite_log_evidence"
  } else {
    NA_character_
  }
  if (evidence_failed) evidence$log_evidence <- -1e100

  list(
    theta_hat = TVGP_to_log_parameters(theta_hat, g_min),
    psi_hat = theta_hat,
    n_train = response_summary$n_train,
    y_mean = response_summary$y_mean,
    y_sd = response_summary$y_sd,
    y_zero_frac = response_summary$y_zero_frac,
    gamma_mean = final_res$gamma_mean,
    gamma_var_factor = final_res$gamma_var_factor,
    sigma2_shape = final_res$sigma2_shape,
    sigma2_scale = final_res$sigma2_scale,
    sigma2_mean = final_res$sigma2_mean,
    sigma2_hat = final_res$sigma2_mean,
    log_evidence = evidence$log_evidence,
    ord_indices = node_data$ord_indices,
    convergence = fit$convergence,
    value = fit$value,
    optimization_failed = evidence_failed,
    optimization_failure_reason = failure_reason,
    optimizer_message = if (is.null(fit$message)) NA_character_ else as.character(fit$message),
    function_evaluations = as.integer(fit$counts[["function"]]),
    gradient_evaluations = as.integer(fit$counts[["gradient"]]),
    gradient_norm = gradient_norm,
    gradient_inf_norm = gradient_inf_norm,
    minimum_bound_distance = minimum_bound_distance,
    mode_on_boundary = mode_on_boundary,
    initial_optimizer_convergence = initial_optimizer_convergence,
    initial_optimizer_message = initial_optimizer_message,
    newton_converged = newton$converged,
    newton_iterations = newton$iterations,
    newton_accepted_steps = newton$accepted_steps,
    newton_damping_steps = newton$damping_steps,
    newton_line_search_evaluations = newton$line_search_evaluations,
    newton_failure_reason = newton$failure_reason,
    lower_bounds = lower_b,
    upper_bounds = upper_b,
    optimizer_gradient_tol = optimizer_gradient_tol,
    optimizer_boundary_tol = optimizer_boundary_tol,
    hessian_status = evidence$hessian_status,
    hessian_n_regularized = evidence$hessian_n_regularized,
    hessian_min_eigenvalue = evidence$hessian_min_eigenvalue,
    hessian_condition_number = evidence$hessian_condition_number,
    hessian_near_singular = evidence$hessian_near_singular,
    laplace_fallback = evidence$laplace_fallback,
    leaf_prior = leaf_prior_spec$name,
    leaf_prior_details = leaf_prior_spec$details,
    cov_prior = cov_prior,
    cov_prior_details = prior_spec$details
  )
}

# Penalized negative log density at candidate starting values, evaluated in the
# coordinates, nugget shift and bounds used by TVGP_optimize_node. The values
# only rank starting values; no optimization is performed here.
TVGP_screen_start_objectives <- function(y, X, indices, nu, m, ctx, cov_prior,
                                         cov_prior_control, leaf_prior_spec,
                                         starts, phi_bounds, nugget_bounds) {
  node_data <- TVGP_prepare_node_data(y, X, indices, m, ctx)
  X_ord <- node_data$X_ord
  D_dim <- ncol(X_ord)
  prior_spec <- TVGP_cov_prior_spec(X_ord, cov_prior, cov_prior_control)
  g_min <- if (cov_prior == "jointly_robust") prior_spec$details$g_min else 0
  d_ranges <- apply(X_ord, 2, function(z) diff(range(z)))
  d_ranges[!is.finite(d_ranges) | d_ranges <= 0] <- 1
  lower_b <- c(log(pmax(d_ranges * phi_bounds[1], 1e-8)), log(nugget_bounds[1]))
  upper_b <- c(log(pmax(d_ranges * phi_bounds[2], d_ranges * phi_bounds[1] * 1.01)),
               log(nugget_bounds[2]))
  if (g_min > 0) {
    lower_b[D_dim + 1L] <- log(g_min) - 25
    upper_b[D_dim + 1L] <- log(max(nugget_bounds[2], 10 * g_min) - g_min)
  }
  vapply(starts, function(theta) {
    if (g_min > 0) theta[D_dim + 1L] <- log(max(exp(theta[D_dim + 1L]) - g_min, g_min))
    theta <- pmin(pmax(theta, lower_b), upper_b)
    value <- tryCatch({
      result <- vecchia_nll_grad_counts(
        TVGP_to_log_parameters(theta, g_min), node_data$y_ord, X_ord,
        node_data$nn_mat, node_data$nn_count, nu, m, prior_spec$code,
        prior_spec$params, leaf_prior_spec$params, FALSE
      )
      if (!isTRUE(result$valid)) {
        Inf
      } else if (g_min > 0) {
        result$nll - (theta[D_dim + 1L] -
                        TVGP_to_log_parameters(theta, g_min)[D_dim + 1L])
      } else {
        result$nll
      }
    }, error = function(e) Inf)
    if (is.finite(value)) value else Inf
  }, numeric(1))
}

TVGP_optimize_node_with_retries <- function(
    y, X, indices, nu, m, ctx,
    cov_prior, cov_prior_control, leaf_prior_spec,
    maxit, hessian, fixed_theta = NULL,
    max_attempts = 12L,
    optimizer_gradient_tol = 1e-3,
    optimizer_boundary_tol = 1e-6,
    newton_maxit = 20L, init_theta = NULL, multistart = NULL) {
  if (!is.null(ctx$fit_cache) && !identical(ctx$fit_cache$data_hash,
      digest::digest(list(y, X), algo = "sha256"))) {
    stop("The fitted-node cache belongs to different data.")
  }
  identity <- list(indices = sort(as.integer(indices)), nu = nu, m = m,
    cov_prior = cov_prior, cov_prior_control = cov_prior_control,
    leaf_prior_spec = leaf_prior_spec, maxit = maxit, hessian = hessian,
    fixed_theta = fixed_theta, max_attempts = max_attempts,
    gradient_tol = optimizer_gradient_tol, boundary_tol = optimizer_boundary_tol,
    newton_maxit = newton_maxit, init_theta = init_theta, multistart = multistart,
    optimizer_factr = ctx$optimizer_factr, zero_fast_start = ctx$zero_fast_start)
  cached <- TVGP_cache_read(ctx$fit_cache, identity)
  if (!is.null(cached)) return(cached)
  node_data <- TVGP_prepare_node_data(y, X, indices, m, ctx)
  attempt_row <- function(fit, attempt, label, init_theta, phi_bounds,
                          nugget_bounds) {
    data.frame(
      attempt = as.integer(attempt),
      attempt_label = label,
      phi_lower_multiplier = phi_bounds[1L],
      phi_upper_multiplier = phi_bounds[2L],
      nugget_lower = nugget_bounds[1L],
      nugget_upper = nugget_bounds[2L],
      initial_theta = paste(signif(init_theta, 10), collapse = ";"),
      final_theta = paste(signif(fit$theta_hat, 10), collapse = ";"),
      optimization_failed = isTRUE(fit$optimization_failed),
      failure_reason = if (is.null(fit$optimization_failure_reason)) {
        NA_character_
      } else {
        fit$optimization_failure_reason
      },
      initial_optimizer_convergence = fit$initial_optimizer_convergence,
      initial_optimizer_message = fit$initial_optimizer_message,
      final_optimizer_convergence = fit$convergence,
      gradient_inf_norm = fit$gradient_inf_norm,
      minimum_bound_distance = fit$minimum_bound_distance,
      mode_on_boundary = fit$mode_on_boundary,
      newton_converged = fit$newton_converged,
      newton_iterations = fit$newton_iterations,
      newton_accepted_steps = fit$newton_accepted_steps,
      newton_damping_steps = fit$newton_damping_steps,
      newton_line_search_evaluations = fit$newton_line_search_evaluations,
      newton_failure_reason = fit$newton_failure_reason,
      hessian_status = fit$hessian_status,
      hessian_min_eigenvalue = fit$hessian_min_eigenvalue,
      hessian_condition_number = fit$hessian_condition_number,
      log_evidence = fit$log_evidence,
      stringsAsFactors = FALSE
    )
  }

  if (!is.null(fixed_theta)) {
    fit <- TVGP_optimize_node(
      y, X, indices, nu, m, ctx,
      cov_prior = cov_prior, cov_prior_control = cov_prior_control,
      leaf_prior_spec = leaf_prior_spec, maxit = maxit,
      hessian = hessian, fixed_theta = fixed_theta,
      optimizer_gradient_tol = optimizer_gradient_tol,
      optimizer_boundary_tol = optimizer_boundary_tol,
      newton_maxit = newton_maxit, node_data = node_data
    )
    fit$optimization_attempts <- 1L
    fit$optimization_attempt_label <- "fixed_theta"
    fit$previous_failure_reasons <- NA_character_
    fit$optimization_attempt_diagnostics <- attempt_row(
      fit, 1L, "fixed_theta", fixed_theta, c(1e-3, 10), c(1e-8, 10)
    )
    TVGP_cache_write(ctx$fit_cache, identity, fit)
    return(fit)
  }

  X_node <- X[indices, , drop = FALSE]
  multistart_rows <- list()
  multistart_reasons <- character()
  multistart_screen <- NULL
  if (!is.null(multistart)) {
    # Rank the common starts and one short-range start per input by their
    # objective values. With select = "first_passing", optimize in rank order
    # and return the first fit that passes every check, trying at most
    # max_fits starts. Otherwise optimize from the default start and the top
    # best-ranked others and keep the largest passing log evidence. The starts
    # depend only on the leaf data, so the score is a deterministic function
    # of the leaf subset.
    ms_phi_bounds <- as.numeric(multistart$phi_bounds)
    ms_nugget_bounds <- c(1e-8, 10)
    common_starts <- list(
      default_start = c(0.20, 0.05),
      long_range_start = c(0.50, 0.10),
      short_range_start = c(0.08, 0.01)
    )
    starts <- lapply(common_starts, function(z) {
      TVGP_get_safe_init_theta(X_node, range_frac = z[1L], nugget = z[2L])
    })
    for (j in seq_len(ncol(X_node))) {
      theta <- TVGP_get_safe_init_theta(
        X_node, range_frac = multistart$other_frac, nugget = multistart$nugget
      )
      theta[j] <- TVGP_get_safe_init_theta(
        X_node, range_frac = multistart$short_frac, nugget = multistart$nugget
      )[j]
      starts[[sprintf("short_range_x%d_start", j)]] <- theta
    }
    screen <- TVGP_screen_start_objectives(
      y, X, indices, nu, m, ctx, cov_prior, cov_prior_control,
      leaf_prior_spec, starts, ms_phi_bounds, ms_nugget_bounds
    )
    first_passing <- identical(multistart$select, "first_passing")
    if (first_passing) {
      selected <- head(order(screen), as.integer(multistart$max_fits))
    } else {
      ranked_others <- setdiff(order(screen), 1L)
      selected <- c(1L, head(ranked_others, as.integer(multistart$top)))
    }
    fits <- list()
    for (a in seq_along(selected)) {
      i <- selected[a]
      fits[[a]] <- TVGP_optimize_node(
        y, X, indices, nu, m, ctx,
        init_theta = starts[[i]],
        cov_prior = cov_prior, cov_prior_control = cov_prior_control,
        leaf_prior_spec = leaf_prior_spec, maxit = maxit, hessian = hessian,
        phi_bounds = ms_phi_bounds,
        nugget_bounds = ms_nugget_bounds,
        optimizer_gradient_tol = optimizer_gradient_tol,
        optimizer_boundary_tol = optimizer_boundary_tol,
        newton_maxit = newton_maxit
      )
      multistart_rows[[a]] <- attempt_row(
        fits[[a]], a, paste0("multistart_", names(starts)[i]), starts[[i]],
        ms_phi_bounds, ms_nugget_bounds
      )
      if (first_passing && !isTRUE(fits[[a]]$optimization_failed)) break
    }
    multistart_screen <- data.frame(
      start = names(starts), screening_objective = unname(screen),
      selected = seq_along(starts) %in% selected[seq_along(fits)],
      stringsAsFactors = FALSE
    )
    failed <- vapply(fits, function(f) isTRUE(f$optimization_failed), logical(1))
    multistart_reasons <- vapply(which(failed), function(a) {
      reason <- fits[[a]]$optimization_failure_reason
      sprintf("multistart[%s]: %s", names(starts)[selected[a]],
              if (is.null(reason)) NA_character_ else reason)
    }, character(1))
    if (any(!failed)) {
      passed <- which(!failed)
      evidence <- vapply(fits[passed], function(f) f$log_evidence, numeric(1))
      best <- passed[which.max(evidence)]
      fit <- fits[[best]]
      fit$optimization_attempts <- length(fits)
      fit$optimization_attempt_label <- paste0("multistart_", names(starts)[selected[best]])
      fit$previous_failure_reasons <- if (length(multistart_reasons)) {
        paste(multistart_reasons, collapse = " | ")
      } else {
        NA_character_
      }
      fit$optimization_attempt_diagnostics <- do.call(rbind, multistart_rows)
      fit$multistart_screen <- multistart_screen
      TVGP_cache_write(ctx$fit_cache, identity, fit)
      return(fit)
    }
  }
  default_phi_bounds <- c(1e-3, 10)
  default_nugget_bounds <- c(1e-8, 10)
  expanded_phi_bounds <- c(1e-4, 100)
  expanded_nugget_bounds <- c(1e-10, 100)
  attempt_plan <- list(
    list(label = "default_start", range_frac = 0.20, nugget = 0.05,
         phi_bounds = default_phi_bounds, nugget_bounds = default_nugget_bounds),
    list(label = "long_range_start", range_frac = 0.50, nugget = 0.10,
         phi_bounds = default_phi_bounds, nugget_bounds = default_nugget_bounds),
    list(label = "short_range_start", range_frac = 0.08, nugget = 0.01,
         phi_bounds = default_phi_bounds, nugget_bounds = default_nugget_bounds),
    list(label = "expanded_bounds_default_start", range_frac = 0.20, nugget = 0.05,
         phi_bounds = expanded_phi_bounds, nugget_bounds = expanded_nugget_bounds),
    list(label = "expanded_bounds_long_range_start", range_frac = 0.50, nugget = 0.10,
         phi_bounds = expanded_phi_bounds, nugget_bounds = expanded_nugget_bounds),
    list(label = "range_1000_default_start", range_frac = 0.20, nugget = 0.05,
         phi_bounds = c(1e-4, 1000), nugget_bounds = expanded_nugget_bounds),
    list(label = "range_1000_long_range_start", range_frac = 0.50, nugget = 0.10,
         phi_bounds = c(1e-4, 1000), nugget_bounds = expanded_nugget_bounds),
    list(label = "range_10000_default_start", range_frac = 0.20, nugget = 0.05,
         phi_bounds = c(1e-4, 10000), nugget_bounds = expanded_nugget_bounds),
    list(label = "range_10000_long_range_start", range_frac = 0.50, nugget = 0.10,
         phi_bounds = c(1e-4, 10000), nugget_bounds = expanded_nugget_bounds),
    list(label = "small_nugget_default_start", range_frac = 0.20, nugget = 0.05,
         phi_bounds = c(1e-4, 100), nugget_bounds = c(1e-14, 100)),
    list(label = "small_nugget_long_range_start", range_frac = 0.50, nugget = 0.10,
         phi_bounds = c(1e-4, 100), nugget_bounds = c(1e-14, 100)),
    list(label = "small_nugget_extended_range_start", range_frac = 0.50, nugget = 0.10,
         phi_bounds = c(1e-4, 10000), nugget_bounds = c(1e-14, 100))
  )
  is_constant <- all(y[indices] == y[indices[1L]])
  if (isTRUE(ctx$zero_fast_start) && is_constant) {
    # Retain the original wide-bound starts without first exhausting narrow bounds.
    attempt_plan <- attempt_plan[c(8L, 9L, 12L)]
    for (i in seq_along(attempt_plan)) attempt_plan[[i]]$maxit <- max(maxit, 800L)
    attempt_plan <- c(list(list(label = "constant_long_range_start",
      range_frac = 100, nugget = max(1e-10, 0.1/(length(indices)/2 + 3)),
      phi_bounds = c(1e-4, 10000), nugget_bounds = c(1e-14, 100),
      maxit = max(maxit, 800L))), attempt_plan)
  }
  if (!is.null(init_theta)) {
    stopifnot(length(init_theta) == ncol(X) + 1L, all(is.finite(init_theta)))
    ranges <- apply(X_node, 2L, function(x) diff(range(x)))
    ranges[!is.finite(ranges) | ranges <= 0] <- 1
    multipliers <- exp(head(init_theta, -1L))/ranges
    warm_bounds <- c(min(1e-4, min(multipliers)/2),
      max(if (is_constant) 10000 else 10, max(multipliers)*2))
    attempt_plan <- c(list(list(label = "parent_warm_start", theta = init_theta,
      range_frac = .2, nugget = .05, phi_bounds = warm_bounds,
      nugget_bounds = c(min(1e-14, exp(tail(init_theta, 1L))/2),
        max(100, 2*exp(tail(init_theta, 1L)))))), attempt_plan)
    # Constant leaves first try their dedicated start; the parent is the next fallback.
    if (isTRUE(ctx$zero_fast_start) && is_constant) {
      attempt_plan <- attempt_plan[c(2L, 1L, seq.int(3L, length(attempt_plan)))]
    } else {
      # Replace the default start while retaining the wide-bound fallbacks.
      attempt_plan <- attempt_plan[-2L]
    }
  }
  max_attempts <- min(length(attempt_plan), max(1L, as.integer(max_attempts)))
  failure_reasons <- multistart_reasons
  attempt_diagnostics <- vector("list", max_attempts)
  fit <- NULL
  for (attempt in seq_len(max_attempts)) {
    plan <- attempt_plan[[attempt]]
    # Attempts after the fifth expand only the numerical search limits.
    attempt_maxit <- if (!is.null(plan$maxit)) plan$maxit else
      if (attempt > 5L) max(maxit, 800L) else maxit
    attempt_newton_maxit <- if (attempt > 5L) max(newton_maxit, 40L) else newton_maxit
    attempt_init <- if (!is.null(plan$theta)) plan$theta else TVGP_get_safe_init_theta(
      X_node, range_frac = plan$range_frac, nugget = plan$nugget
    )
    fit <- TVGP_optimize_node(
      y, X, indices, nu, m, ctx,
      init_theta = attempt_init,
      cov_prior = cov_prior, cov_prior_control = cov_prior_control,
      leaf_prior_spec = leaf_prior_spec, maxit = attempt_maxit, hessian = hessian,
      phi_bounds = plan$phi_bounds,
      nugget_bounds = plan$nugget_bounds,
      optimizer_gradient_tol = optimizer_gradient_tol,
      optimizer_boundary_tol = optimizer_boundary_tol,
      newton_maxit = attempt_newton_maxit, node_data = node_data
    )
    attempt_diagnostics[[attempt]] <- attempt_row(
      fit, attempt, plan$label, attempt_init,
      plan$phi_bounds, plan$nugget_bounds
    )
    if (!isTRUE(fit$optimization_failed)) {
      fit$optimization_attempts <- length(multistart_rows) + attempt
      fit$optimization_attempt_label <- plan$label
      fit$previous_failure_reasons <- if (length(failure_reasons)) {
        paste(failure_reasons, collapse = " | ")
      } else {
        NA_character_
      }
      fit$optimization_attempt_diagnostics <- do.call(
        rbind, c(multistart_rows, attempt_diagnostics[seq_len(attempt)])
      )
      fit$multistart_screen <- multistart_screen
      TVGP_cache_write(ctx$fit_cache, identity, fit)
      return(fit)
    }
    failure_reasons <- c(
      failure_reasons,
      sprintf("attempt_%d[%s]: %s", attempt, plan$label,
              fit$optimization_failure_reason)
    )
  }
  fit$optimization_attempts <- length(multistart_rows) + max_attempts
  fit$optimization_attempt_label <- attempt_plan[[max_attempts]]$label
  fit$previous_failure_reasons <- paste(failure_reasons, collapse = " | ")
  fit$optimization_attempt_diagnostics <- do.call(
    rbind, c(multistart_rows, attempt_diagnostics)
  )
  fit$multistart_screen <- multistart_screen
  fit
}

TVGP_fit_failure_rows <- function(fit, stage, node_key,
                                  candidate_position = NA_integer_,
                                  candidate_index = NA_integer_,
                                  split_dim = NA_integer_, split_cut = NA_real_,
                                  child = NA_character_, child_n = NA_integer_) {
  attempts <- fit$optimization_attempt_diagnostics
  if (is.null(attempts) || !nrow(attempts)) {
    attempts <- data.frame(attempt = NA_integer_, stringsAsFactors = FALSE)
  }
  cbind(
    data.frame(
      stage = stage,
      node_key = node_key,
      candidate_position = as.integer(candidate_position),
      candidate_index = as.integer(candidate_index),
      split_dim = as.integer(split_dim),
      split_cut = as.numeric(split_cut),
      child = child,
      child_n = as.integer(child_n),
      stringsAsFactors = FALSE
    )[rep(1L, nrow(attempts)), , drop = FALSE],
    attempts,
    stringsAsFactors = FALSE
  )
}

TVGP_write_failure_diagnostics <- function(rows, path) {
  if (is.null(path) || !nzchar(path) || is.null(rows) || !nrow(rows)) {
    return(invisible(NULL))
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(rows, path, row.names = FALSE)
  invisible(normalizePath(path, mustWork = FALSE))
}

# ------------------------------------------------------------------------------
# 2. SMC lookahead
# ------------------------------------------------------------------------------
TVGP_generate_candidates <- function(y, X, curr_indices, n_candidates_per_axis = 10,
                                     min_leaf_size = 30, leaf_prior_spec) {
  D <- ncol(X)
  X_curr <- X[curr_indices, , drop = FALSE]
  y_curr <- y[curr_indices]
  parent_iid <- TVGP_log_iid_evidence(y_curr, leaf_prior_spec)
  candidates <- list()
  axis_counts <- integer(D)

  for (j in seq_len(D)) {
    vals <- sort(unique(X_curr[, j]))
    n_vals <- length(vals)
    if (n_vals < 2) next

    if (n_vals <= n_candidates_per_axis + 1L) {
      cutpoints <- vals[-n_vals]
    } else {
      probs <- seq(0, 1, length.out = n_candidates_per_axis + 2L)
      cutpoints <- unique(as.numeric(quantile(vals, probs[-c(1, length(probs))],
                                              names = FALSE, type = 1)))
    }

    for (c_val in cutpoints) {
      left <- X_curr[, j] <= c_val
      n_left <- sum(left)
      n_right <- length(left) - n_left
      if (n_left >= min_leaf_size && n_right >= min_leaf_size) {
        idx_L <- curr_indices[left]
        idx_R <- curr_indices[!left]
        cheap_log_BF <- TVGP_log_iid_evidence(y[idx_L], leaf_prior_spec) +
          TVGP_log_iid_evidence(y[idx_R], leaf_prior_spec) - parent_iid
        candidates[[length(candidates) + 1L]] <- list(
          j = j, c = c_val, idx_L = idx_L, idx_R = idx_R,
          cheap_log_BF = cheap_log_BF
        )
        axis_counts[j] <- axis_counts[j] + 1L
      }
    }
  }

  if (length(candidates) == 0) return(candidates)

  active_axes <- which(axis_counts > 0L)
  d_A <- length(active_axes)
  for (k in seq_along(candidates)) {
    j <- candidates[[k]]$j
    candidates[[k]]$log_prior <- -log(d_A) - log(axis_counts[j])
  }
  candidates
}

TVGP_limit_candidate_support <- function(candidates, budget = 0L) {
  if (is.null(budget) || budget == 0L || length(candidates) <= budget) return(candidates)
  axes <- vapply(candidates, `[[`, integer(1), "j")
  groups <- split(seq_along(candidates), axes)
  if (budget < length(groups)) stop("Candidate budget must include every admissible axis.")
  capacity <- lengths(groups)
  quota <- rep(0L, length(groups))
  while (sum(quota) < budget) {
    available <- which(quota < capacity)
    take <- head(available, budget - sum(quota))
    quota[take] <- quota[take] + 1L
  }
  # Equal axis probabilities, with evenly spaced rules on each axis; y is unused.
  keep <- unlist(Map(function(index, n) {
    positions <- round(seq(0, length(index) + 1L, length.out = n + 2L))
    index[positions[-c(1L, length(positions))]]
  }, groups, quota), use.names = FALSE)
  out <- candidates[keep]
  counts <- table(vapply(out, `[[`, integer(1), "j"))
  for (k in seq_along(out)) {
    out[[k]]$log_prior <- -log(length(counts)) - log(counts[as.character(out[[k]]$j)])
  }
  out
}

TVGP_select_candidates <- function(candidates, max_full_candidates = NULL) {
  K <- length(candidates)
  if (is.null(max_full_candidates) || max_full_candidates <= 0 || K <= max_full_candidates) {
    return(seq_len(K))
  }

  scores <- vapply(candidates, function(z) z$cheap_log_BF + z$log_prior, numeric(1))
  axes <- vapply(candidates, function(z) z$j, integer(1))
  axis_best <- unlist(tapply(seq_len(K), axes, function(ii) ii[which.max(scores[ii])]))
  global_best <- order(scores, decreasing = TRUE)[seq_len(min(K, max_full_candidates))]
  keep <- unique(c(axis_best, global_best))
  keep <- keep[order(scores[keep], decreasing = TRUE)]
  keep[seq_len(min(length(keep), max_full_candidates))]
}

TVGP_evaluate_transition <- function(node, y, X, ctx, nu, m_vecchia,
                                     cov_prior, cov_prior_control,
                                     leaf_prior_spec,
                                     alpha, beta, n_candidates_per_axis,
                                     min_leaf_size, max_full_candidates,
                                     full_maxit, screen_with_fixed_theta,
                                     renormalize_candidate_prior,
                                     hessian, parallel_cores = 1L,
                                     parallel_mc_preschedule = FALSE,
                                     optimizer_gradient_tol = 1e-3,
                                     optimizer_boundary_tol = 1e-6,
                                     optimizer_newton_maxit = 20L,
                                     failure_diagnostics_file = NULL) {
  rho <- alpha * (1 + node$depth)^(-beta)
  candidates <- TVGP_generate_candidates(
    y, X, node$indices, n_candidates_per_axis, min_leaf_size,
    leaf_prior_spec
  )
  candidates <- TVGP_limit_candidate_support(candidates, ctx$candidate_budget)

  if (length(candidates) == 0) {
    return(list(
      can_split = FALSE, log_w = 0, log_prob_split = -Inf,
      candidates = list(), selected = integer(0), n_candidates = 0L,
      forced_stop_reason = "no_valid_split",
      n_failed_candidates = 0L, n_failed_child_fits = 0L,
      n_nonconverged_child_fits = 0L, n_laplace_fallback_child_fits = 0L,
      n_regularized_hessian_child_fits = 0L
    ))
  }

  selected <- TVGP_select_candidates(candidates, max_full_candidates)
  log_prior <- vapply(candidates[selected], `[[`, numeric(1), "log_prior")
  if (isTRUE(renormalize_candidate_prior)) {
    log_prior <- log_prior - TVGP_logsumexp(log_prior)
  }

  eval_candidate <- function(ii) {
    cand <- candidates[[selected[ii]]]
    fixed <- if (isTRUE(screen_with_fixed_theta)) node$theta_hat else NULL
    fit_L <- TVGP_optimize_node_with_retries(
      y, X, cand$idx_L, nu, m_vecchia, ctx,
      cov_prior = cov_prior, cov_prior_control = cov_prior_control,
      leaf_prior_spec = leaf_prior_spec,
      maxit = full_maxit, hessian = hessian,
      fixed_theta = fixed,
      optimizer_gradient_tol = optimizer_gradient_tol,
      optimizer_boundary_tol = optimizer_boundary_tol,
      newton_maxit = optimizer_newton_maxit,
      init_theta = if (isTRUE(ctx$parent_warm_start)) node$theta_hat else NULL
    )
    fit_R <- TVGP_optimize_node_with_retries(
      y, X, cand$idx_R, nu, m_vecchia, ctx,
      cov_prior = cov_prior, cov_prior_control = cov_prior_control,
      leaf_prior_spec = leaf_prior_spec,
      maxit = full_maxit, hessian = hessian,
      fixed_theta = fixed,
      optimizer_gradient_tol = optimizer_gradient_tol,
      optimizer_boundary_tol = optimizer_boundary_tol,
      newton_maxit = optimizer_newton_maxit,
      init_theta = if (isTRUE(ctx$parent_warm_start)) node$theta_hat else NULL
    )

    failed_L <- isTRUE(fit_L$optimization_failed) || fit_L$log_evidence <= -1e90
    failed_R <- isTRUE(fit_R$optimization_failed) || fit_R$log_evidence <= -1e90
    log_BF <- -Inf
    if (!failed_L && !failed_R) {
      log_BF <- fit_L$log_evidence + fit_R$log_evidence - node$log_evidence
    }
    list(
      log_BF = log_BF, fit_L = fit_L, fit_R = fit_R,
      candidate_failed = failed_L || failed_R,
      n_failed_child_fits = as.integer(failed_L) + as.integer(failed_R),
      n_nonconverged_child_fits =
        as.integer(!is.na(fit_L$convergence) && fit_L$convergence != 0L) +
        as.integer(!is.na(fit_R$convergence) && fit_R$convergence != 0L),
      n_laplace_fallback_child_fits =
        as.integer(isTRUE(fit_L$laplace_fallback)) + as.integer(isTRUE(fit_R$laplace_fallback)),
      n_regularized_hessian_child_fits =
        as.integer(identical(fit_L$hessian_status, "eigenvalues_regularized")) +
        as.integer(identical(fit_R$hessian_status, "eigenvalues_regularized"))
    )
  }

  evals <- TVGP_parallel_lapply(
    seq_along(selected), eval_candidate,
    cores = parallel_cores,
    mc_preschedule = parallel_mc_preschedule
  )
  log_BFs <- vapply(evals, `[[`, numeric(1), "log_BF")
  fits <- lapply(evals, function(z) list(fit_L = z$fit_L, fit_R = z$fit_R))
  n_failed_candidates <- sum(vapply(evals, `[[`, logical(1), "candidate_failed"))
  n_failed_child_fits <- sum(vapply(evals, `[[`, integer(1), "n_failed_child_fits"))
  n_nonconverged_child_fits <- sum(vapply(
    evals, `[[`, integer(1), "n_nonconverged_child_fits"
  ))
  n_laplace_fallback_child_fits <- sum(vapply(
    evals, `[[`, integer(1), "n_laplace_fallback_child_fits"
  ))
  n_regularized_hessian_child_fits <- sum(vapply(
    evals, `[[`, integer(1), "n_regularized_hessian_child_fits"
  ))

  if (n_failed_candidates > 0L) {
    failed <- which(vapply(evals, `[[`, logical(1), "candidate_failed"))
    failure_rows <- do.call(rbind, unlist(lapply(failed, function(ii) {
      cand <- candidates[[selected[ii]]]
      fit_pair <- evals[[ii]]
      rows <- list()
      if (isTRUE(fit_pair$fit_L$optimization_failed) ||
          fit_pair$fit_L$log_evidence <= -1e90) {
        rows[[length(rows) + 1L]] <- TVGP_fit_failure_rows(
          fit_pair$fit_L, "candidate_child", node$key,
          candidate_position = ii, candidate_index = selected[ii],
          split_dim = cand$j, split_cut = cand$c,
          child = "left", child_n = length(cand$idx_L)
        )
      }
      if (isTRUE(fit_pair$fit_R$optimization_failed) ||
          fit_pair$fit_R$log_evidence <= -1e90) {
        rows[[length(rows) + 1L]] <- TVGP_fit_failure_rows(
          fit_pair$fit_R, "candidate_child", node$key,
          candidate_position = ii, candidate_index = selected[ii],
          split_dim = cand$j, split_cut = cand$c,
          child = "right", child_n = length(cand$idx_R)
        )
      }
      rows
    }), recursive = FALSE))
    diagnostic_path <- TVGP_write_failure_diagnostics(
      failure_rows, failure_diagnostics_file
    )
    failed_description <- vapply(failed, function(ii) {
      cand <- candidates[[selected[ii]]]
      failed_sides <- c(
        if (isTRUE(evals[[ii]]$fit_L$optimization_failed) ||
            evals[[ii]]$fit_L$log_evidence <= -1e90) "L" else NULL,
        if (isTRUE(evals[[ii]]$fit_R$optimization_failed) ||
            evals[[ii]]$fit_R$log_evidence <= -1e90) "R" else NULL
      )
      sprintf(
        "position=%d,index=%d,x%d<=%.10g,failed=%s",
        ii, selected[ii], cand$j, cand$c, paste(failed_sides, collapse = "+")
      )
    }, character(1))
    stop(sprintf(
      paste0(
        "%d of %d candidate splits failed after deterministic retries for ",
        "node %s (%s).%s"
      ),
      length(failed), length(evals), node$key,
      paste(failed_description, collapse = "; "),
      if (is.null(diagnostic_path)) "" else paste0(
        " Diagnostics: ", diagnostic_path
      )
    ))
  }

  if (all(!is.finite(log_BFs))) {
    stop(sprintf(
      paste0(
        "Every candidate split failed for node %s (depth %d; n=%d). ",
        "The SMC run was stopped because treating this numerical failure as a ",
        "terminal node would change the target distribution."
      ),
      node$key, node$depth, length(node$indices)
    ))
  }

  log_Z_split <- TVGP_logsumexp(log_prior + log_BFs)
  term_stay <- log1p(-rho)
  term_split <- log(rho) + log_Z_split
  log_w <- TVGP_logsumexp(c(term_stay, term_split))
  log_prob_split <- term_split - log_w

  list(
    can_split = TRUE,
    log_w = log_w,
    log_prob_split = log_prob_split,
    candidates = candidates,
    selected = selected,
    log_prior = log_prior,
    log_BFs = log_BFs,
    fits = fits,
    n_candidates = length(candidates),
    forced_stop_reason = NA_character_,
    n_failed_candidates = n_failed_candidates,
    n_failed_child_fits = n_failed_child_fits,
    n_nonconverged_child_fits = n_nonconverged_child_fits,
    n_laplace_fallback_child_fits = n_laplace_fallback_child_fits,
    n_regularized_hessian_child_fits = n_regularized_hessian_child_fits
  )
}

TVGP_next_active_leaf <- function(p, min_leaf_size) {
  leaf_idx <- which(vapply(p$nodes, function(z) {
    isTRUE(z$is_leaf) && !isTRUE(z$terminal) &&
      length(z$indices) >= 2L * min_leaf_size
  }, logical(1)))
  if (length(leaf_idx) == 0) return(NA_integer_)
  leaf_depths <- vapply(leaf_idx, function(idx) p$nodes[[idx]]$depth, numeric(1))
  leaf_idx[order(leaf_depths, leaf_idx)[1L]]
}

TVGP_apply_transition <- function(p, node_idx, trans, min_leaf_size) {
  if (is.na(node_idx)) {
    return(list(
      particle = p, split = FALSE, log_prob_split = -Inf,
      chosen_candidate = NA_integer_, chosen_dim = NA_integer_, chosen_cut = NA_real_,
      stop_reason = "already_complete", n_children_terminal_by_size = 0L
    ))
  }

  if (!isTRUE(trans$can_split)) {
    p$nodes[[node_idx]]$terminal <- TRUE
    p$nodes[[node_idx]]$terminal_reason <- trans$forced_stop_reason
    return(list(
      particle = p, split = FALSE, log_prob_split = trans$log_prob_split,
      chosen_candidate = NA_integer_, chosen_dim = NA_integer_, chosen_cut = NA_real_,
      stop_reason = trans$forced_stop_reason, n_children_terminal_by_size = 0L
    ))
  }

  if (log(runif(1)) >= trans$log_prob_split) {
    p$nodes[[node_idx]]$terminal <- TRUE
    p$nodes[[node_idx]]$terminal_reason <- "posterior_no_split"
    return(list(
      particle = p, split = FALSE, log_prob_split = trans$log_prob_split,
      chosen_candidate = NA_integer_, chosen_dim = NA_integer_, chosen_cut = NA_real_,
      stop_reason = "posterior_no_split", n_children_terminal_by_size = 0L
    ))
  }

  log_choice <- trans$log_prior + trans$log_BFs
  probs <- TVGP_normalize_log_weights(log_choice)
  chosen_local <- sample(seq_along(trans$selected), 1L, prob = probs)
  chosen_global <- trans$selected[chosen_local]
  cand <- trans$candidates[[chosen_global]]
  fit_L <- trans$fits[[chosen_local]]$fit_L
  fit_R <- trans$fits[[chosen_local]]$fit_R

  node <- p$nodes[[node_idx]]
  next_id <- length(p$nodes) + 1L
  child_depth <- node$depth + 1L
  terminal_L <- length(cand$idx_L) < 2L * min_leaf_size
  terminal_R <- length(cand$idx_R) < 2L * min_leaf_size

  p$nodes[[node_idx]]$is_leaf <- FALSE
  p$nodes[[node_idx]]$terminal <- FALSE
  p$nodes[[node_idx]]$split_dim <- cand$j
  p$nodes[[node_idx]]$split_val <- cand$c
  p$nodes[[node_idx]]$left <- next_id
  p$nodes[[node_idx]]$right <- next_id + 1L
  p$nodes[[node_idx]]$terminal_reason <- NA_character_

  p$nodes[[next_id]] <- TVGP_make_node(
    next_id, cand$idx_L, fit_L, child_depth,
    paste0(node$key, "L", cand$j, "@", signif(cand$c, 8)),
    terminal = terminal_L,
    terminal_reason = if (terminal_L) "insufficient_size_for_split" else NA_character_
  )
  p$nodes[[next_id + 1L]] <- TVGP_make_node(
    next_id + 1L, cand$idx_R, fit_R, child_depth,
    paste0(node$key, "R", cand$j, "@", signif(cand$c, 8)),
    terminal = terminal_R,
    terminal_reason = if (terminal_R) "insufficient_size_for_split" else NA_character_
  )

  # Internal-node memberships are recoverable from the children and are not
  # used by subsequent transitions or prediction. Releasing them keeps the
  # particle state proportional to the current leaves rather than tree depth.
  p$nodes[[node_idx]]$indices <- integer(0)
  p$nodes[[node_idx]]$ord_indices <- integer(0)

  list(
    particle = p, split = TRUE, log_prob_split = trans$log_prob_split,
    chosen_candidate = chosen_global, chosen_dim = cand$j, chosen_cut = cand$c,
    stop_reason = NA_character_,
    n_children_terminal_by_size = as.integer(terminal_L) + as.integer(terminal_R)
  )
}

# ------------------------------------------------------------------------------
# 3. SMC main algorithm
# ------------------------------------------------------------------------------
TVGP_init_particle <- function(root_fit, N, min_leaf_size) {
  root_terminal <- N < 2L * min_leaf_size
  root_node <- TVGP_make_node(
    1L, seq_len(N), root_fit, depth = 0L, key = "R",
    terminal = root_terminal,
    terminal_reason = if (root_terminal) "insufficient_size_for_split" else NA_character_
  )
  list(nodes = list(root_node), log_weight = 0)
}

TVGP_tree_log_marginal_likelihood <- function(p) {
  leaf_idx <- which(vapply(p$nodes, function(z) isTRUE(z$is_leaf), logical(1)))
  sum(vapply(leaf_idx, function(i) p$nodes[[i]]$log_evidence, numeric(1)))
}

TVGP_particle_stage_diagnostics <- function(step, particles, weights,
                                            active_before = NULL,
                                            split_flags = NULL,
                                            stop_reasons = NULL) {
  n_particles <- length(particles)
  if (is.null(active_before)) active_before <- rep(FALSE, n_particles)
  if (is.null(split_flags)) split_flags <- rep(FALSE, n_particles)
  if (is.null(stop_reasons)) stop_reasons <- rep(NA_character_, n_particles)
  data.frame(
    step = step,
    particle = seq_len(n_particles),
    active_before_transition = as.logical(active_before),
    split = as.logical(split_flags),
    stop_reason = as.character(stop_reasons),
    log_tree_marginal_likelihood = vapply(
      particles, TVGP_tree_log_marginal_likelihood, numeric(1)
    ),
    normalized_weight_pre_resampling = as.numeric(weights),
    n_nodes = vapply(particles, function(p) length(p$nodes), integer(1)),
    n_leaves = vapply(particles, function(p) {
      sum(vapply(p$nodes, function(z) isTRUE(z$is_leaf), logical(1)))
    }, integer(1)),
    max_depth = vapply(particles, function(p) {
      max(vapply(p$nodes, function(z) z$depth, numeric(1)))
    }, numeric(1)),
    stringsAsFactors = FALSE
  )
}

TVGP_SMC <- function(y, X, n_particles = 5000L,
                     nu = 1.5, m_vecchia = 30, alpha = 0.95, beta = 0.5,
                     n_candidates_per_axis = 10, min_leaf_size = m_vecchia,
                     max_full_candidates = NULL, full_maxit = 150,
                     root_maxit = 200, hessian = TRUE,
                     ordering = c("global", "input", "local"), nn_pool_factor = 8,
                     exact_nn_fallback = FALSE,
                     screen_with_fixed_theta = FALSE,
                     renormalize_candidate_prior = FALSE,
                     parallel_cores = 1L,
                     parallel_mc_preschedule = FALSE,
                     store_history_particles = FALSE,
                     store_step_particle_diagnostics = TRUE,
                     verbose = TRUE,
                     cov_prior = c("inverse_gamma", "jointly_robust"),
                     cov_prior_control = NULL,
                     leaf_prior = "normal_inverse_gamma",
                     leaf_prior_control = NULL,
                     optimizer_gradient_tol = 1e-3,
                     optimizer_boundary_tol = 1e-6,
                     optimizer_newton_maxit = 20L,
                     failure_diagnostics_prefix = NULL,
                     candidate_budget = 0L, prepared_context = NULL,
                     optimizer_factr = NULL, parent_warm_start = FALSE,
                     zero_fast_start = FALSE, node_cache_dir = NULL) {
  ordering <- match.arg(ordering)
  fit_started_elapsed <- proc.time()[["elapsed"]]
  cov_prior <- match.arg(cov_prior)
  leaf_prior <- match.arg(leaf_prior, "normal_inverse_gamma")
  parallel_cores <- as.integer(parallel_cores)
  if (is.na(parallel_cores) || parallel_cores < 1L) parallel_cores <- 1L
  compute_threads <- if (TVGP_parallel_backend() == "threads") parallel_cores else 1L
  old_threads <- TVGP_set_compute_threads(compute_threads)
  on.exit(TVGP_set_compute_threads(old_threads), add = TRUE)
  candidate_budget <- as.integer(candidate_budget)
  if (length(candidate_budget) != 1L || is.na(candidate_budget) ||
      (candidate_budget != 0L && candidate_budget < ncol(X))) {
    stop("candidate_budget must be zero or at least the number of input dimensions.")
  }
  if (candidate_budget > 0L && ((!is.null(max_full_candidates) && max_full_candidates > 0L) ||
      screen_with_fixed_theta || renormalize_candidate_prior)) {
    stop("candidate_budget fixes a design-only support; response-based screening cannot be combined with it.")
  }
  store_history_particles <- isTRUE(store_history_particles)
  store_step_particle_diagnostics <- isTRUE(store_step_particle_diagnostics)
  if (length(optimizer_gradient_tol) != 1L ||
      !is.finite(optimizer_gradient_tol) || optimizer_gradient_tol <= 0) {
    stop("optimizer_gradient_tol must be one positive finite value.")
  }
  if (length(optimizer_boundary_tol) != 1L ||
      !is.finite(optimizer_boundary_tol) || optimizer_boundary_tol <= 0) {
    stop("optimizer_boundary_tol must be one positive finite value.")
  }
  optimizer_newton_maxit <- as.integer(optimizer_newton_maxit)
  if (length(optimizer_newton_maxit) != 1L ||
      is.na(optimizer_newton_maxit) || optimizer_newton_maxit < 1L) {
    stop("optimizer_newton_maxit must be one positive integer.")
  }
  if (!is.null(failure_diagnostics_prefix)) {
    if (length(failure_diagnostics_prefix) != 1L ||
        is.na(failure_diagnostics_prefix) || !nzchar(failure_diagnostics_prefix)) {
      stop("failure_diagnostics_prefix must be NULL or one nonempty path prefix.")
    }
    failure_diagnostics_prefix <- path.expand(failure_diagnostics_prefix)
  }
  N <- nrow(X)
  if (length(y) != N) stop("length(y) must equal nrow(X)")
  min_leaf_size <- as.integer(min_leaf_size)
  if (is.na(min_leaf_size) || min_leaf_size < 1L) {
    stop("min_leaf_size must be a positive integer")
  }
  max_possible_leaves <- max(1L, floor(N / min_leaf_size))
  max_possible_nodes <- max(1L, 2L * max_possible_leaves - 1L)
  leaf_prior_spec <- TVGP_leaf_prior_spec(leaf_prior_control)

  if (is.null(prepared_context)) {
    ctx <- TVGP_prepare_context(
      X, m_vecchia, ordering = ordering, nn_pool_factor = nn_pool_factor,
      exact_nn_fallback = exact_nn_fallback, verbose = verbose)
  } else {
    ctx <- prepared_context
    stopifnot(identical(ctx$design_hash, digest::digest(X, algo = "sha256")),
      ctx$m == m_vecchia, ctx$ordering == ordering,
      ctx$nn_pool_factor == nn_pool_factor,
      identical(ctx$exact_nn_fallback, exact_nn_fallback))
  }
  ctx$candidate_budget <- candidate_budget
  # Storm-surge settings (Supplementary Section B.1): optimizer tolerance, parent warm
  # starts, a dedicated start for constant-response leaves, and a cache of fitted nodes.
  if (!is.null(optimizer_factr)) ctx$optimizer_factr <- optimizer_factr
  if (isTRUE(parent_warm_start)) ctx$parent_warm_start <- TRUE
  if (isTRUE(zero_fast_start)) ctx$zero_fast_start <- TRUE
  if (!is.null(node_cache_dir)) ctx <- TVGP_attach_fit_cache(ctx, y, X, node_cache_dir)

  if (verbose) cat("Optimizing root once...\n")
  root_fit <- TVGP_optimize_node_with_retries(
    y, X, seq_len(N), nu, m_vecchia, ctx,
    cov_prior = cov_prior, cov_prior_control = cov_prior_control,
    leaf_prior_spec = leaf_prior_spec,
    maxit = root_maxit, hessian = hessian,
    optimizer_gradient_tol = optimizer_gradient_tol,
    optimizer_boundary_tol = optimizer_boundary_tol,
    newton_maxit = optimizer_newton_maxit
  )
  if (isTRUE(root_fit$optimization_failed)) {
    root_failure_path <- if (is.null(failure_diagnostics_prefix)) {
      NULL
    } else {
      paste0(failure_diagnostics_prefix, "_root_optimization_failure.csv")
    }
    diagnostic_path <- TVGP_write_failure_diagnostics(
      TVGP_fit_failure_rows(
        root_fit, stage = "root", node_key = "R", child = "root",
        child_n = N
      ),
      root_failure_path
    )
    stop(sprintf(
      "Root optimization failed after %d deterministic attempts (%s): %s%s",
      root_fit$optimization_attempts,
      root_fit$optimization_failure_reason,
      ifelse(is.na(root_fit$optimizer_message), "no optimizer message", root_fit$optimizer_message),
      if (is.null(diagnostic_path)) "" else paste0(". Diagnostics: ", diagnostic_path)
    ))
  }
  particles <- replicate(
    n_particles, TVGP_init_particle(root_fit, N, min_leaf_size), simplify = FALSE
  )

  history_weights <- list(rep(1 / n_particles, n_particles))
  history_particles <- if (store_history_particles) list(particles) else NULL
  diagnostics <- list()
  particle_diagnostics <- if (store_step_particle_diagnostics) {
    list(TVGP_particle_stage_diagnostics(
      0L, particles, rep(1 / n_particles, n_particles)
    ))
  } else NULL
  termination_reason <- NA_character_
  t <- 0L

  repeat {
    node_idx <- vapply(
      particles, TVGP_next_active_leaf, integer(1), min_leaf_size = min_leaf_size
    )
    active <- which(!is.na(node_idx))
    if (length(active) == 0) {
      termination_reason <- "all_particles_complete"
      if (verbose) cat("All particles have completed tree construction.\n")
      break
    }
    if (t >= max_possible_nodes) {
      termination_reason <- "internal_node_bound_violation"
      stop(sprintf(
        paste0(
          "SMC still has active leaves after %d stages, exceeding the finite-tree ",
          "bound implied by n=%d and min_leaf_size=%d."
        ),
        t, N, min_leaf_size
      ))
    }
    t <- t + 1L

    keys <- vapply(active, function(i) particles[[i]]$nodes[[node_idx[i]]]$key, character(1))
    unique_keys <- unique(keys)
    if (verbose) {
      cat(sprintf(
        "SMC step %d: %d active particles, %d unique nodes\n",
        t, length(active), length(unique_keys)
      ))
    }

    transition_elapsed <- system.time({
      transitions <- vector("list", length(unique_keys))
      names(transitions) <- unique_keys
      for (kk in seq_along(unique_keys)) {
        first_i <- active[which(keys == unique_keys[kk])[1L]]
        node <- particles[[first_i]]$nodes[[node_idx[first_i]]]
        transitions[[kk]] <- TVGP_evaluate_transition(
          node, y, X, ctx, nu, m_vecchia,
          cov_prior, cov_prior_control, leaf_prior_spec, alpha, beta,
          n_candidates_per_axis, min_leaf_size, max_full_candidates,
          full_maxit, screen_with_fixed_theta, renormalize_candidate_prior,
          hessian, parallel_cores = parallel_cores,
          parallel_mc_preschedule = parallel_mc_preschedule,
          optimizer_gradient_tol = optimizer_gradient_tol,
          optimizer_boundary_tol = optimizer_boundary_tol,
          optimizer_newton_maxit = optimizer_newton_maxit,
          failure_diagnostics_file = if (is.null(failure_diagnostics_prefix)) {
            NULL
          } else {
            paste0(failure_diagnostics_prefix, "_candidate_optimization_failure.csv")
          }
        )
      }
    })["elapsed"]

    applied <- vector("list", n_particles)
    log_incr_weights <- rep(0, n_particles)
    split_flags <- rep(FALSE, n_particles)
    log_prob_split <- rep(-Inf, n_particles)
    chosen_dim <- rep(NA_integer_, n_particles)
    chosen_cut <- rep(NA_real_, n_particles)
    stop_reasons <- rep(NA_character_, n_particles)
    children_terminal_by_size <- integer(n_particles)

    for (i in seq_len(n_particles)) {
      if (is.na(node_idx[i])) {
        applied[[i]] <- list(
          particle = particles[[i]], split = FALSE, log_prob_split = -Inf,
          chosen_candidate = NA_integer_, chosen_dim = NA_integer_, chosen_cut = NA_real_,
          stop_reason = "already_complete", n_children_terminal_by_size = 0L
        )
      } else {
        key <- particles[[i]]$nodes[[node_idx[i]]]$key
        trans <- transitions[[key]]
        applied[[i]] <- TVGP_apply_transition(
          particles[[i]], node_idx[i], trans, min_leaf_size
        )
        log_incr_weights[i] <- trans$log_w
      }
      split_flags[i] <- isTRUE(applied[[i]]$split)
      log_prob_split[i] <- applied[[i]]$log_prob_split
      chosen_dim[i] <- applied[[i]]$chosen_dim
      chosen_cut[i] <- applied[[i]]$chosen_cut
      stop_reasons[i] <- applied[[i]]$stop_reason
      children_terminal_by_size[i] <- applied[[i]]$n_children_terminal_by_size
    }

    particles <- lapply(applied, `[[`, "particle")
    prev_log_weights <- log(history_weights[[length(history_weights)]])
    norm_weights <- TVGP_normalize_log_weights(prev_log_weights + log_incr_weights)
    ESS <- 1 / sum(norm_weights^2)
    tree_log_marginal <- vapply(
      particles, TVGP_tree_log_marginal_likelihood, numeric(1)
    )
    tree_signatures <- vapply(particles, TVGP_tree_signature, character(1))
    chosen_signature <- if (any(split_flags)) {
      paste(chosen_dim[split_flags], signif(chosen_cut[split_flags], 10), sep = "@")
    } else {
      character(0)
    }
    n_selected_total <- sum(vapply(transitions, function(z) length(z$selected), integer(1)))
    n_candidates_total <- sum(vapply(transitions, function(z) z$n_candidates, integer(1)))
    split_probs <- exp(log_prob_split[is.finite(log_prob_split)])
    weighted_log_marginal <- sum(norm_weights * tree_log_marginal)
    weighted_log_marginal_sd <- sqrt(sum(
      norm_weights * (tree_log_marginal - weighted_log_marginal)^2
    ))

    diagnostics[[t]] <- list(
      step = t,
      ESS = ESS,
      max_particle_weight = max(norm_weights),
      cumulative_elapsed_s = unname(proc.time()[["elapsed"]] - fit_started_elapsed),
      n_active = length(active),
      n_unique_nodes = length(unique_keys),
      n_unique_trees = length(unique(tree_signatures)),
      n_splits = sum(split_flags),
      n_stops_posterior_no_split = sum(stop_reasons == "posterior_no_split", na.rm = TRUE),
      n_stops_no_valid_split = sum(stop_reasons == "no_valid_split", na.rm = TRUE),
      n_stops_all_candidate_optimizations_failed = sum(
        stop_reasons == "all_candidate_optimizations_failed", na.rm = TRUE
      ),
      n_children_terminal_by_size = sum(children_terminal_by_size),
      mean_split_prob = if (length(split_probs)) mean(split_probs) else NA_real_,
      max_split_prob = if (length(split_probs)) max(split_probs) else NA_real_,
      mean_log_incr_weight = mean(log_incr_weights),
      mean_log_tree_marginal_likelihood = mean(tree_log_marginal),
      sd_log_tree_marginal_likelihood = stats::sd(tree_log_marginal),
      min_log_tree_marginal_likelihood = min(tree_log_marginal),
      max_log_tree_marginal_likelihood = max(tree_log_marginal),
      weighted_mean_log_tree_marginal_likelihood = weighted_log_marginal,
      weighted_sd_log_tree_marginal_likelihood = weighted_log_marginal_sd,
      n_candidate_splits_total = n_candidates_total,
      n_selected_candidates_total = n_selected_total,
      n_failed_candidates = sum(vapply(
        transitions, `[[`, integer(1), "n_failed_candidates"
      )),
      n_failed_child_fits = sum(vapply(
        transitions, `[[`, integer(1), "n_failed_child_fits"
      )),
      n_nonconverged_child_fits = sum(vapply(
        transitions, `[[`, integer(1), "n_nonconverged_child_fits"
      )),
      n_laplace_fallback_child_fits = sum(vapply(
        transitions, `[[`, integer(1), "n_laplace_fallback_child_fits"
      )),
      n_regularized_hessian_child_fits = sum(vapply(
        transitions, `[[`, integer(1), "n_regularized_hessian_child_fits"
      )),
      n_unique_chosen_candidates = length(unique(chosen_signature)),
      transition_elapsed_s = as.numeric(transition_elapsed),
      parallel_cores = parallel_cores,
      resampled = FALSE,
      n_unique_trees_after_resampling = length(unique(tree_signatures))
    )

    if (verbose) {
      cat(sprintf(
        paste0(
          "  splits=%d, ESS=%.1f, mean P(split)=%.3f, ",
          "optimization-forced stops=%d, weighted log ML=%.3f\n"
        ),
        diagnostics[[t]]$n_splits, ESS,
        ifelse(is.na(diagnostics[[t]]$mean_split_prob), 0, diagnostics[[t]]$mean_split_prob),
        diagnostics[[t]]$n_stops_all_candidate_optimizations_failed,
        diagnostics[[t]]$weighted_mean_log_tree_marginal_likelihood
      ))
    }

    history_weights[[t + 1L]] <- norm_weights
    if (store_history_particles) history_particles[[t + 1L]] <- particles
    if (store_step_particle_diagnostics) {
      active_before <- seq_len(n_particles) %in% active
      particle_diagnostics[[t + 1L]] <- TVGP_particle_stage_diagnostics(
        t, particles, norm_weights,
        active_before = active_before,
        split_flags = split_flags,
        stop_reasons = stop_reasons
      )
    }

    if (ESS < n_particles / 2) {
      idx <- sample(seq_len(n_particles), n_particles, replace = TRUE, prob = norm_weights)
      particles <- particles[idx]
      history_weights[[t + 1L]] <- rep(1 / n_particles, n_particles)
      if (store_history_particles) history_particles[[t + 1L]] <- particles
      diagnostics[[t]]$resampled <- TRUE
      diagnostics[[t]]$n_unique_trees_after_resampling <- length(unique(vapply(
        particles, TVGP_tree_signature, character(1)
      )))
      if (verbose) cat("  resampled particles\n")
    }
    rm(transitions, applied)
    invisible(gc(verbose = FALSE))
  }

  termination_diagnostics <- data.frame(
    termination_reason = termination_reason,
    n_completed_smc_steps = t,
    finite_tree_node_bound = max_possible_nodes,
    max_possible_leaves = max_possible_leaves,
    min_leaf_size = min_leaf_size,
    root_optimizer_convergence = root_fit$convergence,
    root_optimization_attempts = root_fit$optimization_attempts,
    root_optimization_attempt_label = root_fit$optimization_attempt_label,
    root_previous_failure_reasons = root_fit$previous_failure_reasons,
    root_optimization_failed = isTRUE(root_fit$optimization_failed),
    root_optimization_failure_reason = root_fit$optimization_failure_reason,
    root_optimizer_message = root_fit$optimizer_message,
    root_gradient_norm = root_fit$gradient_norm,
    root_gradient_inf_norm = root_fit$gradient_inf_norm,
    root_minimum_bound_distance = root_fit$minimum_bound_distance,
    root_mode_on_boundary = root_fit$mode_on_boundary,
    root_initial_optimizer_convergence = root_fit$initial_optimizer_convergence,
    root_initial_optimizer_message = root_fit$initial_optimizer_message,
    root_newton_converged = root_fit$newton_converged,
    root_newton_iterations = root_fit$newton_iterations,
    root_newton_accepted_steps = root_fit$newton_accepted_steps,
    root_newton_damping_steps = root_fit$newton_damping_steps,
    root_newton_line_search_evaluations = root_fit$newton_line_search_evaluations,
    root_newton_failure_reason = root_fit$newton_failure_reason,
    optimizer_gradient_tol = optimizer_gradient_tol,
    optimizer_boundary_tol = optimizer_boundary_tol,
    root_hessian_status = root_fit$hessian_status,
    root_hessian_n_regularized = root_fit$hessian_n_regularized,
    root_hessian_min_eigenvalue = root_fit$hessian_min_eigenvalue,
    root_hessian_condition_number = root_fit$hessian_condition_number,
    root_hessian_near_singular = root_fit$hessian_near_singular,
    root_laplace_fallback = isTRUE(root_fit$laplace_fallback),
    stringsAsFactors = FALSE
  )

  list(
    final_particles = particles,
    history_weights = history_weights,
    history_particles = history_particles,
    diagnostics = diagnostics,
    particle_diagnostics = particle_diagnostics,
    termination_diagnostics = termination_diagnostics,
    root_fit_diagnostics = root_fit,
    candidate_budget = candidate_budget,
    compute_threads = compute_threads,
    parallel_backend = TVGP_parallel_backend(),
    node_cache_statistics = TVGP_cache_stats(ctx),
    leaf_prior = leaf_prior,
    leaf_prior_control = leaf_prior_spec$details,
    cov_prior = cov_prior,
    cov_prior_control = cov_prior_control,
    context = ctx
  )
}

# ------------------------------------------------------------------------------
# 4. Prediction functions
# ------------------------------------------------------------------------------
find_leaf_id <- function(x_row, nodes, current_id = 1L) {
  node <- nodes[[current_id]]
  if (isTRUE(node$is_leaf)) return(current_id)
  if (x_row[node$split_dim] <= node$split_val) {
    find_leaf_id(x_row, nodes, node$left)
  } else {
    find_leaf_id(x_row, nodes, node$right)
  }
}

TVGP_predict_particle_pointwise <- function(p, X_train, y_train, X_test,
                                            nu = 1.5, m_vecchia = 30,
                                            predict_obs = FALSE) {
  n_test <- nrow(X_test)
  mu <- numeric(n_test)
  variance <- numeric(n_test)
  scale2 <- numeric(n_test)
  df <- numeric(n_test)
  nodes <- p$nodes
  leaf_ids <- vapply(seq_len(n_test), function(k) find_leaf_id(X_test[k, ], nodes, 1L), integer(1))
  for (lid in unique(leaf_ids)) {
    test_idx <- which(leaf_ids == lid)
    leaf <- nodes[[lid]]
    train_idx <- leaf$ord_indices

    res <- vecchia_prediction_moments(
      y_train[train_idx],
      X_train[train_idx, , drop = FALSE],
      X_test[test_idx, , drop = FALSE],
      leaf$theta_hat,
      leaf$gamma_mean,
      leaf$gamma_var_factor,
      leaf$sigma2_shape,
      leaf$sigma2_scale,
      nu, m_vecchia,
      predict_obs
    )
    mu[test_idx] <- res$mu
    variance[test_idx] <- res$variance
    scale2[test_idx] <- res$scale2
    df[test_idx] <- res$df
  }
  list(
    mu = mu, variance = variance, scale2 = scale2, df = df,
    prediction_target = if (isTRUE(predict_obs)) "observation" else "latent_function",
    prediction_mode = "pointwise"
  )
}

TVGP_predict_particle_joint <- function(p, X_train, y_train, X_test,
                                        nu = 1.5, m_vecchia = 30,
                                        predict_obs = FALSE,
                                        joint_ordering = c("maximin", "input"),
                                        return_covariance = TRUE) {
  joint_ordering <- match.arg(joint_ordering)
  n_test <- nrow(X_test)
  mu <- numeric(n_test)
  covariance <- if (isTRUE(return_covariance)) matrix(0, n_test, n_test) else NULL
  blocks <- list()
  nodes <- p$nodes
  leaf_ids <- vapply(
    seq_len(n_test),
    function(k) find_leaf_id(X_test[k, ], nodes, 1L),
    integer(1)
  )

  for (lid in unique(leaf_ids)) {
    test_idx <- which(leaf_ids == lid)
    leaf <- nodes[[lid]]
    train_idx <- leaf$ord_indices
    local_order <- if (joint_ordering == "maximin" && length(test_idx) > 1L) {
      as.integer(get_maximin_order(X_test[test_idx, , drop = FALSE])) + 1L
    } else {
      seq_along(test_idx)
    }
    ordered_test_idx <- test_idx[local_order]

    res <- tryCatch(vecchia_prediction_joint(
      y_train[train_idx],
      X_train[train_idx, , drop = FALSE],
      X_test[ordered_test_idx, , drop = FALSE],
      leaf$theta_hat,
      leaf$gamma_mean,
      leaf$gamma_var_factor,
      leaf$sigma2_shape,
      leaf$sigma2_scale,
      nu, m_vecchia,
      predict_obs,
      return_covariance
    ), error = function(e) stop(sprintf(
      "Joint prediction in leaf %s (training=%d, test=%d): %s",
      leaf$key, length(train_idx), length(ordered_test_idx), conditionMessage(e)
    ), call. = FALSE))

    parent_index <- as.matrix(res$test_parent_index)
    if (length(parent_index)) {
      valid_parent <- parent_index >= 0L
      parent_index[valid_parent] <- parent_index[valid_parent] + 1L
      parent_index[!valid_parent] <- 0L
      storage.mode(parent_index) <- "integer"
    }
    block <- list(
      leaf_id = lid,
      test_idx = ordered_test_idx,
      mu = as.numeric(res$mu),
      D_star = as.numeric(res$D_star),
      mean_loading = as.numeric(res$mean_loading),
      test_parent_index = parent_index,
      test_parent_coefficient = as.matrix(res$test_parent_coefficient),
      test_parent_count = as.integer(res$test_parent_count),
      gamma_var_factor = as.numeric(res$gamma_var_factor),
      scale_factor = as.numeric(res$scale_factor),
      variance_factor = as.numeric(res$variance_factor),
      df = as.numeric(res$df),
      scale_matrix = if (isTRUE(return_covariance)) as.matrix(res$scale_matrix) else NULL,
      covariance = if (isTRUE(return_covariance)) as.matrix(res$covariance) else NULL
    )
    blocks[[length(blocks) + 1L]] <- block
    mu[ordered_test_idx] <- block$mu
    if (isTRUE(return_covariance)) {
      covariance[ordered_test_idx, ordered_test_idx] <- block$covariance
    }
  }

  list(
    mu = mu,
    variance = if (isTRUE(return_covariance)) diag(covariance) else NULL,
    covariance = covariance,
    blocks = blocks,
    prediction_target = if (isTRUE(predict_obs)) "observation" else "latent_function",
    prediction_mode = "joint",
    joint_ordering = joint_ordering
  )
}

TVGP_predict_particle <- function(p, X_train, y_train, X_test,
                                  nu = 1.5, m_vecchia = 30,
                                  predict_obs = FALSE,
                                  prediction_mode = c("pointwise", "joint"),
                                  joint_ordering = c("maximin", "input"),
                                  return_covariance = NULL) {
  prediction_mode <- match.arg(prediction_mode)
  joint_ordering <- match.arg(joint_ordering)
  if (prediction_mode == "pointwise") {
    return(TVGP_predict_particle_pointwise(
      p, X_train, y_train, X_test,
      nu = nu, m_vecchia = m_vecchia, predict_obs = predict_obs
    ))
  }
  if (is.null(return_covariance)) return_covariance <- TRUE
  TVGP_predict_particle_joint(
    p, X_train, y_train, X_test,
    nu = nu, m_vecchia = m_vecchia, predict_obs = predict_obs,
    joint_ordering = joint_ordering,
    return_covariance = return_covariance
  )
}

TVGP_sample_t_mixture <- function(mu_mat, scale2_mat, df_mat, weights,
                                  n_draws, seed = NULL) {
  n_draws <- as.integer(n_draws)
  if (is.na(n_draws) || n_draws < 1L) return(NULL)
  if (!is.null(seed)) set.seed(seed)
  n_test <- nrow(mu_mat)
  n_components <- ncol(mu_mat)
  weights <- as.numeric(weights) / sum(weights)
  component <- matrix(
    sample.int(
      n_components, n_test * n_draws, replace = TRUE, prob = weights
    ),
    nrow = n_test, ncol = n_draws
  )
  row_index <- rep(seq_len(n_test), times = n_draws)
  matrix_index <- cbind(row_index, as.vector(component))
  location <- mu_mat[matrix_index]
  scale <- sqrt(pmax(scale2_mat[matrix_index], 0))
  degrees_freedom <- df_mat[matrix_index]
  matrix(
    location + scale * stats::rt(length(location), df = degrees_freedom),
    nrow = n_test, ncol = n_draws
  )
}

TVGP_sample_joint_leaf_block <- function(block, n_draws) {
  n_draws <- as.integer(n_draws)
  n_test <- length(block$mu)
  if (n_draws < 1L) return(matrix(numeric(), nrow = n_test, ncol = 0L))

  innovations <- matrix(rnorm(n_test * n_draws), nrow = n_test)
  for (i in seq_len(n_test)) {
    innovations[i, ] <- sqrt(max(block$D_star[i], 0)) * innovations[i, ]
    count <- block$test_parent_count[i]
    if (count > 0L) {
      for (r in seq_len(count)) {
        previous <- block$test_parent_index[i, r]
        innovations[i, ] <- innovations[i, ] +
          block$test_parent_coefficient[i, r] * innovations[previous, ]
      }
    }
  }

  gamma_part <- outer(
    block$mean_loading * sqrt(block$gamma_var_factor),
    rnorm(n_draws)
  )
  radial_scale <- sqrt(block$scale_factor) *
    sqrt(block$df / stats::rchisq(n_draws, df = block$df))
  centered <- sweep(innovations + gamma_part, 2L, radial_scale, "*")
  sweep(centered, 1L, block$mu, "+")
}

TVGP_sample_joint_component <- function(prediction, n_draws) {
  n_test <- length(prediction$mu)
  out <- matrix(NA_real_, nrow = n_test, ncol = n_draws)
  for (block in prediction$blocks) {
    out[block$test_idx, ] <- TVGP_sample_joint_leaf_block(block, n_draws)
  }
  out
}

TVGP_sample_joint_t_mixture <- function(pred_list, weights, n_draws,
                                        seed = NULL) {
  n_draws <- as.integer(n_draws)
  if (is.na(n_draws) || n_draws < 1L) {
    return(list(draws = NULL, tree_component = integer()))
  }
  if (!is.null(seed)) set.seed(seed)
  weights <- as.numeric(weights) / sum(weights)
  tree_component <- sample.int(
    length(pred_list), n_draws, replace = TRUE, prob = weights
  )
  n_test <- length(pred_list[[1L]]$mu)
  draws <- matrix(NA_real_, nrow = n_test, ncol = n_draws)
  for (component in unique(tree_component)) {
    draw_idx <- which(tree_component == component)
    draws[, draw_idx] <- TVGP_sample_joint_component(
      pred_list[[component]], length(draw_idx)
    )
  }
  list(draws = draws, tree_component = tree_component)
}

TVGP_combine_predictive_components <- function(pred_list, weights,
                                                n_draws = 0L,
                                                draw_seed = NULL) {
  n_test <- length(pred_list[[1L]]$mu)
  mu_mat <- vapply(pred_list, `[[`, numeric(n_test), "mu")
  var_mat <- vapply(pred_list, `[[`, numeric(n_test), "variance")
  scale2_mat <- vapply(pred_list, `[[`, numeric(n_test), "scale2")
  df_mat <- vapply(pred_list, `[[`, numeric(n_test), "df")
  if (is.null(dim(mu_mat))) mu_mat <- matrix(mu_mat, nrow = n_test)
  if (is.null(dim(var_mat))) var_mat <- matrix(var_mat, nrow = n_test)
  if (is.null(dim(scale2_mat))) scale2_mat <- matrix(scale2_mat, nrow = n_test)
  if (is.null(dim(df_mat))) df_mat <- matrix(df_mat, nrow = n_test)

  weights <- as.numeric(weights) / sum(weights)
  global_mean <- as.vector(mu_mat %*% weights)
  within_component <- as.vector(var_mat %*% weights)
  between_component <- numeric(n_test)
  for (i in seq_along(weights)) {
    between_component <- between_component +
      weights[i] * (mu_mat[, i] - global_mean)^2
  }

  list(
    mean = global_mean,
    variance = within_component + between_component,
    draws = TVGP_sample_t_mixture(
      mu_mat, scale2_mat, df_mat, weights,
      n_draws = n_draws, seed = draw_seed
    )
  )
}

TVGP_combine_joint_predictive_components <- function(pred_list, weights,
                                                      n_draws = 0L,
                                                      draw_seed = NULL,
                                                      return_covariance = TRUE) {
  n_test <- length(pred_list[[1L]]$mu)
  weights <- as.numeric(weights) / sum(weights)
  mu_mat <- vapply(pred_list, `[[`, numeric(n_test), "mu")
  if (is.null(dim(mu_mat))) mu_mat <- matrix(mu_mat, nrow = n_test)
  global_mean <- as.vector(mu_mat %*% weights)

  global_covariance <- NULL
  global_variance <- NULL
  if (isTRUE(return_covariance)) {
    global_covariance <- matrix(0, n_test, n_test)
    for (i in seq_along(weights)) {
      delta <- pred_list[[i]]$mu - global_mean
      global_covariance <- global_covariance + weights[i] *
        (pred_list[[i]]$covariance + tcrossprod(delta))
    }
    global_covariance <- 0.5 * (global_covariance + t(global_covariance))
    global_variance <- diag(global_covariance)
  }

  sampled <- TVGP_sample_joint_t_mixture(
    pred_list, weights, n_draws = n_draws, seed = draw_seed
  )
  list(
    mean = global_mean,
    variance = global_variance,
    covariance = global_covariance,
    draws = sampled$draws,
    draw_tree_component = sampled$tree_component
  )
}

# Stream sparse joint components; the tree and leaf scales are shared over
# the entire test vector in each draw, including across processing batches.
TVGP_predict_joint_batched <- function(indices, weights, predict_component,
                                       n_test, n_draws, draw_seed,
                                       batch_size = 16L, parallel_cores = 1L,
                                       mc_preschedule = FALSE) {
  batch_size <- as.integer(batch_size)
  n_draws <- as.integer(n_draws)
  if (is.na(batch_size) || batch_size < 1L) stop("joint_batch_size must be positive.")
  if (is.na(n_draws) || n_draws < 0L) stop("n_draws must be nonnegative.")
  weights <- as.numeric(weights) / sum(weights)
  if (!is.null(draw_seed)) set.seed(draw_seed)
  tree_component <- if (n_draws > 0L) {
    sample.int(length(indices), n_draws, replace = TRUE, prob = weights)
  } else integer()
  draws <- if (n_draws > 0L) matrix(NA_real_, n_test, n_draws) else NULL
  global_mean <- numeric(n_test)
  # Match the unbatched sampler's component order and random-number stream.
  component_order <- c(unique(tree_component),
                       setdiff(seq_along(indices), tree_component))
  for (start in seq.int(1L, length(component_order), by = batch_size)) {
    components <- component_order[seq.int(start, min(length(component_order),
                                                      start + batch_size - 1L))]
    predictions <- TVGP_parallel_lapply(
      components, function(k) predict_component(indices[k]),
      cores = parallel_cores, mc_preschedule = mc_preschedule
    )
    for (i in seq_along(components)) {
      k <- components[i]
      prediction <- predictions[[i]]
      if (inherits(prediction, "try-error")) {
        stop(sprintf("Joint prediction failed for tree component %d: %s", k,
                     as.character(prediction)), call. = FALSE)
      }
      if (!is.list(prediction) ||
          length(prediction$mu) != n_test || any(!is.finite(prediction$mu))) {
        stop(sprintf("Joint prediction failed for tree component %d.", k))
      }
      global_mean <- global_mean + weights[k] * prediction$mu
      draw_idx <- which(tree_component == k)
      if (length(draw_idx)) {
        draws[, draw_idx] <- TVGP_sample_joint_component(prediction, length(draw_idx))
      }
    }
    rm(predictions, prediction)
  }
  list(mean = global_mean, variance = NULL, covariance = NULL,
       draws = draws, draw_tree_component = tree_component)
}

TVGP_predict <- function(smc_result, X_train, y_train, X_test,
                         nu = 1.5, m_vecchia = 30, predict_obs = FALSE,
                         n_draws = 0L, draw_seed = NULL,
                         cache_identical_trees = TRUE,
                         parallel_cores = 1L,
                         parallel_mc_preschedule = FALSE,
                         prediction_mode = c("pointwise", "joint"),
                         joint_ordering = c("maximin", "input"),
                         return_covariance = NULL,
                         joint_covariance_max_points = 100L,
                         joint_batch_size = 16L) {
  prediction_mode <- match.arg(prediction_mode)
  joint_ordering <- match.arg(joint_ordering)
  if (is.null(return_covariance)) return_covariance <- prediction_mode == "joint"
  particles <- smc_result$final_particles
  weights <- smc_result$history_weights[[length(smc_result$history_weights)]]
  n_test <- nrow(X_test)
  n_particles <- length(particles)

  if (prediction_mode == "joint" && isTRUE(return_covariance) &&
      n_test > as.integer(joint_covariance_max_points)) {
    stop(sprintf(
      paste0(
        "Joint covariance for %d prediction inputs exceeds ",
        "joint_covariance_max_points=%d. Increase the limit explicitly, ",
        "or set return_covariance=FALSE and request joint draws only."
      ),
      n_test, as.integer(joint_covariance_max_points)
    ))
  }

  predict_component <- function(particle) {
    TVGP_predict_particle(
      particle, X_train, y_train, X_test,
      nu = nu, m_vecchia = m_vecchia, predict_obs = predict_obs,
      prediction_mode = prediction_mode,
      joint_ordering = joint_ordering,
      return_covariance = return_covariance
    )
  }

  combine_components <- function(pred_list, component_weights) {
    if (prediction_mode == "pointwise") {
      return(TVGP_combine_predictive_components(
        pred_list, component_weights,
        n_draws = n_draws, draw_seed = draw_seed
      ))
    }
    TVGP_combine_joint_predictive_components(
      pred_list, component_weights,
      n_draws = n_draws, draw_seed = draw_seed,
      return_covariance = return_covariance
    )
  }

  predict_components <- function(indices, component_weights) {
    if (prediction_mode == "joint" && !isTRUE(return_covariance)) {
      return(TVGP_predict_joint_batched(
        indices, component_weights,
        function(i) predict_component(particles[[i]]),
        n_test, n_draws, draw_seed,
        batch_size = joint_batch_size, parallel_cores = parallel_cores,
        mc_preschedule = parallel_mc_preschedule
      ))
    }
    pred_list <- TVGP_parallel_lapply(
      indices, function(i) predict_component(particles[[i]]),
      cores = parallel_cores, mc_preschedule = parallel_mc_preschedule
    )
    combine_components(pred_list, component_weights)
  }

  cat("Predicting...\n")
  if (isTRUE(cache_identical_trees)) {
    signatures <- vapply(
      particles, TVGP_tree_signature, character(1),
      include_params = TRUE
    )
    groups <- split(seq_len(n_particles), signatures)
    reps <- vapply(groups, `[`, integer(1), 1L)
    group_weights <- vapply(groups, function(ii) sum(weights[ii]), numeric(1))
    group_weights <- group_weights / sum(group_weights)

    combined <- predict_components(reps, group_weights)
    return(c(combined, list(
      n_prediction_trees = length(groups),
      n_particles = n_particles,
      predictive_distribution = if (prediction_mode == "joint") {
        "tree mixture of leafwise multivariate Student-t distributions"
      } else {
        "pointwise posterior mixture of univariate Student-t distributions"
      },
      prediction_target = if (isTRUE(predict_obs)) "observation" else "latent_function",
      prediction_mode = prediction_mode,
      joint_ordering = if (prediction_mode == "joint") joint_ordering else NULL
    )))
  }

  combined <- predict_components(seq_len(n_particles), weights)
  c(combined, list(
    n_prediction_trees = n_particles,
    n_particles = n_particles,
    predictive_distribution = if (prediction_mode == "joint") {
      "tree mixture of leafwise multivariate Student-t distributions"
    } else {
      "pointwise posterior mixture of univariate Student-t distributions"
    },
    prediction_target = if (isTRUE(predict_obs)) "observation" else "latent_function",
    prediction_mode = prediction_mode,
    joint_ordering = if (prediction_mode == "joint") joint_ordering else NULL
  ))
}

TVGP_leaf_covariance_summary <- function(smc_result, y_train = NULL,
                                         input_names = NULL) {
  particles <- smc_result$final_particles
  weights <- smc_result$history_weights[[length(smc_result$history_weights)]]
  particle_signatures <- vapply(particles, TVGP_tree_signature, character(1))
  unique_signatures <- unique(particle_signatures)
  tree_ids <- match(particle_signatures, unique_signatures)
  if (is.null(input_names)) {
    p_dim <- length(particles[[1]]$nodes[[1]]$theta_hat) - 1L
    input_names <- paste0("x", seq_len(p_dim))
  }
  phi_names <- paste0("phi_", make.names(input_names))
  beta_names <- paste0("inverse_range_", make.names(input_names))
  jr_C_names <- paste0("jr_C_", make.names(input_names))
  jr_influence_names <- paste0("jr_influence_", make.names(input_names))

  rows <- list()
  row_id <- 0L
  for (i in seq_along(particles)) {
    p <- particles[[i]]
    leaf_ids <- which(vapply(p$nodes, function(z) isTRUE(z$is_leaf), logical(1)))
    for (lid in leaf_ids) {
      leaf <- p$nodes[[lid]]
      theta <- as.numeric(leaf$theta_hat)
      phi <- exp(theta[seq_len(length(theta) - 1L)])
      row <- list(
        particle = i,
        particle_weight = weights[i],
        tree_id = tree_ids[i],
        leaf_id = lid,
        leaf_key = leaf$key,
        depth = leaf$depth,
        n_train = TVGP_node_n_train(leaf),
        terminal = isTRUE(leaf$terminal),
        terminal_reason = leaf$terminal_reason,
        optimization_failed = isTRUE(leaf$optimization_failed),
        optimization_failure_reason = leaf$optimization_failure_reason,
        optimizer_convergence = leaf$optimizer_convergence,
        optimization_attempts = leaf$optimization_attempts,
        optimization_attempt_label = leaf$optimization_attempt_label,
        previous_failure_reasons = leaf$previous_failure_reasons,
        gradient_norm = leaf$gradient_norm,
        gradient_inf_norm = leaf$gradient_inf_norm,
        minimum_bound_distance = leaf$minimum_bound_distance,
        mode_on_boundary = leaf$mode_on_boundary,
        initial_optimizer_convergence = leaf$initial_optimizer_convergence,
        initial_optimizer_message = leaf$initial_optimizer_message,
        newton_converged = leaf$newton_converged,
        newton_iterations = leaf$newton_iterations,
        newton_accepted_steps = leaf$newton_accepted_steps,
        newton_damping_steps = leaf$newton_damping_steps,
        newton_line_search_evaluations = leaf$newton_line_search_evaluations,
        newton_failure_reason = leaf$newton_failure_reason,
        optimizer_gradient_tol = leaf$optimizer_gradient_tol,
        optimizer_boundary_tol = leaf$optimizer_boundary_tol,
        hessian_status = leaf$hessian_status,
        hessian_n_regularized = leaf$hessian_n_regularized,
        hessian_min_eigenvalue = leaf$hessian_min_eigenvalue,
        hessian_condition_number = leaf$hessian_condition_number,
        hessian_near_singular = leaf$hessian_near_singular,
        laplace_fallback = isTRUE(leaf$laplace_fallback),
        leaf_prior = leaf$leaf_prior,
        cov_prior = leaf$cov_prior,
        gamma_mean = leaf$gamma_mean,
        gamma_var_factor = leaf$gamma_var_factor,
        sigma2_shape = leaf$sigma2_shape,
        sigma2_scale = leaf$sigma2_scale,
        sigma2_mean = leaf$sigma2_mean,
        sigma2_hat = leaf$sigma2_mean,
        nugget = exp(theta[length(theta)]),
        log_evidence = leaf$log_evidence
      )
      inverse_range <- 1 / phi
      jr_C <- rep(NA_real_, length(phi))
      jr_influence <- rep(NA_real_, length(phi))
      if (identical(leaf$cov_prior, "jointly_robust")) {
        jr_C <- as.numeric(leaf$cov_prior_details$C)
        scaled_inverse_range <- jr_C * inverse_range
        jr_influence <- length(phi) * scaled_inverse_range /
          sum(scaled_inverse_range)
      }
      for (j in seq_along(phi)) {
        row[[phi_names[j]]] <- phi[j]
        row[[beta_names[j]]] <- inverse_range[j]
        row[[jr_C_names[j]]] <- jr_C[j]
        row[[jr_influence_names[j]]] <- jr_influence[j]
      }
      if (!is.null(y_train)) {
        y_leaf <- y_train[leaf$indices]
        row$y_mean <- mean(y_leaf)
        row$y_sd <- stats::sd(y_leaf)
        row$y_zero_frac <- mean(abs(y_leaf) <= 1e-12)
      }
      row_id <- row_id + 1L
      rows[[row_id]] <- as.data.frame(row, stringsAsFactors = FALSE)
    }
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

TVGP_tree_split_summary <- function(smc_result, input_names = NULL) {
  particles <- smc_result$final_particles
  weights <- smc_result$history_weights[[length(smc_result$history_weights)]]
  particle_signatures <- vapply(particles, TVGP_tree_signature, character(1))
  unique_signatures <- unique(particle_signatures)
  tree_ids <- match(particle_signatures, unique_signatures)
  if (is.null(input_names)) {
    p_dim <- length(particles[[1]]$nodes[[1]]$theta_hat) - 1L
    input_names <- paste0("x", seq_len(p_dim))
  }

  rows <- list()
  row_id <- 0L
  for (i in seq_along(particles)) {
    p <- particles[[i]]
    split_ids <- which(!vapply(p$nodes, function(z) isTRUE(z$is_leaf), logical(1)))
    for (sid in split_ids) {
      node <- p$nodes[[sid]]
      row_id <- row_id + 1L
      rows[[row_id]] <- data.frame(
        particle = i,
        particle_weight = weights[i],
        tree_id = tree_ids[i],
        node_id = sid,
        node_key = node$key,
        depth = node$depth,
        n_train = TVGP_node_n_train(node),
        split_dim = node$split_dim,
        split_var = input_names[node$split_dim],
        split_val_scaled = node$split_val,
        left = node$left,
        right = node$right,
        stringsAsFactors = FALSE
      )
    }
  }
  if (length(rows) == 0L) return(data.frame())
  do.call(rbind, rows)
}

TVGP_tree_signature_summary <- function(smc_result) {
  particles <- smc_result$final_particles
  weights <- smc_result$history_weights[[length(smc_result$history_weights)]]
  signatures <- vapply(particles, TVGP_tree_signature, character(1))
  unique_signatures <- unique(signatures)
  rows <- lapply(seq_along(unique_signatures), function(k) {
    idx <- which(signatures == unique_signatures[k])
    p <- particles[[idx[1L]]]
    data.frame(
      tree_id = k,
      n_particles = length(idx),
      weight_sum = sum(weights[idx]),
      n_nodes = length(p$nodes),
      n_leaves = sum(vapply(p$nodes, function(z) isTRUE(z$is_leaf), logical(1))),
      n_splits = sum(!vapply(p$nodes, function(z) isTRUE(z$is_leaf), logical(1))),
      max_depth = max(vapply(p$nodes, function(z) z$depth, numeric(1))),
      tree_signature = unique_signatures[k],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}
