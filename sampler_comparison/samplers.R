# TVGP-SMC and TVGP-RJ-MCMC on a common finite tree space (Section 4.1): quantile cut grid,
# maximum depth, and leaf scores from tvgp/tvgp_smc.R shared by both samplers.
# All implementation symbols are kept inside the .tvgp_samplers environment.
.tvgp_samplers <- local({
  files <- vapply(sys.frames(), function(f) if (is.null(f$ofile)) "" else f$ofile, "")
  files <- files[basename(files) == "samplers.R"]
  if (!length(files)) stop("Source sampler_comparison/samplers.R by filename.")
  bundle <- dirname(dirname(normalizePath(tail(files, 1L), mustWork = TRUE)))
  module <- environment()
  source(file.path(bundle, "sampler_comparison", "rjmcmc_kernel.R"), local = module)
  core_loaded <- FALSE

  load_core <- function() {
    if (!core_loaded) {
      if (!requireNamespace("digest", quietly = TRUE)) stop("Package digest is required.")
      source(file.path(bundle, "tvgp", "tvgp_smc.R"),
             local = module, chdir = TRUE)
      core_loaded <<- TRUE
    }
    invisible(NULL)
  }

  SVR_logsumexp <- function(x) {
    if (!length(x) || anyNA(x) || any(x == Inf)) stop("Invalid log weights.")
    z <- max(x)
    if (z == -Inf) return(-Inf)
    z + log(sum(exp(x - z)))
  }
  SVR_normalize_log_weights <- function(x) {
    z <- SVR_logsumexp(x)
    if (!is.finite(z)) stop("All weights are zero; no uniform fallback is allowed.")
    exp(x - z)
  }

  config <- function(X, cores = 16L, smoke = FALSE) {
    cfg <- list(split_grid = SVR_make_split_grid(X, 10L, list(), "quantile"),
         grid_size = 10L, grid_type = "quantile", max_depth = 3L,
         min_leaf_size = 30L, m_vecchia = 30L, nu = 2.5,
         alpha = 0.95, beta = 0.5, split_tol = 1e-10,
         ordering = "global", nn_pool_factor = 8, exact_nn_fallback = TRUE,
         cov_prior = "jointly_robust", cov_prior_control = list(jr_definition = "gu_2019",
           jr_a = 0.2, jr_b = 1, g_min = 1e-6),
         leaf_prior_control = list(m0 = 0, V0 = 100, a0 = 2, b0 = 1),
         full_maxit = 400L, hessian = TRUE, max_attempts = 12L,
         optimizer_gradient_tol = 1e-3, optimizer_boundary_tol = 1e-6,
         newton_maxit = 20L, parallel_cores = as.integer(cores),
         cache_limit = 2048L, rotate_enabled = TRUE,
         leaf_multistart = list(select = "first_passing", max_fits = 3L,
           phi_bounds = c(1e-3, 1e3), short_frac = 0.03, other_frac = 0.30,
           nugget = 0.01))
    if (smoke) {
      cfg$split_grid <- SVR_make_split_grid(X, 3L, list(), "quantile")
      cfg$grid_size <- 3L
      cfg$max_depth <- 2L
      cfg$m_vecchia <- 10L
      cfg$min_leaf_size <- 10L
    }
    cfg
  }

  new_target <- function(y, X, cfg = config(X)) {
    load_core()
    if (cfg$cache_limit < 1L || cfg$cache_limit > 2048L) stop("Invalid cache limit.")
    t <- new.env(parent = emptyenv())
    t$X <- as.matrix(X)
    t$y <- as.numeric(y)
    t$config <- cfg
    t$ctx <- TVGP_prepare_context(t$X, cfg$m_vecchia, ordering = cfg$ordering,
      nn_pool_factor = cfg$nn_pool_factor, exact_nn_fallback = cfg$exact_nn_fallback,
      verbose = FALSE)
    t$leaf_prior <- TVGP_leaf_prior_spec(cfg$leaf_prior_control)
    t$map <- new.env(hash = TRUE, parent = emptyenv())
    t$queue <- character()
    t$n_eval <- t$n_hit <- t$n_evicted <- t$peak_entries <- 0L
    t
  }

  subset_key <- function(indices) {
    digest::digest(sort(as.integer(indices)), algo = "sha256", serialize = TRUE)
  }

  strict_fit <- function(node, target) {
    cfg <- target$config
    idx <- sort(as.integer(node$indices))
    if (anyDuplicated(idx) || length(idx) < cfg$min_leaf_size) stop("Invalid leaf subset.")
    fit <- TVGP_optimize_node_with_retries(
      y = target$y, X = target$X, indices = idx, nu = cfg$nu,
      m = min(cfg$m_vecchia, length(idx) - 1L), ctx = target$ctx,
      cov_prior = cfg$cov_prior, cov_prior_control = cfg$cov_prior_control,
      leaf_prior_spec = target$leaf_prior, maxit = cfg$full_maxit,
      hessian = TRUE, max_attempts = cfg$max_attempts,
      optimizer_gradient_tol = cfg$optimizer_gradient_tol,
      optimizer_boundary_tol = cfg$optimizer_boundary_tol,
      newton_maxit = cfg$newton_maxit, multistart = cfg$leaf_multistart)
    if (isTRUE(fit$optimization_failed) || !is.finite(fit$log_evidence) ||
        fit$log_evidence <= -1e90 || !identical(fit$hessian_status, "analytic") ||
        !isTRUE(fit$hessian_min_eigenvalue > 0) ||
        !identical(fit$hessian_n_regularized, 0L) || isTRUE(fit$laplace_fallback)) {
      stop(structure(list(message = paste("Strict leaf score failed:", subset_key(idx),
        fit$optimization_failure_reason, fit$hessian_status), call = NULL,
        subset_key = subset_key(idx), indices = idx,
        attempts = fit$optimization_attempt_diagnostics),
        class = c("sampler_score_failure", "error", "condition")))
    }
    fit
  }

  fit_record <- function(node, target) {
    fit <- strict_fit(node, target)
    c(logPhi = fit$log_evidence, n = length(node$indices),
      attempts = fit$optimization_attempts, gradient = fit$gradient_inf_norm,
      hessian_min = fit$hessian_min_eigenvalue,
      hessian_condition = fit$hessian_condition_number)
  }

  cache_put <- function(key, record, target) {
    if (exists(key, target$map, inherits = FALSE)) return(invisible(NULL))
    if (length(target$queue) >= target$config$cache_limit) {
      rm(list = target$queue[1L], envir = target$map)
      target$queue <- target$queue[-1L]
      target$n_evicted <- target$n_evicted + 1L
    }
    assign(key, record, envir = target$map)
    target$queue <- c(target$queue, key)
    target$peak_entries <- max(target$peak_entries, length(target$queue))
    invisible(NULL)
  }

  SVR_logPhi_nodes <- function(nodes, target) {
    if (!length(nodes)) return(numeric())
    keys <- vapply(nodes, function(n) subset_key(n$indices), character(1))
    unique_keys <- unique(keys)
    first <- match(unique_keys, keys)
    values <- numeric(length(unique_keys))
    hit <- vapply(unique_keys, exists, logical(1), envir = target$map, inherits = FALSE)
    for (i in which(hit)) values[i] <- get(unique_keys[i], target$map)[["logPhi"]]
    target$n_hit <- target$n_hit + sum(hit)
    missing <- which(!hit)
    # Batches cap temporary worker results even when a query exceeds cache capacity.
    if (length(missing)) for (start in seq.int(1L, length(missing), by = 32L)) {
      batch <- missing[seq.int(start, min(start + 31L, length(missing)))]
      records <- TVGP_parallel_lapply(nodes[first[batch]], function(node) {
        tryCatch(list(ok = TRUE, record = fit_record(node, target)),
                 error = function(e) list(ok = FALSE, error = e))
      }, cores = target$config$parallel_cores, mc_preschedule = TRUE)
      for (j in seq_along(batch)) {
        result <- records[[j]]
        if (inherits(result, "try-error") || !is.list(result)) stop("Leaf worker failed.")
        if (!isTRUE(result$ok)) stop(result$error)
        i <- batch[j]
        values[i] <- result$record[["logPhi"]]
        target$n_eval <- target$n_eval + 1L
        cache_put(unique_keys[i], result$record, target)
      }
    }
    values[match(keys, unique_keys)]
  }
  SVR_logPhi_node <- function(node, target) SVR_logPhi_nodes(list(node), target)[1L]

  tree_stats <- function(tree, target, logML = NULL) {
    if (is.null(logML)) logML <- SVR_leaf_logPhi_sum(tree, target)
    lp <- log_prior_tree(tree, target)
    if (!is.finite(lp) || !is.finite(logML)) stop("Invalid scored tree.")
    size <- SVR_tree_size(tree)
    c(logML = logML, logprior = lp, logpost = logML + lp,
      leaves = size$n_leaves, depth = size$max_depth)
  }

  compact_tree <- function(tree) {
    rows <- list()
    visit <- function(node, id) {
      if (SVR_is_leaf(node)) return(invisible(NULL))
      rows[[length(rows) + 1L]] <<- c(node = id, axis = node$split_dim, cut = node$split_val)
      visit(node$left, 2L * id)
      visit(node$right, 2L * id + 1L)
    }
    visit(tree$root, 1L)
    if (!length(rows)) return(matrix(numeric(), 0L, 3L,
                                    dimnames = list(NULL, c("node", "axis", "cut"))))
    do.call(rbind, rows)
  }

  expand_tree <- function(compact, target) {
    build <- function(id) {
      node <- SVR_make_leaf_node(0L, numeric(), numeric(), integer())
      hit <- match(id, compact[, "node"])
      if (!is.na(hit)) {
        node$split_dim <- as.integer(compact[hit, "axis"])
        node$split_val <- as.numeric(compact[hit, "cut"])
        node$left <- build(2L * id)
        node$right <- build(2L * id + 1L)
      }
      node
    }
    tree <- SVR_recompute_tree(list(root = build(1L)), target$X, target$config)
    if (is.null(tree)) stop("Invalid compact tree.")
    tree
  }

  new_dictionary <- function() {
    d <- new.env(parent = emptyenv())
    d$map <- new.env(hash = TRUE, parent = emptyenv())
    d$trees <- d$stats <- list()
    d$counts <- d$weights <- numeric()
    d
  }

  retain_tree <- function(d, tree, stats, count, weight) {
    compact <- compact_tree(tree)
    key <- digest::digest(compact, algo = "sha256")
    if (exists(key, d$map, inherits = FALSE)) {
      id <- get(key, d$map)
    } else {
      id <- length(d$trees) + 1L
      assign(key, id, envir = d$map)
      d$trees[[id]] <- compact
      d$stats[[id]] <- stats
      d$counts[id] <- d$weights[id] <- 0
    }
    d$counts[id] <- d$counts[id] + count
    d$weights[id] <- d$weights[id] + weight
    id
  }

  dictionary_result <- function(d) {
    table <- data.frame(tree_id = seq_along(d$trees), count = d$counts,
      weight = d$weights / sum(d$weights), do.call(rbind, d$stats), row.names = NULL)
    list(tree_counts = table, tree_dict = d$trees)
  }

  autocorr <- function(x, lag = 1L) {
    if (length(x) <= lag || any(!is.finite(x)) || var(x) == 0) return(NA_real_)
    as.numeric(stats::acf(x, lag.max = lag, plot = FALSE)$acf[lag + 1L])
  }

  ess <- function(x) {
    n <- length(x)
    if (n < 4L || any(!is.finite(x)) || var(x) == 0) return(NA_real_)
    rho <- as.numeric(stats::acf(x, lag.max = n - 1L, plot = FALSE)$acf)
    pairs <- rho[seq.int(1L, n - 1L, by = 2L)] + rho[seq.int(2L, n, by = 2L)]
    first_bad <- which(pairs <= 0)[1L]
    if (!is.na(first_bad)) pairs <- head(pairs, first_bad - 1L)
    if (!length(pairs)) return(NA_real_)
    tau <- -1 + 2 * sum(cummin(pairs))
    if (!is.finite(tau) || tau <= 0) return(NA_real_)
    min(n, n / tau)
  }

  run_rjmcmc <- function(target, initial, n_iter, burn, started = proc.time()[["elapsed"]],
                         checkpoint = NULL, checkpoint_iterations = integer()) {
    tree <- initial
    state <- tree_stats(tree, target)
    trace <- as.data.frame(matrix(NA_real_, n_iter, 12L))
    names(trace) <- c("iteration", names(state), "time", "acceptance", "valid_proposal",
                      "kernel", "retained", "tree_id")
    d <- new_dictionary()
    running <- new_dictionary()
    current <- function() {
      z <- new_dictionary()
      retain_tree(z, tree, state, 1, 1)
      dictionary_result(z)
    }
    if (!is.null(checkpoint)) checkpoint(0L, proc.time()[["elapsed"]] - started,
      list(running = current(), current = current()))
    kernel_names <- c("grow", "prune", "change", "swap", "rotate")
    for (i in seq_len(n_iter)) {
      proposal <- SVR_propose_rjmcmc(tree, target)
      accepted <- FALSE
      if (isTRUE(proposal$valid)) {
        proposed <- tree_stats(proposal$tree, target)
        log_alpha <- proposed[["logpost"]] - state[["logpost"]] +
          proposal$log_q_reverse - proposal$log_q_forward
        if (is.na(log_alpha)) stop("Nonfinite MH ratio.")
        accepted <- log(runif(1L)) < min(0, log_alpha)
        if (accepted) {
          tree <- proposal$tree
          state <- proposed
        }
      }
      id <- if (i > burn) retain_tree(d, tree, state, 1, 1) else NA_integer_
      if (!is.null(checkpoint)) retain_tree(running, tree, state, 1, 1)
      trace[i, ] <- c(i, state, proc.time()[["elapsed"]] - started, as.integer(accepted),
        as.integer(proposal$valid), match(proposal$kernel, kernel_names), as.integer(i > burn), id)
      if (!is.null(checkpoint) && i %in% checkpoint_iterations) {
        populations <- list(running = dictionary_result(running), current = current())
        if (i > burn) populations$post_burn <- dictionary_result(d)
        checkpoint(i, proc.time()[["elapsed"]] - started, populations)
      }
      if (!is.null(target$checkpoint_dir) && i %% 250L == 0L) {
        write.table(trace[seq.int(i - 249L, i), ],
          file.path(target$checkpoint_dir, "partial_trace.csv"), sep = ",", row.names = FALSE,
          col.names = i == 250L, append = i != 250L)
        cat(sprintf("RJMCMC iteration %d/%d; elapsed %.1f s\n", i, n_iter, trace$time[i]))
        flush.console()
      }
    }
    trace$cumulative_elapsed_s <- trace$time
    retained <- trace[trace$retained == 1, , drop = FALSE]
    diagnostics <- do.call(rbind, lapply(names(state), function(nm) {
      data.frame(functional = nm, mcmc_ess = ess(retained[[nm]]),
                 autocorrelation_lag1 = autocorr(retained[[nm]]),
                 constant = var(retained[[nm]]) == 0)
    }))
    moves <- do.call(rbind, lapply(seq_along(kernel_names), function(k) {
      z <- trace[trace$kernel == k, , drop = FALSE]
      data.frame(kernel = kernel_names[k], proposed = nrow(z), valid = sum(z$valid_proposal),
        accepted = sum(z$acceptance), acceptance = if (nrow(z)) mean(z$acceptance) else NA_real_)
    }))
    c(list(trace = trace, diagnostics = diagnostics, move_diagnostics = moves), dictionary_result(d))
  }

  run_smc <- function(target, n_particles, started = proc.time()[["elapsed"]],
                      resample_fraction = 0.5, checkpoint = NULL) {
    root <- SVR_make_root_tree(target$X)
    root_ml <- SVR_logPhi_node(root$root, target)
    states <- list(list(tree = root, count = n_particles, logw = -log(n_particles), logML = root_ml))
    trace_rows <- list()
    logZ <- root_ml
    max_steps <- 2^target$config$max_depth - 1L
    for (step in 0:max_steps) {
      active_count <- 0L
      if (step > 0L) {
        next_states <- list()
        trans_map <- new.env(hash = TRUE, parent = emptyenv())
        trans_queue <- character()
        for (s in states) {
          path <- SVR_next_active_leaf_path(s$tree, target)
          if (is.null(path)) {
            next_states[[length(next_states) + 1L]] <- s
            next
          }
          active_count <- active_count + s$count
          node <- SVR_get_node(s$tree$root, path)
          key <- digest::digest(list(node$depth, node$lower, node$upper, subset_key(node$indices)))
          if (exists(key, trans_map, inherits = FALSE)) {
            trans <- get(key, trans_map)
          } else {
            trans <- SVR_evaluate_smc_transition(node, target)
            if (length(trans_queue) >= 64L) {
              rm(list = trans_queue[1L], envir = trans_map)
              trans_queue <- trans_queue[-1L]
            }
            assign(key, trans, trans_map)
            trans_queue <- c(trans_queue, key)
          }
          rho <- target$config$alpha * (1 + node$depth)^(-target$config$beta)
          log_probs <- c(log1p(-rho), log(rho) + trans$table$log_choice)
          counts <- as.vector(rmultinom(1L, s$count, SVR_normalize_log_weights(log_probs)))
          for (choice in which(counts > 0L)) {
            if (choice == 1L) {
              new_tree <- SVR_stop_leaf(s$tree, path)
              new_ml <- s$logML
            } else {
              j <- choice - 1L
              new_tree <- SVR_grow_tree(s$tree, path, trans$table$split_dim[j],
                trans$table$split_val[j], target$X, target$config)
              if (is.null(new_tree)) stop("SMC selected an invalid supported split.")
              new_ml <- s$logML + trans$table$log_BF[j]
            }
            next_states[[length(next_states) + 1L]] <- list(tree = new_tree,
              count = counts[choice], logw = s$logw + trans$log_w, logML = new_ml)
          }
        }
        if (!active_count) break
        states <- next_states
      }
      counts <- vapply(states, `[[`, numeric(1), "count")
      logw <- vapply(states, `[[`, numeric(1), "logw")
      z <- SVR_logsumexp(log(counts) + logw)
      if (!is.finite(z)) stop("Invalid SMC normalizer.")
      if (step > 0L) logZ <- logZ + z
      logw <- logw - z
      weights <- exp(logw)
      mass <- counts * weights
      pre_ess <- 1 / sum(counts * weights^2)
      stats <- vapply(states, function(s) tree_stats(s$tree, target, s$logML), numeric(5))
      signatures <- vapply(states, function(s) digest::digest(compact_tree(s$tree)), character(1))
      tree_mass <- tapply(mass, signatures, sum)
      resampled <- step > 0L && pre_ess < n_particles * resample_fraction
      after_counts <- if (resampled) as.vector(rmultinom(1L, n_particles, mass)) else counts
      trace_rows[[length(trace_rows) + 1L]] <- data.frame(iteration = step,
        as.list(as.vector(stats %*% mass)), time = proc.time()[["elapsed"]] - started,
        acceptance = NA_real_, pre_resample_ess = pre_ess,
        pre_resample_weight_max = max(weights), unique_trees = length(tree_mass),
        tree_weight_max = max(tree_mass), resampled = as.integer(resampled),
        unique_trees_after = length(unique(signatures[after_counts > 0])),
        post_resample_ess = if (resampled) n_particles else pre_ess,
        active_particles = if (step == 0L) n_particles else active_count,
        logZ = logZ, live_groups = length(states))
      names(trace_rows[[length(trace_rows)]])[2:6] <- rownames(stats)
      if (!is.null(target$checkpoint_dir)) {
        write.table(tail(trace_rows, 1L)[[1L]],
          file.path(target$checkpoint_dir, "partial_trace.csv"), sep = ",", row.names = FALSE,
          col.names = length(trace_rows) == 1L, append = length(trace_rows) != 1L)
        cat(sprintf("SMC stage %d; elapsed %.1f s; ESS %.1f\n", step,
          tail(trace_rows, 1L)[[1L]]$time, pre_ess))
        flush.console()
      }
      for (i in seq_along(states)) {
        states[[i]]$count <- after_counts[i]
        states[[i]]$logw <- if (resampled) -log(n_particles) else logw[i]
      }
      states <- states[after_counts > 0L]
      if (!is.null(checkpoint)) {
        population <- new_dictionary()
        for (s in states) retain_tree(population, s$tree,
          tree_stats(s$tree, target, s$logML), s$count, s$count * exp(s$logw))
        checkpoint(step, proc.time()[["elapsed"]] - started,
          list(population = dictionary_result(population)))
      }
    }
    if (any(vapply(states, function(s) !is.null(SVR_next_active_leaf_path(s$tree, target)), logical(1)))) {
      stop("SMC stopped before all decisions were processed.")
    }
    d <- new_dictionary()
    for (s in states) {
      # Recompute only final unique tree scores, checking the telescoping update.
      stat <- tree_stats(s$tree, target)
      if (abs(stat[["logML"]] - s$logML) > 1e-7 * max(1, abs(s$logML))) stop("SMC score mismatch.")
      retain_tree(d, s$tree, stat, s$count, s$count * exp(s$logw))
    }
    trace <- do.call(rbind, trace_rows)
    trace$cumulative_elapsed_s <- trace$time
    c(list(trace = trace, diagnostics = trace[, c("iteration", "pre_resample_ess",
      "pre_resample_weight_max", "unique_trees", "tree_weight_max", "resampled",
      "unique_trees_after", "post_resample_ess", "live_groups")]), dictionary_result(d))
  }

  prediction_particle <- function(compact, target) {
    tree <- expand_tree(compact, target)
    nodes <- list()
    visit <- function(node) {
      id <- length(nodes) + 1L
      nodes[[id]] <<- list(is_leaf = SVR_is_leaf(node), split_dim = node$split_dim,
                          split_val = node$split_val)
      if (SVR_is_leaf(node)) {
        fit <- strict_fit(node, target)
        key <- subset_key(node$indices)
        if (exists(key, target$map, inherits = FALSE) &&
            !identical(unname(get(key, target$map)[["logPhi"]]), unname(fit$log_evidence))) {
          stop("Prediction refit changed the canonical leaf score.")
        }
        nodes[[id]] <<- TVGP_make_node(as.integer(id), node$indices, fit, node$depth, key, TRUE)
      } else {
        left <- visit(node$left)
        right <- visit(node$right)
        nodes[[id]]$left <<- left
        nodes[[id]]$right <<- right
      }
      as.integer(id)
    }
    visit(tree$root)
    leaf <- vapply(nodes, function(node) isTRUE(node$is_leaf), logical(1))
    list(nodes = nodes, logML = sum(vapply(nodes[leaf], `[[`, numeric(1), "log_evidence")))
  }

  environment()
})
