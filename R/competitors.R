# Competing methods: TGP, DGP VEC (deepgp), SVecchia (scaled Vecchia), SPGP, JGP, laGP, and a
# global Vecchia GP (SGV). run_baseline(method, data, settings, outdir) fits one method and
# returns the predictive mean and 300 predictive draws of the latent function at data$Xt.
.baseline_root <- local({
  files <- vapply(sys.frames(), function(f) if (is.null(f$ofile)) "" else f$ofile, "")
  files <- files[basename(files) == "competitors.R"]
  if (!length(files)) stop("Source R/competitors.R by filename.")
  dirname(dirname(normalizePath(tail(files, 1L), mustWork = TRUE)))
})

.baseline_packages <- list(
  tgp = "tgp", deepgp = "deepgp",
  scaled_vecchia = c("GpGp", "GPvecchia", "FNN"),
  lagp = "laGP", sgv = c("GpGp", "FNN"), jumpgp = character(),
  spgp = c("igraph", "fields", "FNN", "genlasso", "GpGp", "Rfast",
           "deldir", "RANN", "Matrix", "ramcmc")
)

.bl_python <- function(settings = list()) {
  if (!is.null(settings$python)) return(settings$python)
  Sys.getenv("TVGP_BASELINE_PYTHON", unset = "python3")
}

baseline_dependencies <- function(methods = names(.baseline_packages), python = .bl_python()) {
  do.call(rbind, lapply(methods, function(method) {
    if (!method %in% names(.baseline_packages)) stop("Unknown baseline: ", method)
    pkgs <- .baseline_packages[[method]]
    missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
    details <- if (length(missing)) paste("Missing R packages:", paste(missing, collapse = ", ")) else ""
    if (method == "tgp" && !length(missing) && utils::packageVersion("tgp") != "2.4.23") {
      details <- "TGP latent-variance audit requires tgp 2.4-23."
    }
    if (method == "deepgp" && !length(missing) && utils::packageVersion("deepgp") != "1.2.3") {
      details <- "Memory-bounded continuation is audited for deepgp 1.2.3; install that version or re-audit."
    }
    if (method == "spgp" && !file.exists(file.path(.baseline_root, "vendor", "SPGP", "BRISC", "BRISC.so"))) {
      details <- paste(details, "Missing SPGP BRISC.so (run dependency preparation).")
    }
    if (method == "spgp" && !nzchar(details)) {
      backend <- file.path(.baseline_root, "vendor", "SPGP", "BRISC", "BRISC.so")
      details <- tryCatch({
        loaded <- vapply(getLoadedDLLs(), function(x) x[["path"]], "")
        if (!backend %in% loaded) {
          dyn.load(backend)
          dyn.unload(backend)
        }
        ""
      }, error = function(e) paste("SPGP native backend cannot load:", conditionMessage(e)))
    }
    if (method == "jumpgp") {
      probe <- tryCatch(suppressWarnings(system2(python,
        c(shQuote(file.path(.baseline_root, "vendor", "jumpgp_bridge.py")), "--check"),
        stdout = TRUE, stderr = TRUE, env = c("PYTHONDONTWRITEBYTECODE=1", "MPLBACKEND=Agg"))),
        error = function(e) structure(conditionMessage(e), status = 1L))
      status <- attr(probe, "status")
      if (!is.null(status) && status != 0L) details <- paste(tail(probe, 2L), collapse = " ")
    }
    data.frame(method = method, available = !nzchar(trimws(details)),
               detail = trimws(details), stringsAsFactors = FALSE)
  }))
}

.bl_int <- function(x, name, minimum = 1L) {
  if (length(x) != 1L || !is.numeric(x) || !is.finite(x) ||
      x != floor(x) || x < minimum || x > .Machine$integer.max) stop("Invalid ", name)
  as.integer(x)
}

.bl_settings <- function(data, settings) {
  defaults <- list(m = 30L, nu = 2.5, cores = 4L, n_draws = 300L,
                   seed_fit = if (is.null(data$seed_fit)) data$rep + 50000L else data$seed_fit,
                   seed_predict = if (is.null(data$seed_predict)) data$rep + 21000L else data$seed_predict,
                   smoke = FALSE)
  s <- utils::modifyList(defaults, settings)
  for (name in c("m", "cores", "n_draws")) s[[name]] <- .bl_int(s[[name]], name)
  for (name in c("seed_fit", "seed_predict")) s[[name]] <- .bl_int(s[[name]], name, 0L)
  if (s$n_draws != 300L && !isTRUE(s$smoke)) stop("The unified baseline contract requires n_draws=300.")
  s$n_draws <- 300L
  if (length(s$nu) != 1L || !is.finite(s$nu) || s$nu <= 0) stop("Invalid nu")
  if (!is.logical(s$smoke) || length(s$smoke) != 1L || is.na(s$smoke)) stop("Invalid smoke")
  s
}

.bl_matrix <- function(x, name, d = NULL) {
  x <- as.matrix(x)
  if (!is.numeric(x) || !nrow(x) || !ncol(x) || any(!is.finite(x)) ||
      (!is.null(d) && ncol(x) != d)) stop("Invalid ", name, " dimensions or values")
  storage.mode(x) <- "double"
  x
}

.bl_moments <- function(mu, variance, n) {
  mu <- as.numeric(mu); variance <- as.numeric(variance)
  if (length(mu) != n || length(variance) != n || any(!is.finite(mu)) ||
      any(!is.finite(variance)) || any(variance < 0)) stop("Invalid predictive moments; no rows may be dropped.")
  list(mean = mu, variance = variance)
}

.bl_nonnegative <- function(x, scale = 1) {
  if (any(!is.finite(x))) stop("Non-finite latent variance")
  if (any(x < -2e-6 * max(1, abs(scale)))) stop("Materially negative latent variance after noise removal")
  pmax(x, 0)
}

.bl_draws <- function(mu, variance, n = 300L) {
  mu + sqrt(variance) * matrix(stats::rnorm(length(mu) * n), nrow = length(mu), ncol = n)
}

.bl_scale_y <- function(y) {
  scale <- stats::sd(y)
  if (!is.finite(scale) || scale <= 1e-12) scale <- 1
  list(y = (y - mean(y)) / scale, center = mean(y), scale = scale)
}

.bl_chunks <- function(n, size = 100L) split(seq_len(n), ceiling(seq_len(n) / size))

.bl_with_seed <- function(seed, code) {
  old <- get(".Random.seed", envir = .GlobalEnv)
  on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
  set.seed(seed)
  force(code)
}

.bl_tgp_bte <- function(experiment, smoke) {
  if (smoke) c(10L, 40L, 1L) else if (grepl("^bench_", experiment)) {
    c(20000L, 50000L, 100L)
  } else c(2000L, 7000L, 20L)
}

.bl_source <- function(file) {
  e <- new.env(parent = globalenv())
  sys.source(file, envir = e)
  e
}

.bl_surface <- function(result, data, predict_mean) {
  if (is.null(data$surface)) return(result)
  timer <- system.time({ result$surface_mean <- as.numeric(predict_mean(data$surface$X)) })
  if (length(result$surface_mean) != nrow(data$surface$X) || any(!is.finite(result$surface_mean))) {
    stop("Invalid surface predictions; no surface rows may be dropped.")
  }
  result$surface_s <- unname(timer[["elapsed"]])
  result
}

.bl_tgp_trace <- function(pr, fit) {
  km <- t(as.matrix(pr$trace$preds$ZZ.km))
  kv <- t(as.matrix(pr$trace$preds$ZZ.ks2))
  tr <- pr$trace$XX
  if (length(tr) != nrow(km) || !identical(dim(km), dim(kv))) stop("Missing TGP per-input kriging traces")
  noise <- t(vapply(tr, function(z) {
    if (!all(c("s2", "nug") %in% names(z)) || nrow(z) != ncol(km)) stop("Missing TGP s2/nug traces")
    as.numeric(z$s2 * z$nug)
  }, numeric(ncol(km))))
  # tgp 2.4-23 predictive_var uses s2 * (1 + nug + trend - reduction).
  # The raw kriging trace and per-leaf parameters are on the internal y scale.
  rng <- if (isTRUE(fit$m0r1)) diff(range(fit$Z)) else 1
  center <- if (isTRUE(fit$m0r1)) mean(fit$Z) else 0
  list(mean = center + rng * km,
       variance = .bl_nonnegative((kv - noise) * rng^2, kv * rng^2),
       response_variance = kv * rng^2, noise = noise * rng^2)
}

.bl_tgp <- function(data, s, outdir) {
  if (utils::packageVersion("tgp") != "2.4.23") stop("TGP nugget audit is pinned to tgp 2.4-23; re-audit a different version.")
  bte <- .bl_tgp_bte(data$experiment, s$smoke)
  tgp_corr <- if (is.null(s$tgp_corr)) "matern" else s$tgp_corr
  if (!tgp_corr %in% c("matern", "expsep", "exp")) stop("Unsupported TGP correlation: ", tgp_corr)
  work <- tempfile("tgp_", tmpdir = outdir)
  dir.create(work); old <- setwd(work)
  on.exit({ setwd(old); unlink(work, recursive = TRUE) }, add = TRUE)
  set.seed(s$seed_fit)
  ft <- system.time({
    # The paper uses the package's isotropic Matern correlation with nu = 2.5.
    tgp_args <- list(data$X, data$y, XX = data$Xt, corr = tgp_corr,
                     meanfn = "constant", tree = c(0.95, 0.5),
                     BTE = bte, pred.n = FALSE, trace = TRUE, krige = TRUE,
                     zcov = FALSE, verb = 1)
    if (tgp_corr == "matern") tgp_args$nu <- s$nu
    fit <- do.call(tgp::btgp, tgp_args)
  })[["elapsed"]]
  message("TGP primary chain and test kriging finished in ", round(ft, 1), " seconds.")
  # Test kriging is integrated into this single chain, not a second predict chain.
  saveRDS(list(trees=fit$trees,parts=fit$parts,posts=fit$posts,params=fit$params,
    Xsplit=fit$Xsplit,trace=fit$trace[c("parts","post","hier")],BTE=bte,
    meanfn="constant",tree_prior=c(.95,.5)),file.path(outdir,"tgp_tree_structure.rds"),compress=FALSE)
  set.seed(s$seed_predict)
  pt <- system.time({
    tr <- .bl_tgp_trace(fit, fit)
    mu <- rowMeans(tr$mean)
    ids <- sample.int(ncol(tr$mean), 300L, replace = ncol(tr$mean) < 300L)
    draws <- tr$mean[, ids, drop = FALSE] + sqrt(tr$variance[, ids, drop = FALSE]) *
      matrix(stats::rnorm(nrow(data$Xt) * 300L), nrow(data$Xt), 300L)
  })[["elapsed"]]
  result <- list(mean = mu, draws = draws, fit_s = ft, pred_s = pt,
    settings = list(BTE = bte, primary_chain_iterations = bte[2L],
      retained_trace_states = ncol(tr$mean), corr = tgp_corr, package_version = "2.4-23",
      meanfn = "constant", tree = c(0.95, 0.5), bprior = fit$params$bprior,
      fit_s_includes_kriging = TRUE, surface_resumed_chain = !is.null(data$surface)),
    diagnostics = list(kriging_trace_includes_nugget = TRUE,
      noise_variance = as.numeric(rowMeans(tr$noise)), response_conditional_variance = as.numeric(rowMeans(tr$response_variance)),
      latent_conditional_variance = as.numeric(rowMeans(tr$variance)),
      latent_variance = as.numeric(rowMeans(tr$variance + tr$mean^2) - mu^2),
      prediction_chain = "Single btgp chain with integrated test kriging; pred_s is draw summarization",
      native_leaf_prior = fit$params,
      surface_chain = "Separate predict.tgp continuation from fitted MAP tree, excluded from primary timing"))
  .bl_surface(result, data, function(x) {
    # Marginal kriging does not require the full test-test covariance matrix.
    message("TGP surface prediction: one continuation for ", nrow(x), " grid points.")
    set.seed(s$seed_predict)
    p <- stats::predict(fit, XX = x, BTE = bte, MAP = FALSE,
                        pred.n = FALSE, krige = TRUE, zcov = FALSE, trace = FALSE, verb = 1)
    as.numeric(p$ZZ.km)
  })
}

.bl_deep_noise <- function(fit) {
  tau <- fit[["tau2_y"]]
  if (is.null(tau)) tau <- fit[["tau2"]]
  g <- fit[["g"]]
  if (is.null(tau) || is.null(g)) stop("Missing deepGP outer-layer noise parameters")
  tau <- as.numeric(tau); g <- as.numeric(g)
  n <- fit$nmcmc
  if (!length(tau) %in% c(1L, n) || !length(g) %in% c(1L, n)) stop("Incompatible deepGP noise traces")
  noise <- rep(tau, length.out = n) * rep(g, length.out = n)
  if (any(!is.finite(noise)) || any(noise < 0)) stop("Invalid deepGP noise traces")
  noise
}

.bl_deep_continue <- function(object, new_mcmc, cores = object$x_approx$cores, verb = FALSE) {
  # Exact non-gradient/non-monowarp branch of deepgp 1.2.3 continue.dgp2vec,
  # except discard old states instead of abind'ing two enormous latent arrays.
  # Explicit true_g=NULL is essential: a one-state trim must NOT fix the nugget.
  if (utils::packageVersion("deepgp") != "1.2.3") stop("Re-audit deepGP continuation for this package version")
  cores <- .bl_int(cores, "deepGP fit cores")
  if (cores > 16L) stop("This driver limits deepGP fitting to sixteen cores")
  if (!is.null(object$dydx) || isTRUE(object$settings$monowarp)) stop("Unsupported deepGP continuation variant")
  last <- object$nmcmc
  initial <- list(theta_y = object$theta_y[last], theta_w = object$theta_w[last, ],
                  g = object$g[last], w = object$w[last, , ])
  sampler <- getFromNamespace("gibbs_two_layer_vec", "deepgp")
  samples <- sampler(x = object$x, y = object$y, nmcmc = new_mcmc + 1L,
    D = dim(object$w)[3L], verb = verb, initial = initial, true_g = NULL,
    settings = object$settings, v = object$v, m = object$x_approx$m,
    ord = NULL, cores = cores, x_approx = NULL, w_approx = NULL)
  for (name in c("tau2_y", "theta_y", "theta_w", "g", "w", "ll", "x_approx", "w_approx")) {
    object[[name]] <- samples[[name]]
  }
  object$nmcmc <- new_mcmc + 1L
  object
}

.bl_deepgp <- function(data, s, outdir) {
  nphase <- if (s$smoke) 8L else 6000L
  fit_cores <- min(16L, s$cores)
  if (getFromNamespace("check_cores", "deepgp")(fit_cores) != fit_cores) {
    stop("deepGP cannot use the requested CPU count on this host")
  }
  latent_bytes <- as.double(nrow(data$X)) * ncol(data$X) * nphase * 8
  cores <- min(2L, s$cores, if (latent_bytes > 1024^3) 1L else 2L)
  memory_estimate <- (cores + 2L) * latent_bytes + 2 * 1024^3
  if (memory_estimate > 16 * 1024^3) stop("deepGP estimated chain/worker memory exceeds 16 GiB")
  m <- min(s$m, nrow(data$X) - 1L)
  ys <- .bl_scale_y(data$y)
  set.seed(s$seed_fit)
  ft <- system.time({
    message("deepGP initial phase: ", nphase, " states, fit cores=", fit_cores)
    fit <- deepgp::fit_two_layer(x = data$X, y = ys$y, nmcmc = nphase,
      D = ncol(data$X), pmx = TRUE, g_0 = 0.01, true_g = NULL,
      v = s$nu, cov = "matern", vecchia = TRUE, m = m, cores = fit_cores, verb = TRUE)
    saveRDS(fit[c("tau2_y","theta_y","theta_w","g","ll")],
      file.path(outdir,"deepgp_initial_trace.rds"),compress=FALSE)
    fit <- deepgp::trim(fit, burn = nphase - 1L, thin = 1L)
    invisible(gc(FALSE))
    message("deepGP continuation: ", nphase, " new states, fit cores=", fit_cores)
    fit <- .bl_deep_continue(fit, new_mcmc = nphase, cores = fit_cores, verb = TRUE)
    invisible(gc(FALSE))
    fit <- deepgp::trim(fit, burn = 1L, thin = 1L)
    invisible(gc(FALSE))
    if (fit$nmcmc != nphase) stop("deepGP retained the wrong number of iterations")
    noise <- mean(.bl_deep_noise(fit))
  })[["elapsed"]]
  message("deepGP fitting finished in ", round(ft, 1), " seconds; prediction cores=", cores)
  saveRDS(fit[c("tau2_y","theta_y","theta_w","g","ll")],
    file.path(outdir,"deepgp_retained_trace.rds"),compress=FALSE)
  predict_moments <- function(x) {
    mu <- vr <- numeric(nrow(x))
    for (ii in .bl_chunks(nrow(x), 100L)) {
      p <- stats::predict(fit, x_new = x[ii, , drop = FALSE], m = m,
                          lite = TRUE, mean_map = TRUE, return_all = FALSE, cores = cores)
      z <- .bl_moments(p$mean, p$s2, length(ii))
      mu[ii] <- z$mean; vr[ii] <- z$variance
      rm(p, z)
      invisible(gc(FALSE))
    }
    list(mean = ys$center + ys$scale * mu, response_variance = ys$scale^2 * vr,
         variance = .bl_nonnegative(ys$scale^2 * (vr - noise), ys$scale^2 * vr))
  }
  set.seed(s$seed_predict)
  pt <- system.time({
    p <- predict_moments(data$Xt)
    draws <- .bl_draws(p$mean, p$variance)
  })[["elapsed"]]
  result <- list(mean = p$mean, draws = draws, fit_s = ft, pred_s = pt,
    settings = list(initial_nmcmc = nphase, continuation_nmcmc = nphase,
      burn = nphase, retained = fit$nmcmc, thin = 1L, re_approx = TRUE,
      m_fit = m, m_pred = m, cores = cores, fit_cores = fit_cores, prediction_chunk_size = 100L,
      memory_bounded_continuation = TRUE, latent_chain_bytes = latent_bytes,
      estimated_memory_bytes = memory_estimate, package_version = "1.2.3",
      lite = TRUE, return_all = FALSE, mean_map = TRUE),
    diagnostics = list(y_center = ys$center, y_scale = ys$scale,
      noise_variance = ys$scale^2 * noise, response_variance = p$response_variance,
      latent_variance = p$variance, draws = "Gaussian marginal moment match"))
  .bl_surface(result, data, function(x) predict_moments(x)$mean)
}

.bl_scaled_source <- function() {
  file <- file.path(.baseline_root, "vendor", "vecchia_scaled.R")
  if (unname(tools::md5sum(file)) != "5a2a1e082eb8e24ab26f970c3309ee79") {
    stop("Pinned Scaled Vecchia source checksum mismatch")
  }
  .bl_source(file)
}

.bl_scaled_fit <- function(e, X, y, s) {
  fit <- e$fit_scaled(y = y, inputs = X, ms = 200L, trend = "pre",
    nu = s$nu, nug = NULL, scale = "parms", select = Inf,
    print.level = 0L, max.it = if (s$smoke) 2L else 32L, tol.dec = 4,
    n.est = min(nrow(X), 5000L), find.vcf = FALSE)
  # Upstream fit_scaled otherwise silently uses fit_vcf's m.pred=140 default.
  fit$vcf <- e$fit_vcf(fit, m.pred = 200L, scale = "parms", scorefun = e$ls)
  fit
}

.bl_gpgp_noise <- function(fit) {
  # GpGp's last covariance parameter is nugget / process variance.
  p0 <- fit$covparms; p0[length(p0)] <- 0
  fun <- get(fit$covfun_name, envir = asNamespace("GpGp"))
  x <- fit$locs[1L, , drop = FALSE]
  noise <- as.numeric(fun(fit$covparms, x) - fun(p0, x))
  if (length(noise) != 1L || !is.finite(noise) || noise < 0) stop("Invalid GpGp fitted noise")
  noise
}

.bl_scaled_vecchia <- function(data, s, outdir) {
  e <- .bl_scaled_source()
  set.seed(s$seed_fit)
  ft <- system.time({ fit <- .bl_scaled_fit(e, data$X, data$y, s) })[["elapsed"]]
  noise <- .bl_gpgp_noise(fit)
  predict_moments <- function(x) {
    p <- e$predictions_scaled(fit, locs_pred = x, m = 200L,
                              joint = FALSE, predvar = TRUE, scale = "parms")
    z <- .bl_moments(p$means, p$vars, nrow(x))
    list(mean = z$mean, response_variance = z$variance,
         variance = .bl_nonnegative(z$variance - fit$vcf * noise, z$variance))
  }
  set.seed(s$seed_predict)
  pt <- system.time({ p <- predict_moments(data$Xt); draws <- .bl_draws(p$mean, p$variance) })[["elapsed"]]
  result <- list(mean = p$mean, draws = draws, fit_s = ft, pred_s = pt,
    settings = list(m_fit = 200L, m_vcf = 200L, m_pred = 200L, n_est = min(nrow(data$X), 5000L),
      effective_m_fit = min(200L, nrow(data$X) - 1L), joint = FALSE, trend = "pre", nug = "estimated"),
    diagnostics = list(covparms = fit$covparms, convergence = fit$conv, vcf = fit$vcf,
      noise_variance = noise, response_variance = p$response_variance,
      latent_variance = p$variance, noise_removed = fit$vcf * noise,
      source_sha256 = "67e76191684740ecb750f8f9e5cc5498c2549e63736e42e0552856ed742c8a9b"))
  .bl_surface(result, data, function(x) predict_moments(x)$mean)
}

.bl_sgv <- function(data, s, outdir) {
  set.seed(s$seed_fit)
  ft <- system.time({
    fit <- GpGp::fit_model(data$y, data$X, matrix(1, nrow(data$X), 1L),
      "matern25_scaledim", m_seq = min(s$m, nrow(data$X) - 1L),
      max_iter = if (s$smoke) 2L else 40L, silent = TRUE)
  })[["elapsed"]]
  noise <- .bl_gpgp_noise(fit)
  # Independent test marginals: GpGp singleton prediction uses only training
  # neighbors. Its conditional variance is computed from the same official kernel.
  predict_moments <- function(x) {
    m <- min(s$m, nrow(fit$locs))
    nn <- FNN::get.knnx(fit$locs, x, k = m)$nn.index
    fun <- get(fit$covfun_name, envir = asNamespace("GpGp"))
    mu <- vr <- numeric(nrow(x))
    for (i in seq_len(nrow(x))) {
      ix <- nn[i, ]
      locs <- rbind(fit$locs[ix, , drop = FALSE], x[i, , drop = FALSE])
      K <- fun(fit$covparms, locs)
      L <- t(chol(K[seq_len(m), seq_len(m), drop = FALSE]))
      a <- forwardsolve(L, K[seq_len(m), m + 1L])
      mu[i] <- fit$betahat + sum(a * forwardsolve(L, fit$y[ix] - fit$betahat))
      vr[i] <- K[m + 1L, m + 1L] - noise - sum(a^2)
    }
    .bl_moments(mu, .bl_nonnegative(vr, fit$covparms[1L]), nrow(x))
  }
  set.seed(s$seed_predict)
  pt <- system.time({ p <- predict_moments(data$Xt); draws <- .bl_draws(p$mean, p$variance) })[["elapsed"]]
  result <- list(mean = p$mean, draws = draws, fit_s = ft, pred_s = pt,
    settings = list(m_fit = s$m, m_pred = s$m, covfun = "matern25_scaledim",
      prediction = "Independent training-neighbor latent marginals of GpGp fit"),
    diagnostics = list(covparms = fit$covparms, convergence = fit$conv,
      noise_variance = noise, latent_variance = p$variance))
  .bl_surface(result, data, function(x) predict_moments(x)$mean)
}

.bl_lagp <- function(data, s, outdir) {
  ys <- .bl_scale_y(data$y)
  end <- min(50L, nrow(data$X)); start <- min(6L, end - 1L)
  predict_local <- function(x, draws = TRUE) {
    mu <- vr <- df <- g_used <- numeric(nrow(x))
    samples <- matrix(NA_real_, nrow(x), 300L)
    fit_s <- pred_s <- 0
    for (i in seq_len(nrow(x))) {
      errors <- character(); ok <- FALSE
      for (g in c(1e-6, 1e-5, 1e-4, 1e-3)) {
        attempt <- tryCatch({
          ft <- system.time(.bl_with_seed(s$seed_fit, {
            local <- laGP::laGP(x[i, , drop = FALSE], start, end, data$X, ys$y,
                                 g = g, method = "alc", Xi.ret = TRUE, lite = TRUE, verb = 0)
            d <- if (!is.null(local$mle$d)) local$mle$d else local$d$start
            gp <- laGP::newGP(data$X[local$Xi, , drop = FALSE], ys$y[local$Xi], d = d, g = g, dK = FALSE)
          }))[["elapsed"]]
          fit_s <- fit_s + ft
          pt <- system.time({
            p <- tryCatch(laGP::predGP(gp, x[i, , drop = FALSE], lite = TRUE, nonug = TRUE),
                          finally = laGP::deleteGP(gp))
            if (!is.finite(p$df) || p$df <= 2) stop("Invalid laGP Student-t degrees of freedom")
            z <- .bl_moments(p$mean, p$s2 * p$df / (p$df - 2), 1L)
            mu[i] <- ys$center + ys$scale * z$mean
            vr[i] <- ys$scale^2 * z$variance; df[i] <- p$df; g_used[i] <- g
            if (draws) samples[i, ] <- mu[i] + ys$scale * sqrt(p$s2) * stats::rt(300L, df = p$df)
          })[["elapsed"]]
          pred_s <- pred_s + pt
          TRUE
        }, error = function(e) { errors <<- c(errors, conditionMessage(e)); FALSE })
        if (attempt) { ok <- TRUE; break }
      }
      if (!ok) stop("laGP failed at test row ", i, ": ", paste(errors, collapse = "; "))
    }
    list(mean = mu, variance = vr, draws = samples, fit_s = fit_s, pred_s = pred_s, df = df, g_used = g_used)
  }
  set.seed(s$seed_predict)
  p <- predict_local(data$Xt)
  result <- list(mean = p$mean, draws = p$draws, fit_s = p$fit_s, pred_s = p$pred_s,
    settings = list(start = start, end = end, method = "alc", nonug = TRUE),
    diagnostics = list(g_used = p$g_used, df = p$df, latent_variance = p$variance,
      y_center = ys$center, y_scale = ys$scale,
      local_fit = "Local ALC selection, reconstruct fitted local GP, predGP(nonug=TRUE)"))
  .bl_surface(result, data, function(x) predict_local(x, draws = FALSE)$mean)
}

.bl_spgp <- function(data, s, outdir) {
  dir <- file.path(.baseline_root, "vendor", "SPGP")
  old <- setwd(dir); on.exit(setwd(old), add = TRUE)
  e <- new.env(parent = globalenv())
  e$source <- function(file, ...) sys.source(file, envir = e)
  sys.source("R/SPGP_func_iso.R", envir = e)
  sys.source("R/utils.R", envir = e)
  sys.source("spgp_init.R", envir = e)
  niter <- if (s$smoke) 12L else 7000L; burn <- if (s$smoke) 6L else 2000L
  design <- matrix(1, nrow(data$X), 1L)
  set.seed(s$seed_fit)
  ft <- system.time({
    init <- e$spgp_faithful_init(data$X, data$y, n_knot_total = if (s$smoke) 36L else 600L,
                                k0 = 4L, L = 3L, seed_proj = data$rep + 5000L)
    fit <- e$STGP_fit(data$y, design, data$X, init$coords_knot, init$graph0,
      init$init_val, init$hyperpar, init$tunning, niter, burn, 1L, s$seed_fit,
      NNGP = TRUE, sample_latent = TRUE, standardizeY = FALSE, fix_proj = FALSE)
  })[["elapsed"]]
  predict_samples <- function(x) {
    result <- NULL
    for (ii in .bl_chunks(nrow(x), size = 1000L)) {
      p <- e$STGP_predict2_latent_mean(fit, X = design, X_ho = matrix(1, length(ii), 1L),
        coords = data$X, coords_ho = x[ii, , drop = FALSE], nn = init$hyperpar[["nn"]],
        return_values = "all", NNGP = FALSE, nn_nngp = init$hyperpar[["nngp"]],
        min_n_nngp = init$tunning[["min_n_nngp"]])
      mat <- as.matrix(p$Y_ho)
      if (nrow(mat) != length(ii) || !ncol(mat) || any(!is.finite(mat))) stop("Invalid SPGP latent draws")
      if (is.null(result)) result <- matrix(NA_real_, nrow(x), ncol(mat))
      result[ii, ] <- mat
    }
    result
  }
  set.seed(s$seed_predict)
  pt <- system.time({
    raw <- predict_samples(data$Xt); mu <- rowMeans(raw)
    ids <- sample.int(ncol(raw), 300L, replace = ncol(raw) < 300L)
    draws <- raw[, ids, drop = FALSE]
  })[["elapsed"]]
  result <- list(mean = mu, draws = draws, fit_s = ft, pred_s = pt,
    settings = list(mcmc = niter, burn = burn, thin = 1L, k0 = 4L,
      initialization = "Existing 6x6 pilot NNGP/fused-lasso initialization; fixed existing priors",
      n_knots = nrow(init$coords_knot), prediction_chunk_size = 1000L,
      projection_seed = data$rep + 5000L,
      standardizeY = FALSE, sample_latent = TRUE),
    diagnostics = list(latent_variance = apply(raw, 1L, stats::var), hyperpar = init$hyperpar))
  .bl_surface(result, data, function(x) {
    mu <- numeric(nrow(x))
    for (ii in .bl_chunks(nrow(x))) mu[ii] <- rowMeans(predict_samples(x[ii, , drop = FALSE]))
    mu
  })
}

.bl_jumpgp <- function(data, s, outdir) {
  work <- tempfile("jumpgp_", tmpdir = outdir); dir.create(work)
  train <- file.path(work, "train.csv"); test <- file.path(work, "test.csv")
  utils::write.table(cbind(data$X, data$y), train, sep = ",", row.names = FALSE, col.names = FALSE)
  utils::write.table(data$Xt, test, sep = ",", row.names = FALSE, col.names = FALSE)
  args <- c(shQuote(file.path(.baseline_root, "vendor", "jumpgp_bridge.py")),
    "--train", shQuote(train), "--test", shQuote(test), "--outdir", shQuote(work),
    "--cores", s$cores, "--neighbors", min(25L, nrow(data$X)),
    "--seed-fit", s$seed_fit, "--seed-predict", s$seed_predict)
  if (!is.null(data$surface)) {
    surf <- file.path(work, "surface.csv")
    utils::write.table(data$surface$X, surf, sep = ",", row.names = FALSE, col.names = FALSE)
    args <- c(args, "--surface", shQuote(surf))
  }
  log <- file.path(work, "python.log")
  status <- system2(.bl_python(s), args, stdout = log, stderr = log,
    env = c("PYTHONDONTWRITEBYTECODE=1", "MPLBACKEND=Agg", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1"))
  if (status != 0L) stop("JumpGP public CEM/external UK failed. See ", log,
                         "; ", paste(tail(readLines(log, warn = FALSE), 5L), collapse = " "))
  raw <- utils::read.csv(file.path(work, "predictions.csv"))
  if (!identical(raw$row, seq_len(nrow(data$Xt)))) stop("JumpGP missing/reordered test rows")
  p <- .bl_moments(raw$uk_mean, raw$uk_latent_variance, nrow(data$Xt))
  times <- utils::read.csv(file.path(work, "timing.csv"))
  set.seed(s$seed_predict)
  dt <- system.time({ draws <- .bl_draws(p$mean, p$variance) })[["elapsed"]]
  result <- list(mean = p$mean, draws = draws, fit_s = times$fit_s, pred_s = times$pred_s + dt,
    settings = list(boundary = if (ncol(data$X) <= 3L) "quadratic" else "linear",
      neighbors = min(25L, nrow(data$X)), shared_m_used = FALSE,
      mode = "CEM", prediction = "External UK latent"),
    diagnostics = list(raw_official_and_external = raw, artifacts = work,
      latent_variance = p$variance, timing = "Measured local CEM and external UK wall time by batch"))
  if (!is.null(data$surface)) {
    surface <- utils::read.csv(file.path(work, "surface_predictions.csv"))
    if (!identical(surface$row, seq_len(nrow(data$surface$X)))) stop("JumpGP missing surface rows")
    result$surface_mean <- .bl_moments(surface$uk_mean, surface$uk_latent_variance,
                                      nrow(data$surface$X))$mean
    result$surface_s <- times$surface_s
    result$diagnostics$surface_raw <- surface
  }
  result
}

run_baseline <- function(method, data, settings, outdir) {
  method <- tolower(method)
  if (length(method) != 1L || !method %in% names(.baseline_packages)) stop("Unknown baseline method")
  data$X <- .bl_matrix(data$X, "X"); data$Xt <- .bl_matrix(data$Xt, "Xt", ncol(data$X))
  if (nrow(data$X) < 8L || !is.numeric(data$y) || length(data$y) != nrow(data$X) ||
      any(!is.finite(data$y)) || length(data$experiment) != 1L || is.na(data$experiment)) stop("Invalid training data/experiment")
  data$y <- as.numeric(data$y)
  if (!is.null(data$surface)) data$surface$X <- .bl_matrix(data$surface$X, "surface$X", ncol(data$X))
  # Test inputs may lie outside [0,1] after training-only storm min/max scaling.
  if (method == "spgp" && (ncol(data$X) != 2L || !grepl("^(sim1|smooth2d)", data$experiment))) stop("SPGP is only supported for sim1/smooth2d, in 2D")
  if (method == "lagp" && !grepl("^storm", data$experiment)) stop("laGP is storm-only")
  if (method == "sgv" && !grepl("^(gfunction4d|gfunction_?4d|gfun4d)", data$experiment)) stop("SGV is gfunction4d-only")
  if (method == "tgp" && grepl("^storm", data$experiment)) stop("TGP is explicitly excluded as infeasible for storm")
  if (method == "deepgp" && grepl("^storm.*full", data$experiment)) stop("deepGP is explicitly excluded as infeasible for stormfull")
  s <- .bl_settings(data, settings)
  if (method %in% c("sgv", "spgp") && s$nu != 2.5) stop("The established SGV/SPGP baseline uses fixed nu=2.5")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  outdir <- normalizePath(outdir, mustWork = TRUE)
  existed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (existed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (existed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  tryCatch({
    dep <- baseline_dependencies(method, .bl_python(s))
    if (!dep$available) stop("Baseline unavailable: ", dep$detail)
    runner <- get(paste0(".bl_", method), mode = "function")
    result <- runner(data, s, outdir)
    if (!is.numeric(result$mean) || length(result$mean) != nrow(data$Xt) ||
        !is.matrix(result$draws) || !identical(dim(result$draws), c(nrow(data$Xt), 300L)) ||
        any(!is.finite(result$mean)) || any(!is.finite(result$draws)) ||
        any(!is.finite(c(result$fit_s, result$pred_s))) || any(c(result$fit_s, result$pred_s) < 0)) stop("Baseline violated the finite prediction contract")
    result$settings <- utils::modifyList(s, result$settings)
    result
  }, error = function(e) {
    failure <- list(method = method, experiment = data$experiment, rep = data$rep,
                    settings = s, error = conditionMessage(e))
    saveRDS(failure, file.path(outdir, paste0("baseline_", method, "_failure.rds")))
    stop("[", method, "] ", conditionMessage(e), call. = FALSE)
  })
}
