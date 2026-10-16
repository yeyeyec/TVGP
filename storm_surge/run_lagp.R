#!/usr/bin/env Rscript

source(file.path(dirname(normalizePath(sub("^--file=", "", grep(
  "^--file=", commandArgs(FALSE), value = TRUE)[1L]))), "common.R"))
required <- c("laGP", "FNN")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
verify_fixed_data()

estimate_global_scaling <- function(X, y, seed, sample_n = 1000L) {
  set.seed(seed)
  index <- sample.int(nrow(X), min(sample_n, nrow(X)))
  Xs <- X[index, , drop = FALSE]
  ys <- y[index]
  da <- laGP::darg(NULL, Xs)
  ga <- laGP::garg(list(mle = TRUE), ys)
  initial_d <- as.numeric(da$start)
  if (length(initial_d) == 1L) initial_d <- rep(initial_d, ncol(Xs))
  if (length(initial_d) != ncol(Xs)) stop("Invalid darg starting values.")
  gp <- laGP::newGPsep(Xs, ys, d = initial_d, g = ga$start, dK = TRUE)
  on.exit(laGP::deleteGPsep(gp), add = TRUE)
  fit <- laGP::jmleGPsep(
    gp, drange = c(da$min, da$max), grange = c(ga$min, ga$max),
    dab = da$ab, gab = ga$ab, maxit = 100L, verb = 0
  )
  d_names <- paste0("d.", seq_len(ncol(Xs)))
  d <- as.numeric(fit[1L, d_names, drop = TRUE])
  g <- as.numeric(fit$g[1L])
  if (!isTRUE(as.integer(fit$dconv[1L]) == 0L)) {
    stop("Global separable lengthscale optimization did not converge.")
  }
  if (length(d) != ncol(X) || any(!is.finite(d)) || any(d <= 0) ||
      length(g) != 1L || !is.finite(g) || g <= 0) {
    stop("Global separable GP calibration returned invalid parameters.")
  }
  list(d = d, g = g, fit = fit, sample_id = index)
}

latent_from_local_design <- function(X, y, x, index, d, g, seed) {
  local_y <- y[index]
  if (diff(range(local_y)) <= 1e-12) {
    return(list(mean = local_y[1L], variance = 0, scale2 = 0,
                df = length(local_y), constant = TRUE, fallback = FALSE))
  }
  if (!is.finite(d) || d <= 0) d <- 1
  if (!is.finite(g) || g <= 0) g <- 1e-4
  set.seed(seed)
  gp <- laGP::newGP(X[index, , drop = FALSE], local_y, d = d, g = g, dK = FALSE)
  on.exit(laGP::deleteGP(gp), add = TRUE)
  prediction <- laGP::predGP(gp, x, lite = TRUE, nonug = TRUE)
  if (!is.finite(prediction$df) || prediction$df <= 2 ||
      !is.finite(prediction$s2) || prediction$s2 < 0 ||
      any(!is.finite(prediction$mean))) stop("Invalid local Student-t prediction.")
  list(
    mean = as.numeric(prediction$mean),
    variance = as.numeric(prediction$s2 * prediction$df / (prediction$df - 2)),
    scale2 = as.numeric(prediction$s2), df = prediction$df,
    constant = FALSE, fallback = FALSE
  )
}

predict_lagp <- function(X, y, Xt, config, global, seed, cores, n_draws = 300L) {
  Xscaled <- sweep(X, 2L, sqrt(global$d), "/")
  Xtscaled <- sweep(Xt, 2L, sqrt(global$d), "/")
  da <- laGP::darg(list(mle = TRUE, start = 1), Xscaled)
  ga <- laGP::garg(list(mle = TRUE, start = global$g), y)
  chunks <- split(seq_len(nrow(Xt)), ceiling(seq_len(nrow(Xt)) / 400L))
  mean <- variance <- scale2 <- df <- rep(NA_real_, nrow(Xt))
  constant <- fallback <- rep(FALSE, nrow(Xt))
  design_s <- reconstruction_s <- 0

  for (rows in chunks) {
    local <- NULL
    elapsed <- system.time({
      local <- tryCatch(
        laGP::aGP(
          Xscaled, y, Xtscaled[rows, , drop = FALSE], start = 6L,
          end = config$end, d = da, g = ga, method = config$method,
          Xi.ret = TRUE, omp.threads = cores, verb = 0
        ), error = function(error) error
      )
    })[["elapsed"]]
    design_s <- design_s + elapsed
    if (inherits(local, "error")) {
      # Isolate a problematic chunk without changing the local-design rule.
      fallback_elapsed <- system.time({
        local_rows <- lapply(rows, function(i) {
          tryCatch(laGP::laGP(
            Xtscaled[i, , drop = FALSE], 6L, config$end, Xscaled, y,
            d = da, g = ga, method = config$method, Xi.ret = TRUE,
            lite = TRUE, verb = 0
          ), error = function(error) error)
        })
      })[["elapsed"]]
      design_s <- design_s + fallback_elapsed
      Xi <- matrix(NA_integer_, length(rows), config$end)
      d_mle <- g_mle <- rep(NA_real_, length(rows))
      for (j in seq_along(rows)) {
        if (!inherits(local_rows[[j]], "error")) {
          Xi[j, ] <- local_rows[[j]]$Xi
          d_mle[j] <- local_rows[[j]]$mle$d
          g_mle[j] <- local_rows[[j]]$mle$g
        }
      }
    } else {
      Xi <- local$Xi
      d_mle <- if (!is.null(local$mle$d)) local$mle$d else rep(NA_real_, length(rows))
      g_mle <- if (!is.null(local$mle$g)) local$mle$g else rep(global$g, length(rows))
    }

    reconstruct_one <- function(j) {
      i <- rows[j]
      index <- Xi[j, ]
      index <- as.integer(index[is.finite(index) & index >= 1L & index <= nrow(X)])
      if (!length(index)) {
        index <- FNN::get.knnx(Xscaled, Xtscaled[i, , drop = FALSE],
                               k = config$end)$nn.index[1L, ]
        d_value <- 1; g_value <- global$g
        used_fallback <- TRUE
      } else {
        d_value <- d_mle[j]; g_value <- g_mle[j]
        used_fallback <- !is.finite(d_value) || !is.finite(g_value)
      }
      answer <- tryCatch(
        latent_from_local_design(
          Xscaled, y, Xtscaled[i, , drop = FALSE], index,
          d_value, g_value, seed + i
        ), error = function(error) NULL
      )
      if (is.null(answer)) {
        index <- FNN::get.knnx(Xscaled, Xtscaled[i, , drop = FALSE],
                               k = config$end)$nn.index[1L, ]
        answer <- latent_from_local_design(
          Xscaled, y, Xtscaled[i, , drop = FALSE], index,
          1, global$g, seed + i
        )
        used_fallback <- TRUE
      }
      answer$fallback <- used_fallback
      answer
    }
    # aGP uses OpenMP above. Forking R workers after an OpenMP region can
    # deadlock on cluster runtimes, so reconstruct these small local GPs in
    # the parent process.
    elapsed <- system.time({
      answers <- lapply(seq_along(rows), reconstruct_one)
    })[["elapsed"]]
    reconstruction_s <- reconstruction_s + elapsed
    for (j in seq_along(rows)) {
      i <- rows[j]; answer <- answers[[j]]
      mean[i] <- answer$mean; variance[i] <- answer$variance
      scale2[i] <- answer$scale2; df[i] <- answer$df
      constant[i] <- answer$constant; fallback[i] <- answer$fallback
    }
  }
  if (any(!is.finite(mean)) || any(!is.finite(variance)) || any(variance < 0)) {
    stop("laGP produced invalid latent moments.")
  }
  set.seed(seed + 100000L)
  draws <- matrix(NA_real_, nrow(Xt), n_draws)
  for (i in seq_len(nrow(Xt))) {
    if (scale2[i] == 0) draws[i, ] <- mean[i]
    else draws[i, ] <- mean[i] + sqrt(scale2[i]) * stats::rt(n_draws, df = df[i])
  }
  list(mean = mean, variance = variance, draws = draws, df = df,
       constant = constant, fallback = fallback,
       design_s = design_s, reconstruction_s = reconstruction_s)
}

tune_lagp <- function(cores) {
  out <- method_output("lagp")
  data <- load_experiment_data("n10000", 1L)
  split <- tuning_split(data, 300L, 91001L)
  training <- split$training; validation <- split$validation
  ys <- scale_response(data$y[training])
  global <- estimate_global_scaling(data$X[training, , drop = FALSE], ys$y, 92001L)
  configs <- data.frame(
    id = c("alc50", "alc100", "nn100"),
    method = c("alc", "alc", "nn"), end = c(50L, 100L, 100L),
    stringsAsFactors = FALSE
  )
  rows <- lapply(seq_len(nrow(configs)), function(i) {
    config <- configs[i, ]
    prediction <- predict_lagp(
      data$X[training, , drop = FALSE], ys$y,
      data$X[validation, , drop = FALSE], config, global,
      93000L + i, cores, 300L
    )
    mean <- ys$center + ys$scale * prediction$mean
    variance <- ys$scale^2 * prediction$variance
    draws <- ys$center + ys$scale * prediction$draws
    cbind(config, score_draws(data$y[validation], mean, variance, draws),
          constant_fraction = mean(prediction$constant),
          fallback_fraction = mean(prediction$fallback),
          time_s = prediction$design_s + prediction$reconstruction_s)
  })
  tuning <- do.call(rbind, rows)
  tuning <- tuning[order(tuning$CRPS, tuning$RMSE), ]
  write_csv_atomic(tuning, file.path(out, "training_only_tuning.csv"))
  selected <- tuning[1L, c("id", "method", "end")]
  write_csv_atomic(selected, file.path(out, "selected_configuration.csv"))
  selected
}

run_one <- function(data_version, rep, config, cores) {
  out <- method_output("lagp", data_version, rep)
  data <- load_experiment_data(data_version, rep)
  ys <- scale_response(data$y)
  fit_s <- system.time({
    global <- estimate_global_scaling(data$X, ys$y, 94000L + rep +
      if (data_version == "full") 1000L else 0L)
  })[["elapsed"]]
  prediction <- predict_lagp(
    data$X, ys$y, data$Xt, config, global,
    95000L + rep + if (data_version == "full") 1000L else 0L,
    cores, 300L
  )
  mean <- ys$center + ys$scale * prediction$mean
  variance <- ys$scale^2 * prediction$variance
  draws <- ys$center + ys$scale * prediction$draws
  metrics <- score_draws(data$truth, mean, variance, draws)
  row <- cbind(data.frame(
    method = "laGP", data_version = data_version, rep = rep,
    n_train = nrow(data$X), n_test = nrow(data$Xt),
    tuning_id = config$id, local_method = config$method,
    local_start = 6L, local_end = config$end,
    global_scaling_sample = min(1000L, nrow(data$X)),
    fit_s = unname(fit_s + prediction$design_s),
    pred_s = unname(prediction$reconstruction_s),
    Time_s = unname(fit_s + prediction$design_s + prediction$reconstruction_s),
    constant_fraction = mean(prediction$constant),
    fallback_fraction = mean(prediction$fallback),
    stringsAsFactors = FALSE
  ), metrics)
  write_csv_atomic(row, file.path(out, "metrics.csv"))
  saveRDS(data.frame(
    row_id = data$source_ids$test_id, truth = data$truth, mean = mean,
    latent_variance = variance, df = prediction$df,
    constant_local_design = prediction$constant,
    numerical_fallback = prediction$fallback
  ), file.path(out, "predictions.rds"), compress = FALSE)
  saveRDS(global, file.path(out, "global_scaling.rds"), compress = FALSE)
  row
}

cores <- as.integer(Sys.getenv("CORES", "16"))
if (!is.finite(cores) || cores < 1L) cores <- 1L
config <- tune_lagp(cores)
grid <- data_grid()
results <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  results[[i]] <- tryCatch(
    run_one(grid$data_version[i], grid$rep[i], config, cores),
    error = function(error) {
      out <- method_output("lagp", grid$data_version[i], grid$rep[i])
      failure <- data.frame(method = "laGP", data_version = grid$data_version[i],
                            rep = grid$rep[i], error = conditionMessage(error))
      write_csv_atomic(failure, file.path(out, "failure.csv")); failure
    }
  )
}
failed <- vapply(results, function(x) "error" %in% names(x), logical(1))
if (any(failed)) stop(sum(failed), " laGP runs failed; inspect per-run failure.csv files.")
