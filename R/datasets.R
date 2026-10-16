# Data-generating models and train/test designs for every experiment in the paper.
# make_dataset(experiment, rep) returns list(X, y, Xt, truth, ...), where truth is the
# latent function (or the simulator response for storm surge) at the test inputs.

# ---------------------------------------------------------------------------
# 2-dimensional piecewise example (Section 4)
# ---------------------------------------------------------------------------
sim1_truth <- function(X) {
  X <- as.matrix(X)
  x1 <- X[, 1L]
  x2 <- X[, 2L]
  mu <- numeric(nrow(X))
  sigma <- numeric(nrow(X))

  r1 <- x1 <= 0.30 & x2 > 0.65
  r2 <- x1 > 0.30 & x2 > 0.65
  r3 <- x1 <= 0.30 & x2 <= 0.65
  r4 <- !(r1 | r2 | r3)

  mu[r1] <- 0.8 * sin(6 * pi * x1[r1]) * cos(4 * pi * x2[r1])
  sigma[r1] <- 0.10
  mu[r2] <- 1.5 * (x1[r2] - 0.5) + (x2[r2] - 0.5)
  sigma[r2] <- 0.05
  mu[r3] <- 3 * (x1[r3] - 0.2)^2 + 2 * (x2[r3] - 0.3)^2 - 0.2
  sigma[r3] <- 0.12
  mu[r4] <- 0.1
  sigma[r4] <- 0.06

  list(mu = mu, sigma = sigma)
}

simulate_sim1 <- function(n, seed, lhs_method = "improved") {
  set.seed(seed)
  X <- make_unit_lhs(n, 2L, lhs_method)
  colnames(X) <- c("x1", "x2")
  truth <- sim1_truth(X)
  list(
    X = X,
    y = truth$mu + stats::rnorm(n, sd = truth$sigma),
    mu = truth$mu,
    sigma = truth$sigma
  )
}

# ---------------------------------------------------------------------------
# Smooth 2-dimensional example (Supplementary Section C.3)
# ---------------------------------------------------------------------------
smooth_truth <- function(X) {
  X <- as.matrix(X)
  X[, 1L] * exp(-(X[, 1L]^2 + X[, 2L]^2))
}

simulate_smooth <- function(n, seed, sigma = 0.04, lhs_method = "improved",
                            lower = -2, upper = 5) {
  set.seed(seed)
  U <- make_unit_lhs(n, 2L, lhs_method)
  X <- lower + (upper - lower) * U
  colnames(X) <- c("x1", "x2")
  mu <- smooth_truth(X)
  list(X = X, y = mu + stats::rnorm(n, sd = sigma), mu = mu)
}

to_unit_square <- function(X, lower = -2, upper = 5) {
  X <- as.matrix(X)
  out <- (X - lower) / (upper - lower)
  colnames(out) <- colnames(X)
  out
}

# ---------------------------------------------------------------------------
# 10-dimensional example (Section 4.4) and its rotated version (Supplement C.4)
# ---------------------------------------------------------------------------
ns10_truth <- function(X, version = c("axis", "rotated"), kappa = 60) {
  version <- match.arg(tolower(version), c("axis", "rotated"))
  X <- as.matrix(X)
  if (ncol(X) < 10L) stop("NS10 requires at least 10 input columns.")

  gate <- if (version == "axis") X[, 1L] else (X[, 1L] + X[, 2L]) / 2
  s1 <- stats::plogis(kappa * (gate - 0.30))
  s2 <- stats::plogis(kappa * (gate - 0.65))
  s3 <- stats::plogis(kappa * (X[, 2L] - 0.55))

  w1 <- 1 - s1
  w2 <- s1 - s2
  w3 <- s2

  base <-
    1.5 * (X[, 2L] - 0.5) +
    2.0 * (X[, 3L] - 0.5)^2 +
    0.75 * sin(2 * pi * X[, 4L] * X[, 6L])
  h1 <- 0.6 * sin(2 * pi * X[, 5L])
  h2 <- sin(6 * pi * X[, 5L]) + 0.35 * cos(2 * pi * X[, 2L])
  h3 <-
    1.4 * sin(16 * pi * X[, 5L]) +
    0.5 * ((1 - s3) * cos(4 * pi * X[, 4L]) +
             s3 * cos(12 * pi * X[, 4L]))

  as.numeric(base + w1 * h1 + w2 * h2 + w3 * h3)
}

simulate_ns10 <- function(n, seed, version, sigma = 0.05,
                          lhs_method = "maximin", kappa = 60) {
  set.seed(seed)
  X <- make_unit_lhs(n, 10L, lhs_method)
  colnames(X) <- paste0("x", seq_len(10L))
  mu <- ns10_truth(X, version, kappa)
  set.seed(seed + 20000L)
  list(X = X, y = mu + stats::rnorm(n, sd = sigma), mu = mu)
}

# ---------------------------------------------------------------------------
# Deterministic benchmarks for the sampler diagnostics (Supplement C.7)
# ---------------------------------------------------------------------------
bench_function <- function(X) {
  X <- as.matrix(X)
  d <- ncol(X)
  if (d == 2L) {                                    # Branin-Hoo
    x1 <- 15 * X[, 1L] - 5
    x2 <- 15 * X[, 2L]
    return((x2 - 5.1 * x1^2 / (4 * pi^2) + 5 * x1 / pi - 6)^2 +
             10 * (1 - 1 / (8 * pi)) * cos(x1) + 10)
  }
  if (d == 5L) {                                    # Friedman
    return(10 * sin(pi * X[, 1L] * X[, 2L]) + 20 * (X[, 3L] - 0.5)^2 +
             10 * X[, 4L] + 5 * X[, 5L])
  }
  if (d == 10L) {                                   # Sobol G-function
    a <- (seq_len(d) - 1) / 2
    terms <- sweep(abs(4 * X - 2), 2L, a, "+")
    terms <- sweep(terms, 2L, 1 + a, "/")
    return(apply(terms, 1L, prod))
  }
  if (d == 20L) {                                   # Welch et al. (1992)
    Z <- X - 0.5
    return(5 * Z[, 12L] / (1 + Z[, 1L]) +
      5 * (Z[, 4L] - Z[, 20L])^2 + Z[, 5L] +
      40 * Z[, 19L]^3 - 5 * Z[, 19L] +
      0.05 * Z[, 2L] + 0.08 * Z[, 3L] - 0.03 * Z[, 6L] +
      0.03 * Z[, 7L] - 0.09 * Z[, 9L] - 0.01 * Z[, 10L] -
      0.07 * Z[, 11L] + 0.25 * Z[, 13L]^2 - 0.04 * Z[, 14L] +
      0.06 * Z[, 15L] - 0.01 * Z[, 17L] - 0.03 * Z[, 18L])
  }
  stop("Benchmarks are defined for 2, 5, 10, and 20 inputs.")
}

simulate_bench <- function(dimension, n_train, n_test, seed) {
  set.seed(seed); X <- make_unit_lhs(n_train, dimension, "random")
  set.seed(seed + 10000L); Xt <- make_unit_lhs(n_test, dimension, "random")
  colnames(X) <- colnames(Xt) <- paste0("x", seq_len(dimension))
  y <- as.numeric(bench_function(X)); yt <- as.numeric(bench_function(Xt))
  center <- mean(y); scale <- stats::sd(y)          # standardized with the training response
  list(X = X, y = (y - center) / scale, Xt = Xt, truth = (yt - center) / scale)
}

# ---------------------------------------------------------------------------
# Exact-zero examples (Supplement C.8)
# ---------------------------------------------------------------------------
simulate_exact_zero <- function(experiment, rep, n = 300L, n_test = 1000L) {
  q <- if (experiment == "zero30") 0.3 else 0.7
  # zero70 and noise70 share inputs and noise draws.
  set.seed(810000L + rep); X <- lhs::randomLHS(n, 2); eps <- stats::rnorm(n, sd = 0.03)
  set.seed(820000L + rep); Xt <- lhs::randomLHS(n_test, 2); eps_t <- stats::rnorm(n_test, sd = 0.03)
  f <- function(x) ifelse(x[, 1] > q, 0.4 + 0.3 * sin(2 * pi * x[, 2])^2 +
                            0.3 * (x[, 1] - q) / (1 - q), 0)
  noisy_all <- experiment == "noise70"
  y <- f(X) + if (noisy_all) eps else eps * (X[, 1] > q)
  list(X = X, y = y, Xt = Xt, truth = f(Xt), q = q,
       group = ifelse(Xt[, 1] > q, "positive", "zero"),
       boundary = abs(Xt[, 1] - q) <= 0.05)
}

# ---------------------------------------------------------------------------
# Storm surge (Section 5). The ADCIRC archive is not distributed with this code.
# STORMSURGE_DATA_PATH must point to an .RData file containing
#   dat        runs x locations matrix of peak surge elevation (m),
#   inputs     runs x 7 matrix (Delta P, R_p, V_f, theta, B, landfall lon, landfall lat),
#   lonlat.sub locations x 2 matrix of longitude and latitude.
# The paper uses location 500.
# ---------------------------------------------------------------------------
load_storm_location <- function(path = Sys.getenv("STORMSURGE_DATA_PATH"), location = 500L) {
  if (!nzchar(path) || !file.exists(path)) stop("Set STORMSURGE_DATA_PATH to the storm-surge .RData file.")
  e <- new.env()
  load(path, envir = e)
  list(X = as.matrix(e$inputs), y = as.numeric(e$dat[, location]),
       location = as.numeric(e$lonlat.sub[location, ]))
}

# Random holdout of Section 5.2: the 10,000-run training set is contained in the
# 50,649-run training set, and the 1,000-run test set in the 5,628-run test set.
storm_split <- function(d, rep, full = FALSE) {
  nr <- nrow(d$X)
  set.seed(rep); o <- sample.int(nr)
  small_train <- o[seq_len(10000L)]
  small_test <- o[10000L + seq_len(1000L)]
  full_test <- o[10000L + seq_len(as.integer(round(nr * 0.1)))]
  full_train <- c(small_train, setdiff(seq_len(nr), c(small_train, full_test)))
  tr <- if (full) full_train else small_train
  te <- if (full) full_test else small_test
  X <- d$X[tr, , drop = FALSE]; Xt <- d$X[te, , drop = FALSE]
  lo <- apply(X, 2, min); span <- apply(X, 2, max) - lo   # scale with training inputs only
  list(X = sweep(sweep(X, 2, lo, "-"), 2, span, "/"),
       Xt = sweep(sweep(Xt, 2, lo, "-"), 2, span, "/"),
       y = d$y[tr], truth = d$y[te], train_id = tr, test_id = te)
}

# Grouped holdout of Supplement C.8: all landfall realizations of a five-dimensional storm
# design (the first five inputs) fall on the same side of the split. Inputs are scaled by
# their ranges over all runs; design groups are numbered in lexicographic order.
storm_grouped_master <- function(d) {
  lo <- apply(d$X, 2, min); span <- apply(d$X, 2, max) - lo
  X <- sweep(sweep(d$X, 2, lo, "-"), 2, span, "/")
  o <- do.call(order, as.data.frame(X[, 1:5]))
  key <- do.call(paste, c(as.data.frame(signif(X[, 1:5], 15)), sep = "|"))
  ukey <- unique(key[o]); group_id <- match(key, ukey)
  list(X = X, y = d$y, group_id = group_id,
       group_design = X[match(seq_along(ukey), group_id), 1:5, drop = FALSE])
}

# Greedy maximin subset of the rows of X within pool.
farthest_subset <- function(X, k, seed, pool = seq_len(nrow(X))) {
  pool <- as.integer(pool)
  set.seed(seed)
  first <- sample(pool, 1L)
  selected <- integer(k); selected[1L] <- first
  min_d2 <- rowSums((X[pool, , drop = FALSE] -
                     matrix(X[first, ], nrow = length(pool), ncol = ncol(X), byrow = TRUE))^2)
  min_d2[match(first, pool)] <- -Inf
  if (k > 1L) {
    jitter <- stats::runif(length(pool), 0, 1e-12)
    for (i in 2:k) {
      position <- which.max(min_d2 + jitter)
      selected[i] <- pool[position]
      distance <- rowSums((X[pool, , drop = FALSE] -
                           matrix(X[selected[i], ], nrow = length(pool), ncol = ncol(X), byrow = TRUE))^2)
      min_d2 <- pmin(min_d2, distance)
      min_d2[match(selected[seq_len(i)], pool)] <- -Inf
    }
  }
  selected
}

# Groups holding the two smallest and two largest values of each input stay in training.
protected_extrema <- function(X, group_id, groups, n_each = 2L) {
  rows <- which(group_id %in% groups)
  unique(unlist(lapply(seq_len(ncol(X)), function(j) {
    group_min <- tapply(X[rows, j], group_id[rows], min)
    group_max <- tapply(X[rows, j], group_id[rows], max)
    as.integer(c(names(sort(group_min))[seq_len(min(n_each, length(group_min)))],
                 names(sort(group_max, decreasing = TRUE))[seq_len(min(n_each, length(group_max)))]))
  }), use.names = FALSE))
}

# Full version: 3,150 training and 350 test groups (about 50,650 and 5,630 runs). Smaller
# version: 690 maximin-selected groups split 621/69 (about 10,000 training runs).
storm_grouped_split <- function(master, rep, full = FALSE) {
  seed <- 926000L + rep + if (full) 1000L else 0L
  design <- master$group_design
  if (full) {
    selected <- seq_len(nrow(design)); n_test_groups <- 350L
  } else {
    selected <- farthest_subset(design, 690L, seed + 11L); n_test_groups <- 69L
  }
  protected <- protected_extrema(master$X, master$group_id, selected, 2L)
  test_groups <- farthest_subset(design, n_test_groups, seed + 29L, setdiff(selected, protected))
  train_groups <- setdiff(selected, test_groups)
  tr <- which(master$group_id %in% train_groups); te <- which(master$group_id %in% test_groups)
  list(X = master$X[tr, , drop = FALSE], y = master$y[tr],
       Xt = master$X[te, , drop = FALSE], truth = master$y[te],
       train_id = tr, test_id = te, train_group_id = master$group_id[tr],
       test_group_id = master$group_id[te])
}

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
make_unit_lhs <- function(n, d, method = c("improved", "maximin", "random")) {
  method <- match.arg(method)
  switch(method,
         improved = lhs::improvedLHS(n, d),
         maximin = lhs::maximinLHS(n, d),
         random = lhs::randomLHS(n, d))
}

experiment_design <- data.frame(
  experiment = c("sim1", "smooth2d", "ns10_axis", "ns10_rotated",
                 "ablation_n500", "ablation_n1000", "ablation_n2000",
                 "sampler_sim1", "sampler_ns10_axis",
                 "bench_branin2", "bench_friedman5", "bench_g10", "bench_welch20",
                 "zero30", "zero70", "noise70", "storm_n10000", "storm_full",
                 "storm_grouped_n10000", "storm_grouped_full"),
  n_train = c(10000, 6000, 2000, 2000, 500, 1000, 2000, 5000, 5000,
              5000, 5000, 5000, 5000, 300, 300, 300, 10000, 50649, NA, NA),
  n_test = c(2000, 1000, 400, 400, 2000, 2000, 2000, 2000, 2000,
             2000, 2000, 2000, 2000, 1000, 1000, 1000, 1000, 5628, NA, NA),
  m = c(30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 70, 70, 70, 70),
  stringsAsFactors = FALSE)

# size = c(n_train, n_test) overrides the paper's sizes (for quick checks only).
make_dataset <- function(experiment, rep, size = NULL) {
  spec <- experiment_design[experiment_design$experiment == experiment, ]
  if (nrow(spec) != 1L) stop("Unknown experiment: ", experiment)
  n <- spec$n_train; nt <- spec$n_test
  if (!is.null(size)) { n <- as.integer(size[1]); nt <- as.integer(size[2]) }
  if (experiment == "sim1" || startsWith(experiment, "ablation_") || experiment == "sampler_sim1") {
    lhs_method <- if (experiment == "sampler_sim1") "random" else "improved"
    train <- simulate_sim1(n, rep, lhs_method); test <- simulate_sim1(nt, rep + 10000L, lhs_method)
    d <- list(X = train$X, y = train$y, Xt = test$X, truth = test$mu)
  } else if (experiment == "smooth2d") {
    train <- simulate_smooth(n, rep); test <- simulate_smooth(nt, rep + 10000L)
    d <- list(X = to_unit_square(train$X), y = train$y, Xt = to_unit_square(test$X), truth = test$mu)
  } else if (experiment %in% c("ns10_axis", "ns10_rotated", "sampler_ns10_axis")) {
    version <- if (experiment == "ns10_rotated") "rotated" else "axis"
    lhs_method <- if (experiment == "sampler_ns10_axis") "random" else "maximin"
    train <- simulate_ns10(n, rep, version, lhs_method = lhs_method)
    test <- simulate_ns10(nt, rep + 10000L, version, lhs_method = lhs_method)
    d <- list(X = train$X, y = train$y, Xt = test$X, truth = test$mu)
  } else if (startsWith(experiment, "bench_")) {
    dimension <- as.integer(sub("^bench_[a-z]+", "", experiment))
    d <- simulate_bench(dimension, n, nt, 730000L + dimension)
  } else if (experiment %in% c("zero30", "zero70", "noise70")) {
    d <- simulate_exact_zero(experiment, rep, n, nt)
  } else if (startsWith(experiment, "storm_grouped_")) {
    d <- storm_grouped_split(storm_grouped_master(load_storm_location()), rep,
                             full = experiment == "storm_grouped_full")
  } else if (startsWith(experiment, "storm_")) {
    d <- storm_split(load_storm_location(), rep, full = experiment == "storm_full")
  } else stop("No generator for ", experiment)
  colnames(d$X) <- colnames(d$Xt) <- paste0("x", seq_len(ncol(d$X)))
  d$experiment <- experiment; d$rep <- rep; d$m <- spec$m
  d
}
