# Run one method on one replication of one experiment and write metrics.csv.
#
#   Rscript experiments/run.R <experiment> <method> <rep> [options]
#
# experiment  sim1, smooth2d, ns10_axis, ns10_rotated, ablation_n500, ablation_n1000,
#             ablation_n2000, zero30, zero70, noise70, storm_n10000, storm_full
# method      tvgp | global_vecchia | treed_dense | global_dense (TVGP and ablation arms)
#             tgp | deepgp | scaled_vecchia | spgp | jumpgp | lagp | sgv (competitors)
# rep         replication 1-10; it sets the data seed and the fitting seeds.
# options     --particles N   number of SMC particles (default 5000; Supplement C.2 varies it)
#             --cores C       CPU cores (default 16)
#             --jr a,b,g_min  jointly robust prior settings (default 0.2,1,1e-6; Supplement C.5)
#             --out DIR       output folder (default results/<experiment>/<method>/rep<rep>)
#             --size n,nt     smaller training and test sets, for a quick check of the setup
#             --data-rep R    dataset replication, when it differs from rep (Supplement C.2 fits
#                             ten seeds to the dataset of replication 1)
#
# Storm surge requires STORMSURGE_DATA_PATH (see README); its competitors are in storm_surge/.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) stop("Usage: Rscript experiments/run.R <experiment> <method> <rep> [options]")
experiment <- args[1]; method <- args[2]; rep <- as.integer(args[3])
option <- function(name, default) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) default else args[i + 1L]
}
root <- normalizePath(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
source(file.path(root, "R/datasets.R"))
source(file.path(root, "R/metrics.R"))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

size <- option("size", NA)
data <- make_dataset(experiment, as.integer(option("data-rep", rep)),
                     if (is.na(size)) NULL else as.integer(strsplit(size, ",")[[1]]))
cores <- as.integer(option("cores", 16L))
out <- option("out", file.path(root, "results", experiment, method, sprintf("rep%03d", rep)))
dir.create(out, recursive = TRUE, showWarnings = FALSE)

if (method %in% c("tvgp", "global_vecchia", "treed_dense", "global_dense")) {
  if (startsWith(experiment, "storm_")) options(tvgp.parallel_backend = "threads")
  source(file.path(root, "tvgp/tvgp_smc.R"))
  if (method %in% c("treed_dense", "global_dense")) source(file.path(root, "tvgp/dense_backend.R"))
  source(file.path(root, "R/fit_tvgp.R"))
  s <- tvgp_settings(experiment, rep, m = data$m,
                     n_particles = as.integer(option("particles", 5000L)), cores = cores)
  jr <- option("jr", NA)
  if (!is.na(jr)) {
    v <- as.numeric(strsplit(jr, ",")[[1]])
    s$cov_prior_control[c("jr_a", "jr_b", "g_min")] <- as.list(v)
  }
  result <- run_tvgp(method, data, s, out)
  predictions <- result$predictions
} else {
  source(file.path(root, "R/competitors.R"))
  data$seed_fit <- 50000L + rep; data$seed_predict <- 21000L + rep
  result <- run_baseline(method, data, list(m = data$m, nu = 2.5, cores = cores), out)
  predictions <- list(pointwise = result[c("mean", "draws", "fit_s", "pred_s")])
}

metrics <- do.call(rbind, lapply(names(predictions), function(mode) {
  p <- predictions[[mode]]
  data.frame(experiment = experiment, method = method, rep = rep, prediction_mode = mode,
             group_metric_values(data, p), Fit_s = p$fit_s, Pred_s = p$pred_s,
             Time_s = p$fit_s + p$pred_s, n_train = nrow(data$X))
}))
utils::write.csv(metrics, file.path(out, "metrics.csv"), row.names = FALSE)
saveRDS(predictions, file.path(out, "predictions.rds"))
print(metrics[metrics$group == "all", ])
