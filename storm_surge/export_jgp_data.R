#!/usr/bin/env Rscript
# Write one storm-surge split as CSV files for run_jgp.py.
#   Rscript storm_surge/export_jgp_data.R <n10000|full> <rep> train.csv.gz test.csv.gz [pilot]
# With "pilot", 250 training runs (whole design groups under the grouped holdout) form a
# validation set for choosing the JGP neighborhood size.
source(file.path(dirname(normalizePath(sub("^--file=", "", grep(
  "^--file=", commandArgs(FALSE), value = TRUE)[1L]))), "common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) stop("Usage: export_jgp_data.R <n10000|full> <rep> train.csv.gz test.csv.gz [pilot]")
data <- load_experiment_data(args[1L], as.integer(args[2L]))
pilot <- length(args) >= 5L && args[5L] == "pilot"

write_dataset <- function(path, X, y, truth) {
  frame <- as.data.frame(X)
  names(frame) <- paste0("x", seq_len(ncol(X)))
  frame$y_obs <- as.numeric(y)
  frame$truth <- as.numeric(truth)
  connection <- gzfile(path, open = "wt")
  on.exit(close(connection), add = TRUE)
  utils::write.csv(frame, connection, row.names = FALSE)
}

if (pilot) {
  split <- tuning_split(data, 250L, 96001L)
  write_dataset(args[3L], data$X[split$training, , drop = FALSE],
                data$y[split$training], data$y[split$training])
  write_dataset(args[4L], data$X[split$validation, , drop = FALSE],
                data$y[split$validation], data$y[split$validation])
} else {
  write_dataset(args[3L], data$X, data$y, data$y)
  write_dataset(args[4L], data$Xt, data$truth, data$truth)
}
