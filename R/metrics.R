# Scores for latent prediction: RMSE of the predictive mean, and CRPS, coverage, width, and
# interval score of 95% intervals computed from predictive draws (300 per test input).
metric_values <- function(truth, mean, draws) {
  if (length(mean) != length(truth) || any(!is.finite(mean))) stop("Invalid predictive mean.")
  if (!is.matrix(draws) || nrow(draws) != length(truth) || ncol(draws) < 20L || any(!is.finite(draws)))
    stop("Invalid predictive draws.")
  qs <- t(apply(draws, 1, stats::quantile, probs = c(.025, .975), names = FALSE))
  ordered <- t(apply(draws, 1, sort)); k <- ncol(draws)
  crps <- rowMeans(abs(draws - truth)) - as.vector(ordered %*% (2 * seq_len(k) - k - 1)) / k^2
  c(RMSE = sqrt(mean((mean - truth)^2)), CRPS = mean(crps),
    Cover95 = mean(truth >= qs[, 1] & truth <= qs[, 2]), MeanCI95 = mean(qs[, 2] - qs[, 1]),
    IntervalScore95 = mean(qs[, 2] - qs[, 1] + 40 * pmax(qs[, 1] - truth, 0) +
                             40 * pmax(truth - qs[, 2], 0)))
}

# Scores on subsets of the test set (exact-zero examples: zero, positive, boundary strip).
group_metric_values <- function(data, pred) {
  groups <- list(all = seq_along(data$truth))
  if (!is.null(data$group)) {
    groups$zero <- which(data$group == "zero")
    groups$positive <- which(data$group == "positive")
    groups$boundary <- which(data$boundary)
  }
  do.call(rbind, lapply(names(groups), function(g) {
    i <- groups[[g]]
    if (!length(i)) return(NULL)
    data.frame(group = g, n_test = length(i),
               t(metric_values(data$truth[i], pred$mean[i], pred$draws[i, , drop = FALSE])))
  }))
}
