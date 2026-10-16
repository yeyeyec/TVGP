# Chains of the official tgp sampler on the datasets of Section 4.1 (Supplement C.6).
#
#   Rscript sampler_comparison/official_tgp.R fit <sim1|ns10_axis> <chain 1-10> [--out DIR]
#   Rscript sampler_comparison/official_tgp.R summarize <sim1|ns10_axis> [--out DIR]
#
# Each chain runs tgp::btgp (version 2.4-23) for 50,000 iterations and keeps the last 30,000
# states, with a constant mean, Matern correlation (nu = 2.5), tree prior (0.95, 0.5) and
# minimum leaf size 30. TGP samples GP parameters along with the tree, so its log posterior is
# not on the scale of the TVGP leaf scores. `summarize` gives rank-normalized R-hat and bulk
# and tail ESS across the ten chains (posterior package).

args <- commandArgs(trailingOnly = TRUE)
option <- function(name, default) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) default else args[i + 1L]
}
root <- normalizePath(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), ".."))
source(file.path(root, "R/datasets.R"))
command <- args[1]; example <- args[2]
out <- option("out", file.path(root, "results", "official_tgp", example))
burn <- 20000L; total <- 50000L

if (command == "fit") {
  chain <- as.integer(args[3])
  d <- make_dataset(paste0("sampler_", example), 1L)
  dir <- file.path(out, sprintf("chain%02d", chain)); dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  old <- setwd(dir); on.exit(setwd(old), add = TRUE)
  set.seed(940000L + 1000L * ncol(d$X) + chain)
  elapsed <- system.time(fit <- tgp::btgp(X = d$X, Z = d$y, XX = NULL, meanfn = "constant",
    bprior = "bflat", corr = "matern", nu = 2.5, tree = c(0.95, 0.5, 30), BTE = c(burn, total, 1L),
    R = 1L, m0r1 = TRUE, linburn = FALSE, itemps = NULL, trace = TRUE, pred.n = FALSE,
    krige = FALSE, zcov = FALSE, Ds2x = FALSE, improv = FALSE, verb = 1))[["elapsed"]]
  post <- as.data.frame(fit$trace$post)
  post$iteration <- seq.int(burn + 1L, total)
  utils::write.csv(post, file.path(dir, "trace.csv"), row.names = FALSE)
  utils::write.csv(data.frame(chain = chain, fit_wall_s = elapsed), file.path(dir, "summary.csv"), row.names = FALSE)
} else if (command == "summarize") {
  files <- sort(Sys.glob(file.path(out, "chain*", "trace.csv")))
  traces <- lapply(files, utils::read.csv)
  rows <- lapply(c("lpost", "height", "leaves"), function(nm) {
    x <- do.call(cbind, lapply(traces, `[[`, nm))       # iterations x chains
    data.frame(parameter = nm, chains = ncol(x), rhat = posterior::rhat(x),
               bulk_ess = posterior::ess_bulk(x), tail_ess = posterior::ess_tail(x))
  })
  z <- do.call(rbind, rows)
  utils::write.csv(z, file.path(out, "between_chain_diagnostics.csv"), row.names = FALSE)
  print(z)
} else stop("command must be fit or summarize")
