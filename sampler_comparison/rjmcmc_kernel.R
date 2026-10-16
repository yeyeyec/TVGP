# Vendored finite-grid kernel and QC helpers from the 2026-08-29 bundle.
# Leaf scoring and production run loops deliberately live in R/samplers.R.
SVR_make_split_grid <- function(X, grid_size = 31L, include_cuts = list(),
                                type = c("quantile", "fixed"),
                                digits = 12L) {
  type <- match.arg(type)
  X <- as.matrix(X)
  p <- ncol(X)
  grid <- vector("list", p)
  for (j in seq_len(p)) {
    base <- if (type == "fixed") {
      seq(0, 1, length.out = grid_size + 2L)[-c(1L, grid_size + 2L)]
    } else {
      probs <- seq(0, 1, length.out = grid_size + 2L)
      as.numeric(stats::quantile(X[, j], probs = probs[-c(1L, length(probs))],
                                 names = FALSE, type = 1))
    }
    extra <- NULL
    if (!is.null(names(include_cuts))) {
      extra <- include_cuts[[paste0("x", j)]]
    } else if (length(include_cuts) >= j) {
      extra <- include_cuts[[j]]
    }
    cuts <- sort(unique(round(c(base, extra), digits)))
    cuts <- cuts[is.finite(cuts) & cuts > 0 & cuts < 1]
    grid[[j]] <- cuts
  }
  names(grid) <- colnames(X)
  grid
}

## -----------------------------------------------------------------------------
## Tree representation and validation
## -----------------------------------------------------------------------------

SVR_make_leaf_node <- function(depth, lower, upper, indices, stopped = FALSE) {
  list(
    depth = as.integer(depth),
    lower = as.numeric(lower),
    upper = as.numeric(upper),
    indices = as.integer(indices),
    stopped = isTRUE(stopped),
    split_dim = NA_integer_,
    split_val = NA_real_,
    left = NULL,
    right = NULL
  )
}

SVR_is_leaf <- function(node) is.null(node$left) && is.null(node$right)

SVR_make_root_tree <- function(X, stopped = FALSE) {
  p <- ncol(X)
  list(root = SVR_make_leaf_node(0L, rep(0, p), rep(1, p), seq_len(nrow(X)), stopped))
}

SVR_copy_tree <- function(tree) tree # R copy-on-modify; no serialized deep copies.

SVR_path_string <- function(path) {
  if (length(path) == 0L) "root" else paste(path, collapse = "")
}

SVR_get_node <- function(node, path) {
  if (length(path) == 0L) return(node)
  side <- path[1L]
  child <- if (side == "L") node$left else node$right
  SVR_get_node(child, path[-1L])
}

SVR_set_node <- function(node, path, value) {
  if (length(path) == 0L) return(value)
  side <- path[1L]
  if (side == "L") {
    node$left <- SVR_set_node(node$left, path[-1L], value)
  } else {
    node$right <- SVR_set_node(node$right, path[-1L], value)
  }
  node
}

SVR_region_signature <- function(lower, upper, digits = 12L) {
  paste(
    sprintf("x%d:(%s,%s]", seq_along(lower),
            formatC(lower, digits = digits, format = "fg"),
            formatC(upper, digits = digits, format = "fg")),
    collapse = "|"
  )
}

SVR_tree_signature_node <- function(node, digits = 10L) {
  if (SVR_is_leaf(node)) {
    return(paste0("Leaf{", SVR_region_signature(node$lower, node$upper, digits), "}"))
  }
  paste0(
    "Split{x", node$split_dim, "<=", formatC(node$split_val, digits = digits, format = "fg"),
    ";", SVR_tree_signature_node(node$left, digits),
    ";", SVR_tree_signature_node(node$right, digits), "}"
  )
}

SVR_tree_signature <- function(tree, digits = 10L) {
  SVR_tree_signature_node(tree$root, digits)
}

SVR_collect_paths <- function(node, path = character(0), kind = c("leaf", "internal", "all")) {
  kind <- match.arg(kind)
  rows <- list()
  keep <- kind == "all" || (kind == "leaf" && SVR_is_leaf(node)) ||
    (kind == "internal" && !SVR_is_leaf(node))
  if (keep) rows[[length(rows) + 1L]] <- list(path = path, node = node)
  if (!SVR_is_leaf(node)) {
    rows <- c(rows, SVR_collect_paths(node$left, c(path, "L"), kind))
    rows <- c(rows, SVR_collect_paths(node$right, c(path, "R"), kind))
  }
  rows
}

SVR_leaf_paths <- function(tree) lapply(SVR_collect_paths(tree$root, kind = "leaf"), `[[`, "path")
SVR_internal_paths <- function(tree) lapply(SVR_collect_paths(tree$root, kind = "internal"), `[[`, "path")

SVR_leaf_nodes <- function(tree) lapply(SVR_collect_paths(tree$root, kind = "leaf"), `[[`, "node")

SVR_tree_size <- function(tree) {
  nodes <- SVR_collect_paths(tree$root, kind = "all")
  leaf <- vapply(nodes, function(z) SVR_is_leaf(z$node), logical(1))
  depths <- vapply(nodes, function(z) z$node$depth, integer(1))
  list(n_nodes = length(nodes), n_leaves = sum(leaf), n_splits = sum(!leaf), max_depth = max(depths))
}

SVR_valid_split_table <- function(node, X, config) {
  if (!SVR_is_leaf(node) && is.null(node$indices)) return(data.frame())
  if (node$depth >= config$max_depth) return(data.frame())
  p <- ncol(X)
  rows <- list()
  for (j in seq_len(p)) {
    cuts <- config$split_grid[[j]]
    cuts <- cuts[cuts > node$lower[j] + config$split_tol & cuts < node$upper[j] - config$split_tol]
    if (length(cuts) == 0L) next
    # Count the same <= partitions in one pass per axis, preserving cut order.
    n_left <- findInterval(cuts, sort(X[node$indices, j]))
    n_right <- length(node$indices) - n_left
    keep <- n_left >= config$min_leaf_size & n_right >= config$min_leaf_size
    if (any(keep)) {
      rows[[length(rows) + 1L]] <- data.frame(
        split_dim = j, split_val = cuts[keep], n_left = n_left[keep], n_right = n_right[keep]
      )
    }
  }
  if (length(rows) == 0L) return(data.frame())
  tab <- do.call(rbind, rows)
  active_dims <- sort(unique(tab$split_dim))
  tab$log_split_prior <- NA_real_
  for (j in active_dims) {
    jj <- which(tab$split_dim == j)
    tab$log_split_prior[jj] <- -log(length(active_dims)) - log(length(jj))
  }
  tab
}

SVR_find_split_prior_log <- function(node, X, config, split_dim, split_val) {
  tab <- SVR_valid_split_table(node, X, config)
  if (nrow(tab) == 0L) return(-Inf)
  hit <- which(tab$split_dim == split_dim & abs(tab$split_val - split_val) <= config$split_tol)
  if (length(hit) == 0L) return(-Inf)
  tab$log_split_prior[hit[1L]]
}

SVR_recompute_node <- function(node, X, config, depth, lower, upper, indices) {
  node$depth <- as.integer(depth)
  node$lower <- as.numeric(lower)
  node$upper <- as.numeric(upper)
  node$indices <- as.integer(indices)
  if (length(indices) < config$min_leaf_size) return(NULL)
  if (SVR_is_leaf(node)) return(node)

  j <- node$split_dim
  cc <- node$split_val
  if (!is.finite(j) || !is.finite(cc) || j < 1L || j > ncol(X)) return(NULL)
  if (depth >= config$max_depth) return(NULL)
  if (!(cc > lower[j] + config$split_tol && cc < upper[j] - config$split_tol)) return(NULL)
  grid_hit <- any(abs(config$split_grid[[j]] - cc) <= config$split_tol)
  if (!grid_hit) return(NULL)

  left_mask <- X[indices, j] <= cc
  idx_l <- indices[left_mask]
  idx_r <- indices[!left_mask]
  if (length(idx_l) < config$min_leaf_size || length(idx_r) < config$min_leaf_size) return(NULL)

  lower_l <- lower
  upper_l <- upper
  upper_l[j] <- cc
  lower_r <- lower
  upper_r <- upper
  lower_r[j] <- cc
  node$left <- SVR_recompute_node(node$left, X, config, depth + 1L, lower_l, upper_l, idx_l)
  if (is.null(node$left)) return(NULL)
  node$right <- SVR_recompute_node(node$right, X, config, depth + 1L, lower_r, upper_r, idx_r)
  if (is.null(node$right)) return(NULL)
  node
}

SVR_recompute_tree <- function(tree, X, config) {
  p <- ncol(X)
  root <- SVR_recompute_node(tree$root, X, config, 0L, rep(0, p), rep(1, p), seq_len(nrow(X)))
  if (is.null(root)) return(NULL)
  list(root = root)
}

SVR_validate_tree <- function(tree, X, config) {
  ok <- !is.null(SVR_recompute_tree(tree, X, config))
  data.frame(valid = ok, tree_signature = if (ok) SVR_tree_signature(tree) else NA_character_)
}

SVR_grow_tree <- function(tree, path, split_dim, split_val, X, config) {
  out <- SVR_copy_tree(tree)
  leaf <- SVR_get_node(out$root, path)
  if (!SVR_is_leaf(leaf)) return(NULL)
  leaf$stopped <- FALSE
  leaf$split_dim <- as.integer(split_dim)
  leaf$split_val <- as.numeric(split_val)
  leaf$left <- SVR_make_leaf_node(leaf$depth + 1L, leaf$lower, leaf$upper, leaf$indices)
  leaf$right <- SVR_make_leaf_node(leaf$depth + 1L, leaf$lower, leaf$upper, leaf$indices)
  out$root <- SVR_set_node(out$root, path, leaf)
  SVR_recompute_tree(out, X, config)
}

SVR_split_node_direct <- function(node, split_dim, split_val, X, config) {
  out <- SVR_copy_tree(node)
  if (!SVR_is_leaf(out)) return(NULL)
  out$stopped <- FALSE
  out$split_dim <- as.integer(split_dim)
  out$split_val <- as.numeric(split_val)
  out$left <- SVR_make_leaf_node(out$depth + 1L, out$lower, out$upper, out$indices)
  out$right <- SVR_make_leaf_node(out$depth + 1L, out$lower, out$upper, out$indices)
  SVR_recompute_node(out, X, config, out$depth, out$lower, out$upper, out$indices)
}

SVR_stop_leaf <- function(tree, path) {
  out <- SVR_copy_tree(tree)
  leaf <- SVR_get_node(out$root, path)
  if (!SVR_is_leaf(leaf)) return(NULL)
  leaf$stopped <- TRUE
  out$root <- SVR_set_node(out$root, path, leaf)
  out
}

SVR_prune_tree <- function(tree, path, X, config) {
  out <- SVR_copy_tree(tree)
  node <- SVR_get_node(out$root, path)
  if (SVR_is_leaf(node)) return(NULL)
  node$split_dim <- NA_integer_
  node$split_val <- NA_real_
  node$left <- NULL
  node$right <- NULL
  node$stopped <- FALSE
  out$root <- SVR_set_node(out$root, path, node)
  SVR_recompute_tree(out, X, config)
}

SVR_change_tree <- function(tree, path, split_dim, split_val, X, config) {
  out <- SVR_copy_tree(tree)
  node <- SVR_get_node(out$root, path)
  if (SVR_is_leaf(node)) return(NULL)
  node$split_dim <- as.integer(split_dim)
  node$split_val <- as.numeric(split_val)
  out$root <- SVR_set_node(out$root, path, node)
  SVR_recompute_tree(out, X, config)
}

SVR_swap_tree <- function(tree, parent_path, child_side, X, config) {
  child_path <- c(parent_path, child_side)
  out <- SVR_copy_tree(tree)
  parent <- SVR_get_node(out$root, parent_path)
  child <- SVR_get_node(out$root, child_path)
  if (SVR_is_leaf(parent) || SVR_is_leaf(child)) return(NULL)
  p_dim <- parent$split_dim
  p_cut <- parent$split_val
  parent$split_dim <- child$split_dim
  parent$split_val <- child$split_val
  child$split_dim <- p_dim
  child$split_val <- p_cut
  parent <- if (child_side == "L") {
    parent$left <- child
    parent
  } else {
    parent$right <- child
    parent
  }
  out$root <- SVR_set_node(out$root, parent_path, parent)
  SVR_recompute_tree(out, X, config)
}

## Same-dimension parent-child swaps in tgp are binary-tree rotations. The
## promoted child keeps its split rule, the old parent moves down one level,
## and the terminal partition is preserved. Recompute all region metadata after
## rewiring so depth caps and finite-grid constraints remain part of the target.
SVR_rotate_tree <- function(tree, parent_path, child_side, X, config) {
  if (!child_side %in% c("L", "R")) return(NULL)
  out <- SVR_copy_tree(tree)
  parent <- SVR_get_node(out$root, parent_path)
  child <- if (child_side == "L") parent$left else parent$right
  if (SVR_is_leaf(parent) || is.null(child) || SVR_is_leaf(child)) return(NULL)
  if (parent$split_dim != child$split_dim) return(NULL)

  if (child_side == "L") {
    if (!(child$split_val < parent$split_val - config$split_tol)) return(NULL)
    parent$left <- child$right
    child$right <- parent
  } else {
    if (!(child$split_val > parent$split_val + config$split_tol)) return(NULL)
    parent$right <- child$left
    child$left <- parent
  }
  out$root <- SVR_set_node(out$root, parent_path, child)
  SVR_recompute_tree(out, X, config)
}

SVR_rotate_inverse_side <- function(child_side) {
  if (identical(child_side, "L")) "R" else if (identical(child_side, "R")) "L" else NA_character_
}

SVR_growable_paths <- function(tree, target, active_only = FALSE) {
  leaves <- SVR_collect_paths(tree$root, kind = "leaf")
  out <- list()
  for (z in leaves) {
    if (isTRUE(active_only) && isTRUE(z$node$stopped)) next
    tab <- SVR_valid_split_table(z$node, target$X, target$config)
    if (nrow(tab) > 0L) out[[length(out) + 1L]] <- z$path
  }
  out
}

SVR_prunable_paths <- function(tree) {
  internals <- SVR_collect_paths(tree$root, kind = "internal")
  out <- list()
  for (z in internals) {
    if (SVR_is_leaf(z$node$left) && SVR_is_leaf(z$node$right)) {
      out[[length(out) + 1L]] <- z$path
    }
  }
  out
}

SVR_leaf_logPhi_sum <- function(tree, target) {
  leaves <- SVR_leaf_nodes(tree)
  sum(SVR_logPhi_nodes(leaves, target))
}

SVR_tree_log_prior_node <- function(node, target) {
  cfg <- target$config
  if (SVR_is_leaf(node)) {
    tab <- SVR_valid_split_table(node, target$X, cfg)
    if (node$depth >= cfg$max_depth || nrow(tab) == 0L) return(0)
    rho <- cfg$alpha * (1 + node$depth)^(-cfg$beta)
    return(log1p(-rho))
  }
  tab <- SVR_valid_split_table(node, target$X, cfg)
  hit <- which(tab$split_dim == node$split_dim & abs(tab$split_val - node$split_val) <= cfg$split_tol)
  if (length(hit) == 0L) return(-Inf)
  rho <- cfg$alpha * (1 + node$depth)^(-cfg$beta)
  log(rho) + tab$log_split_prior[hit[1L]] +
    SVR_tree_log_prior_node(node$left, target) +
    SVR_tree_log_prior_node(node$right, target)
}

log_prior_tree <- function(tree, target) {
  SVR_tree_log_prior_node(tree$root, target)
}

logpost_tree <- function(tree, target) {
  lp <- log_prior_tree(tree, target)
  if (!is.finite(lp)) return(-Inf)
  lp + SVR_leaf_logPhi_sum(tree, target)
}

SVR_evaluate_smc_transition <- function(node, target) {
  cfg <- target$config
  tab <- SVR_valid_split_table(node, target$X, cfg)
  if (node$depth >= cfg$max_depth || nrow(tab) == 0L) {
    return(list(can_split = FALSE, log_w = 0, log_prob_split = -Inf, table = tab))
  }
  before <- target$n_eval
  rho <- cfg$alpha * (1 + node$depth)^(-cfg$beta)
  parent_phi <- SVR_logPhi_node(node, target)
  children <- vector("list", 2L * nrow(tab))
  for (i in seq_len(nrow(tab))) {
    left <- node
    right <- node
    j <- tab$split_dim[i]
    cc <- tab$split_val[i]
    mask <- target$X[node$indices, j] <= cc
    left$upper[j] <- cc
    left$indices <- node$indices[mask]
    left$depth <- node$depth + 1L
    left$left <- left$right <- NULL
    right$lower[j] <- cc
    right$indices <- node$indices[!mask]
    right$depth <- node$depth + 1L
    right$left <- right$right <- NULL
    children[[2L * i - 1L]] <- left
    children[[2L * i]] <- right
  }
  child_phi <- SVR_logPhi_nodes(children, target)
  log_bf <- child_phi[seq.int(1L, length(child_phi), by = 2L)] +
    child_phi[seq.int(2L, length(child_phi), by = 2L)] - parent_phi
  log_choice <- tab$log_split_prior + log_bf
  log_z_split <- SVR_logsumexp(log_choice)
  term_stop <- log1p(-rho)
  term_split <- log(rho) + log_z_split
  log_w <- SVR_logsumexp(c(term_stop, term_split))
  tab$log_BF <- log_bf
  tab$log_choice <- log_choice
  list(
    can_split = TRUE,
    log_w = log_w,
    log_prob_split = term_split - log_w,
    table = tab,
    n_new_logPhi = target$n_eval - before
  )
}

SVR_next_active_leaf_path <- function(tree, target) {
  paths <- SVR_growable_paths(tree, target, active_only = TRUE)
  if (length(paths) == 0L) return(NULL)
  depths <- vapply(paths, length, integer(1))
  paths[[order(depths, vapply(paths, SVR_path_string, character(1)))[1L]]]
}

SVR_apply_smc_transition <- function(tree, path, trans, target) {
  if (is.null(path) || !isTRUE(trans$can_split) || log(runif(1)) >= trans$log_prob_split) {
    return(list(tree = SVR_stop_leaf(tree, path), action = "stop",
                split_axis = NA_integer_, split_cut = NA_real_))
  }
  probs <- SVR_normalize_log_weights(trans$table$log_choice)
  k <- sample(seq_len(nrow(trans$table)), 1L, prob = probs)
  out <- SVR_grow_tree(tree, path, trans$table$split_dim[k], trans$table$split_val[k],
                       target$X, target$config)
  list(tree = out, action = "split", split_axis = trans$table$split_dim[k],
       split_cut = trans$table$split_val[k])
}

SVR_tgp_move_probs <- function() {
  c(grow = 0.20, prune = 0.20, change = 0.40, swap = 0.20)
}

SVR_validate_move_probs <- function(move_probs) {
  required <- c("grow", "prune", "change", "swap")
  if (is.null(names(move_probs)) || !all(required %in% names(move_probs))) {
    stop("move_probs must name grow, prune, change, and swap.")
  }
  move_probs <- as.numeric(move_probs[required])
  names(move_probs) <- required
  if (any(!is.finite(move_probs)) || any(move_probs < 0) ||
      abs(sum(move_probs) - 1) > 1e-12) {
    stop("RJMCMC move probabilities must be nonnegative and sum to one.")
  }
  move_probs
}

## The current tgp source ranks global candidate cutpoints by distance from
## the node midpoint, assigns weights 1/rank, and gives equal total mass to
## either side of the midpoint when both sides are available. This is a
## finite-grid implementation of that proposal; it is not the tree prior.
SVR_tgp_cut_table <- function(node, split_dim, target) {
  cfg <- target$config
  cuts <- sort(unique(as.numeric(cfg$split_grid[[split_dim]])))
  if (length(cuts) == 0L) return(data.frame())
  midpoint <- (node$lower[split_dim] + node$upper[split_dim]) / 2
  ord <- order((cuts - midpoint)^2, cuts)
  rank <- integer(length(cuts))
  rank[ord] <- seq_along(ord)
  raw <- 1 / rank
  inside <- cuts > node$lower[split_dim] + cfg$split_tol &
    cuts < node$upper[split_dim] - cfg$split_tol
  raw[!inside] <- 0
  left <- cuts < midpoint
  sum_left <- sum(raw[left])
  sum_right <- sum(raw[!left])
  prob <- numeric(length(cuts))
  if (sum_left > 0 && sum_right > 0) {
    prob[left] <- 0.5 * raw[left] / sum_left
    prob[!left] <- 0.5 * raw[!left] / sum_right
  } else if (sum(raw) > 0) {
    prob <- raw / sum(raw)
  }
  keep <- prob > 0
  data.frame(split_val = cuts[keep], probability = prob[keep])
}

SVR_log_tgp_grow_rule_prob <- function(node, split_dim, split_val, target) {
  p <- ncol(target$X)
  tab <- SVR_tgp_cut_table(node, split_dim, target)
  if (nrow(tab) == 0L) return(-Inf)
  hit <- which(abs(tab$split_val - split_val) <= target$config$split_tol)
  if (length(hit) == 0L) return(-Inf)
  -log(p) + log(tab$probability[hit[1L]])
}

SVR_sample_tgp_grow_rule <- function(node, target) {
  split_dim <- sample.int(ncol(target$X), 1L)
  tab <- SVR_tgp_cut_table(node, split_dim, target)
  if (nrow(tab) == 0L) return(NULL)
  k <- sample.int(nrow(tab), 1L, prob = tab$probability)
  list(
    split_dim = split_dim,
    split_val = tab$split_val[k],
    log_prob = -log(ncol(target$X)) + log(tab$probability[k])
  )
}

SVR_tgp_adjacent_cut <- function(node, direction, target) {
  cuts <- sort(unique(as.numeric(target$config$split_grid[[node$split_dim]])))
  tol <- target$config$split_tol
  candidates <- if (direction == "lower") {
    cuts[cuts < node$split_val - tol]
  } else {
    cuts[cuts > node$split_val + tol]
  }
  if (length(candidates) == 0L) return(NA_real_)
  if (direction == "lower") max(candidates) else min(candidates)
}

SVR_tgp_swap_paths <- function(tree) {
  paths <- SVR_internal_paths(tree)
  paths[vapply(paths, length, integer(1)) > 0L]
}

SVR_move_info <- function(tree, target, move_probs = SVR_tgp_move_probs()) {
  move_probs <- SVR_validate_move_probs(move_probs)
  grow <- SVR_leaf_paths(tree)
  prune <- SVR_prunable_paths(tree)
  change <- SVR_internal_paths(tree)
  swap <- SVR_tgp_swap_paths(tree)
  list(
    counts = c(grow = length(grow), prune = length(prune),
               change = length(change), swap = length(swap)),
    move_probs = move_probs,
    grow_paths = grow,
    prune_paths = prune,
    change_paths = change,
    swap_paths = swap
  )
}

SVR_invalid_rjmcmc_proposal <- function(move, detail, kernel = move) {
  list(
    tree = NULL,
    move = move,
    kernel = kernel,
    valid = FALSE,
    log_q_forward = NA_real_,
    log_q_reverse = NA_real_,
    detail = detail
  )
}

SVR_propose_rjmcmc <- function(tree, target,
                               move_probs = SVR_tgp_move_probs()) {
  info <- SVR_move_info(tree, target, move_probs)
  move <- sample(names(info$move_probs), 1L, prob = info$move_probs)

  if (move == "grow") {
    if (info$counts[["grow"]] == 0L) {
      return(SVR_invalid_rjmcmc_proposal(move, "no_leaf"))
    }
    path <- info$grow_paths[[sample.int(info$counts[["grow"]], 1L)]]
    node <- SVR_get_node(tree$root, path)
    if (node$depth >= target$config$max_depth) {
      return(SVR_invalid_rjmcmc_proposal(move, "selected_leaf_at_max_depth"))
    }
    rule <- SVR_sample_tgp_grow_rule(node, target)
    if (is.null(rule)) {
      return(SVR_invalid_rjmcmc_proposal(move, "selected_dimension_has_no_cut"))
    }
    prop_tree <- SVR_grow_tree(
      tree, path, rule$split_dim, rule$split_val, target$X, target$config
    )
    if (is.null(prop_tree)) {
      return(SVR_invalid_rjmcmc_proposal(move, "split_violates_tree_constraints"))
    }
    rev_info <- SVR_move_info(prop_tree, target, move_probs)
    log_q_f <- log(info$move_probs[["grow"]]) - log(info$counts[["grow"]]) +
      rule$log_prob
    log_q_r <- log(rev_info$move_probs[["prune"]]) -
      log(rev_info$counts[["prune"]])
    return(list(
      tree = prop_tree, move = move, kernel = "grow", valid = TRUE,
      log_q_forward = log_q_f, log_q_reverse = log_q_r,
      detail = paste(SVR_path_string(path), rule$split_dim,
                     format(rule$split_val, digits = 12), sep = ":")
    ))
  }

  if (move == "prune") {
    if (info$counts[["prune"]] == 0L) {
      return(SVR_invalid_rjmcmc_proposal(move, "no_prunable_node"))
    }
    path <- info$prune_paths[[sample.int(info$counts[["prune"]], 1L)]]
    node <- SVR_get_node(tree$root, path)
    old_dim <- node$split_dim
    old_cut <- node$split_val
    prop_tree <- SVR_prune_tree(tree, path, target$X, target$config)
    if (is.null(prop_tree)) {
      return(SVR_invalid_rjmcmc_proposal(move, "prune_failed"))
    }
    rev_info <- SVR_move_info(prop_tree, target, move_probs)
    prop_node <- SVR_get_node(prop_tree$root, path)
    log_rule <- SVR_log_tgp_grow_rule_prob(
      prop_node, old_dim, old_cut, target
    )
    log_q_f <- log(info$move_probs[["prune"]]) -
      log(info$counts[["prune"]])
    log_q_r <- log(rev_info$move_probs[["grow"]]) -
      log(rev_info$counts[["grow"]]) + log_rule
    return(list(
      tree = prop_tree, move = move, kernel = "prune",
      valid = is.finite(log_q_r),
      log_q_forward = log_q_f, log_q_reverse = log_q_r,
      detail = SVR_path_string(path)
    ))
  }

  if (move == "change") {
    if (info$counts[["change"]] == 0L) {
      return(SVR_invalid_rjmcmc_proposal(move, "no_internal_node"))
    }
    path <- info$change_paths[[sample.int(info$counts[["change"]], 1L)]]
    node <- SVR_get_node(tree$root, path)
    direction <- sample(c("lower", "upper"), 1L)
    new_cut <- SVR_tgp_adjacent_cut(node, direction, target)
    if (!is.finite(new_cut)) {
      return(SVR_invalid_rjmcmc_proposal(move, "no_adjacent_cut"))
    }
    prop_tree <- SVR_change_tree(
      tree, path, node$split_dim, new_cut, target$X, target$config
    )
    if (is.null(prop_tree)) {
      return(SVR_invalid_rjmcmc_proposal(move, "adjacent_change_invalidates_tree"))
    }
    rev_info <- SVR_move_info(prop_tree, target, move_probs)
    log_q_f <- log(info$move_probs[["change"]]) -
      log(info$counts[["change"]]) - log(2)
    log_q_r <- log(rev_info$move_probs[["change"]]) -
      log(rev_info$counts[["change"]]) - log(2)
    return(list(
      tree = prop_tree, move = move, kernel = "change", valid = TRUE,
      log_q_forward = log_q_f, log_q_reverse = log_q_r,
      detail = paste(SVR_path_string(path), direction,
                     format(new_cut, digits = 12), sep = ":")
    ))
  }

  if (info$counts[["swap"]] == 0L) {
    return(SVR_invalid_rjmcmc_proposal(move, "no_parent_child_internal_pair"))
  }
  child_path <- info$swap_paths[[sample.int(info$counts[["swap"]], 1L)]]
  parent_path <- head(child_path, -1L)
  child_side <- tail(child_path, 1L)
  parent <- SVR_get_node(tree$root, parent_path)
  child <- SVR_get_node(tree$root, child_path)
  if (parent$split_dim == child$split_dim) {
    direction <- if (child_side == "L") "right" else "left"
    prop_tree <- SVR_rotate_tree(
      tree, parent_path, child_side, target$X, target$config
    )
    if (is.null(prop_tree)) {
      return(SVR_invalid_rjmcmc_proposal(
        move, "rotation_invalidates_tree", kernel = "rotate"
      ))
    }
    rev_info <- SVR_move_info(prop_tree, target, move_probs)
    log_q_f <- log(info$move_probs[["swap"]]) -
      log(info$counts[["swap"]])
    log_q_r <- log(rev_info$move_probs[["swap"]]) -
      log(rev_info$counts[["swap"]])
    return(list(
      tree = prop_tree, move = move, kernel = "rotate", valid = TRUE,
      log_q_forward = log_q_f, log_q_reverse = log_q_r,
      detail = paste(
        paste0("rotate_", direction), SVR_path_string(parent_path),
        child_side, sep = ":"
      )
    ))
  }
  prop_tree <- SVR_swap_tree(
    tree, parent_path, child_side, target$X, target$config
  )
  if (is.null(prop_tree)) {
    return(SVR_invalid_rjmcmc_proposal(move, "swap_invalidates_tree"))
  }
  rev_info <- SVR_move_info(prop_tree, target, move_probs)
  log_q_f <- log(info$move_probs[["swap"]]) - log(info$counts[["swap"]])
  log_q_r <- log(rev_info$move_probs[["swap"]]) -
    log(rev_info$counts[["swap"]])
  list(
    tree = prop_tree, move = move, kernel = "swap", valid = TRUE,
    log_q_forward = log_q_f, log_q_reverse = log_q_r,
    detail = paste(SVR_path_string(parent_path), child_side, sep = ":")
  )
}

SVR_random_prior_tree_node <- function(node, target) {
  tab <- SVR_valid_split_table(node, target$X, target$config)
  if (node$depth >= target$config$max_depth || nrow(tab) == 0L) return(node)
  rho <- target$config$alpha * (1 + node$depth)^(-target$config$beta)
  if (runif(1) > rho) return(node)
  probs <- exp(tab$log_split_prior)
  k <- sample(seq_len(nrow(tab)), 1L, prob = probs)
  out <- SVR_split_node_direct(node, tab$split_dim[k], tab$split_val[k],
                               target$X, target$config)
  if (is.null(out)) stop("Exact prior draw produced an invalid split.")
  out$left <- SVR_random_prior_tree_node(out$left, target)
  out$right <- SVR_random_prior_tree_node(out$right, target)
  rec <- SVR_recompute_node(out, target$X, target$config,
                            out$depth, out$lower, out$upper, out$indices)
  if (is.null(rec)) stop("Exact prior draw produced an invalid tree.")
  rec
}

SVR_random_prior_tree <- function(target, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  root <- SVR_random_prior_tree_node(SVR_make_root_tree(target$X)$root, target)
  SVR_recompute_tree(list(root = root), target$X, target$config)
}

SVR_enumerate_nodes <- function(node, target, max_trees = Inf, counter_env = NULL) {
  if (is.null(counter_env)) {
    counter_env <- new.env(parent = emptyenv())
    counter_env$n <- 0L
  }
  out <- list(node)
  counter_env$n <- counter_env$n + 1L
  if (counter_env$n >= max_trees) return(out)
  tab <- SVR_valid_split_table(node, target$X, target$config)
  if (node$depth >= target$config$max_depth || nrow(tab) == 0L) return(out)
  for (i in seq_len(nrow(tab))) {
    if (counter_env$n >= max_trees) break
    split_node <- SVR_split_node_direct(node, tab$split_dim[i], tab$split_val[i],
                                        target$X, target$config)
    if (is.null(split_node)) next
    lefts <- SVR_enumerate_nodes(split_node$left, target, max_trees, counter_env)
    rights <- SVR_enumerate_nodes(split_node$right, target, max_trees, counter_env)
    for (ll in lefts) {
      for (rr in rights) {
        nd <- split_node
        nd$left <- ll
        nd$right <- rr
        rec <- SVR_recompute_node(nd, target$X, target$config,
                                  nd$depth, nd$lower, nd$upper, nd$indices)
        if (!is.null(rec)) out[[length(out) + 1L]] <- rec
        if (length(out) >= max_trees) break
      }
      if (length(out) >= max_trees) break
    }
  }
  out
}

SVR_qc_target_equality <- function(target, n_trees = 100L, seed = 1L) {
  set.seed(seed)
  rows <- vector("list", n_trees)
  for (i in seq_len(n_trees)) {
    tr <- SVR_random_prior_tree(target)
    a <- logpost_tree(tr, target)
    b <- log_prior_tree(tr, target) + SVR_leaf_logPhi_sum(tr, target)
    rows[[i]] <- data.frame(
      tree_id = i,
      logpost_tree = a,
      decomposed_logpost = b,
      abs_diff = abs(a - b),
      passed = abs(a - b) <= 1e-8
    )
  }
  do.call(rbind, rows)
}

SVR_leaf_region_signatures <- function(tree) {
  sort(vapply(
    SVR_leaf_nodes(tree),
    function(node) SVR_region_signature(node$lower, node$upper),
    character(1)
  ))
}

SVR_make_rotate_test_case <- function(target, seed = 1L) {
  set.seed(seed)
  root <- SVR_make_root_tree(target$X)
  root_tab <- SVR_valid_split_table(root$root, target$X, target$config)
  if (nrow(root_tab) == 0L) return(NULL)
  for (ii in sample(seq_len(nrow(root_tab)))) {
    tr <- SVR_grow_tree(
      root, character(0), root_tab$split_dim[ii], root_tab$split_val[ii],
      target$X, target$config
    )
    if (is.null(tr)) next
    for (side in sample(c("L", "R"))) {
      child <- SVR_get_node(tr$root, side)
      child_tab <- SVR_valid_split_table(child, target$X, target$config)
      same <- which(child_tab$split_dim == tr$root$split_dim)
      if (length(same) == 0L) next
      for (jj in same[sample.int(length(same))]) {
        out <- SVR_grow_tree(
          tr, side, child_tab$split_dim[jj], child_tab$split_val[jj],
          target$X, target$config
        )
        if (!is.null(out)) {
          return(list(
            tree = out, parent_path = character(0), child_side = side
          ))
        }
      }
    }
  }
  NULL
}

SVR_qc_detailed_balance_rotate <- function(
    target, n_pairs = 10L, seed = 1L,
    move_probs = SVR_tgp_move_probs()) {
  move_probs <- SVR_validate_move_probs(move_probs)
  rows <- list()
  attempts <- 0L
  while (length(rows) < n_pairs && attempts < 100L * n_pairs) {
    attempts <- attempts + 1L
    test <- SVR_make_rotate_test_case(target, seed + attempts - 1L)
    if (is.null(test)) next
    tr <- test$tree
    prop <- SVR_rotate_tree(
      tr, test$parent_path, test$child_side, target$X, target$config
    )
    if (is.null(prop)) next
    inverse_side <- SVR_rotate_inverse_side(test$child_side)
    restored <- SVR_rotate_tree(
      prop, test$parent_path, inverse_side, target$X, target$config
    )
    if (is.null(restored)) next

    info <- SVR_move_info(tr, target, move_probs)
    rev_info <- SVR_move_info(prop, target, move_probs)
    log_q_f <- log(info$move_probs[["swap"]]) -
      log(info$counts[["swap"]])
    log_q_r <- log(rev_info$move_probs[["swap"]]) -
      log(rev_info$counts[["swap"]])
    lp <- logpost_tree(tr, target)
    lp_prop <- logpost_tree(prop, target)
    log_alpha_f <- min(0, lp_prop - lp + log_q_r - log_q_f)
    log_alpha_r <- min(0, lp - lp_prop + log_q_f - log_q_r)
    log_lhs <- lp + log_q_f + log_alpha_f
    log_rhs <- lp_prop + log_q_r + log_alpha_r
    scale <- max(log_lhs, log_rhs)
    lhs_scaled <- exp(log_lhs - scale)
    rhs_scaled <- exp(log_rhs - scale)
    rel <- abs(lhs_scaled - rhs_scaled) /
      max(lhs_scaled, rhs_scaled, 1e-300)
    leaf_before <- SVR_leaf_region_signatures(tr)
    leaf_after <- SVR_leaf_region_signatures(prop)
    leaf_phi_diff <- abs(
      SVR_leaf_logPhi_sum(tr, target) - SVR_leaf_logPhi_sum(prop, target)
    )
    roundtrip_equal <- identical(
      SVR_tree_signature(tr), SVR_tree_signature(restored)
    )
    regions_equal <- identical(leaf_before, leaf_after)
    rows[[length(rows) + 1L]] <- data.frame(
      pair_id = length(rows) + 1L,
      variant = "tgp_style_rotate",
      move_type = "rotate",
      direction = if (test$child_side == "L") "right" else "left",
      forward_swapable_count = info$counts[["swap"]],
      reverse_swapable_count = rev_info$counts[["swap"]],
      roundtrip_signature_equal = roundtrip_equal,
      leaf_regions_equal = regions_equal,
      leaf_logPhi_abs_diff = leaf_phi_diff,
      log_lhs = log_lhs,
      log_rhs = log_rhs,
      relative_difference = rel,
      passed = roundtrip_equal && regions_equal &&
        leaf_phi_diff <= 1e-8 && is.finite(rel) && rel <= 1e-6,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) {
    return(data.frame(
      pair_id = integer(0), variant = character(0),
      move_type = character(0), direction = character(0),
      forward_swapable_count = integer(0),
      reverse_swapable_count = integer(0),
      roundtrip_signature_equal = logical(0),
      leaf_regions_equal = logical(0), leaf_logPhi_abs_diff = numeric(0),
      log_lhs = numeric(0), log_rhs = numeric(0),
      relative_difference = numeric(0), passed = logical(0)
    ))
  }
  do.call(rbind, rows)
}

SVR_qc_proposal_probability <- function(target, n_trees = 10L, seed = 1L,
                                        move_probs = SVR_tgp_move_probs()) {
  move_probs <- SVR_validate_move_probs(move_probs)
  set.seed(seed)
  rows <- list()
  for (i in seq_len(n_trees)) {
    tr <- SVR_random_prior_tree(target)
    info <- SVR_move_info(tr, target, move_probs)
    move_sum <- sum(info$move_probs)
    rotate_pairs <- sum(vapply(info$swap_paths, function(path) {
      parent <- SVR_get_node(tr$root, head(path, -1L))
      child <- SVR_get_node(tr$root, path)
      parent$split_dim == child$split_dim
    }, logical(1)))
    cut_distribution_max_abs_diff <- NA_real_
    if (info$counts[["grow"]] > 0L) {
      diffs <- unlist(lapply(info$grow_paths, function(path) {
        node <- SVR_get_node(tr$root, path)
        vapply(seq_len(ncol(target$X)), function(j) {
          tab <- SVR_tgp_cut_table(node, j, target)
          if (nrow(tab) == 0L) return(NA_real_)
          abs(sum(tab$probability) - 1)
        }, numeric(1))
      }))
      diffs <- diffs[is.finite(diffs)]
      if (length(diffs)) cut_distribution_max_abs_diff <- max(diffs)
    }
    rows[[i]] <- data.frame(
      tree_id = i,
      variant = "tgp_style_rotate",
      n_grow = info$counts[["grow"]],
      n_prune = info$counts[["prune"]],
      n_change = info$counts[["change"]],
      n_swap = info$counts[["swap"]],
      n_rotate_pairs_within_swap = rotate_pairs,
      move_probability_sum = move_sum,
      cut_distribution_max_abs_diff = cut_distribution_max_abs_diff,
      passed = abs(move_sum - 1) <= 1e-8 &&
        (is.na(cut_distribution_max_abs_diff) ||
           cut_distribution_max_abs_diff <= 1e-8)
    )
  }
  do.call(rbind, rows)
}

SVR_qc_detailed_balance_grow_prune <- function(target, n_pairs = 10L,
                                               seed = 1L,
                                               move_probs = SVR_tgp_move_probs()) {
  move_probs <- SVR_validate_move_probs(move_probs)
  set.seed(seed)
  rows <- list()
  attempts <- 0L
  while (length(rows) < n_pairs && attempts < 100L * n_pairs) {
    attempts <- attempts + 1L
    tr <- SVR_random_prior_tree(target)
    info <- SVR_move_info(tr, target, move_probs)
    if (info$counts[["grow"]] == 0L) next
    path <- info$grow_paths[[sample(seq_along(info$grow_paths), 1L)]]
    node <- SVR_get_node(tr$root, path)
    if (node$depth >= target$config$max_depth) next
    rule <- SVR_sample_tgp_grow_rule(node, target)
    if (is.null(rule)) next
    prop <- SVR_grow_tree(tr, path, rule$split_dim, rule$split_val, target$X, target$config)
    if (is.null(prop)) next
    rev_info <- SVR_move_info(prop, target, move_probs)
    if (rev_info$counts[["prune"]] == 0L) next
    lp <- logpost_tree(tr, target)
    lp_prop <- logpost_tree(prop, target)
    log_q_f <- log(info$move_probs[["grow"]]) -
      log(info$counts[["grow"]]) + rule$log_prob
    log_q_r <- log(rev_info$move_probs[["prune"]]) -
      log(rev_info$counts[["prune"]])
    log_alpha_f <- min(0, lp_prop - lp + log_q_r - log_q_f)
    log_alpha_r <- min(0, lp - lp_prop + log_q_f - log_q_r)
    log_lhs <- lp + log_q_f + log_alpha_f
    log_rhs <- lp_prop + log_q_r + log_alpha_r
    rel <- abs(exp(log_lhs - max(log_lhs, log_rhs)) - exp(log_rhs - max(log_lhs, log_rhs))) /
      max(exp(log_lhs - max(log_lhs, log_rhs)), exp(log_rhs - max(log_lhs, log_rhs)), 1e-300)
    rows[[length(rows) + 1L]] <- data.frame(
      pair_id = length(rows) + 1L,
      variant = "tgp_style_rotate",
      move_type = "grow_prune",
      log_lhs = log_lhs,
      log_rhs = log_rhs,
      relative_difference = rel,
      passed = is.finite(rel) && rel <= 1e-6
    )
  }
  if (length(rows) == 0L) {
    return(data.frame(pair_id = integer(0), variant = character(0), move_type = character(0),
                      log_lhs = numeric(0), log_rhs = numeric(0),
                      relative_difference = numeric(0), passed = logical(0)))
  }
  do.call(rbind, rows)
}
