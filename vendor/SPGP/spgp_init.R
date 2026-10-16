# SPGP faithful initialization (same as in main.R)
# Source after R/SPGP_func_iso.R and R/utils.R (getGridIdx, genGrid, dentrigraph, proposeMST, getEdgeStatus, randomProject).

spgp_faithful_init <- function(coords,
                               Y,
                               n_knot_total = 600L,
                               k0 = 5L,
                               L = 3L,
                               nngp_neighbors_pilot = 15L,
                               nu_pilot = 2.5,
                               threshold_graph = 0.2,
                               seed_proj = 1L) {
  if (!is.matrix(coords)) coords <- as.matrix(coords)
  stopifnot(ncol(coords) == 2)

  x_cutoff <- seq(0, 1, length.out = 7)
  y_cutoff <- seq(0, 1, length.out = 7)
  n_cell <- (length(x_cutoff) - 1) * (length(y_cutoff) - 1)
  cell_id <- getGridIdx(coords, x_cutoff, y_cutoff)

  bottomleft_cell <- expand.grid(y_cutoff[-length(y_cutoff)], x_cutoff[-length(x_cutoff)])
  bottomleft_cell[, 1:2] <- bottomleft_cell[, c(2, 1)]
  colnames(bottomleft_cell) <- c("lon", "lat")
  topright_cell <- expand.grid(y_cutoff[-1], x_cutoff[-1])
  topright_cell[, 1:2] <- topright_cell[, c(2, 1)]
  colnames(topright_cell) <- c("lon", "lat")

  theta_cell <- rep(NA_real_, n_cell)
  for (i in seq_len(n_cell)) {
    idx_cell <- which(cell_id == i)
    if (length(idx_cell) == 0) next
    X_cell <- matrix(1, nrow = length(idx_cell), ncol = 1)
    gp_res <- tryCatch({
      GpGp::fit_model(Y[idx_cell], coords[idx_cell, ], X_cell,
        covfun_name = "matern25_isotropic", m_seq = nngp_neighbors_pilot, silent = TRUE)
    }, error = function(e1) {
      tryCatch({
        GpGp::fit_model(y = Y[idx_cell], x = coords[idx_cell, ], X = X_cell,
          cov_model = "matern", nu = nu_pilot,
          nn = min(nngp_neighbors_pilot, length(idx_cell) - 1L), verbose = FALSE)
      }, error = function(e2) NULL)
    })
    if (is.null(gp_res)) next
    sig2 <- NA_real_; phi <- NA_real_
    if (!is.null(gp_res$covparms) && length(gp_res$covparms) >= 2) {
      sig2 <- as.numeric(gp_res$covparms[1]); phi <- as.numeric(gp_res$covparms[2])
    } else {
      pars <- gp_res$theta; if (is.null(pars)) pars <- gp_res$param; if (is.null(pars)) pars <- gp_res
      if (!is.null(pars$sigmasq)) sig2 <- as.numeric(pars$sigmasq)
      if (!is.null(pars$sigma2)) sig2 <- as.numeric(pars$sigma2)
      if (!is.null(pars$phi)) phi <- as.numeric(pars$phi)
      if (!is.null(pars$range)) phi <- as.numeric(pars$range)
    }
    if (is.finite(sig2) && is.finite(phi) && sig2 > 0 && phi > 0)
      theta_cell[i] <- log(sig2) - 5 * log(phi)
  }

  if (all(!is.finite(theta_cell))) theta_cell[] <- 1.0
  else { theta_med <- median(theta_cell[is.finite(theta_cell)], na.rm = TRUE); theta_cell[!is.finite(theta_cell)] <- theta_med }
  theta_alloc <- theta_cell
  if (!is.finite(sum(theta_alloc)) || sum(theta_alloc) <= 0 || any(theta_alloc <= 0))
    theta_alloc <- theta_alloc - min(theta_alloc) + 1e-6
  n_knot_cells <- n_knot_total * theta_alloc / sum(theta_alloc)

  coords_knot <- matrix(0, nrow = 0, ncol = 2)
  cell_id_knot <- integer(0)
  for (i in seq_len(n_cell)) {
    n_side <- max(1L, round(sqrt(n_knot_cells[i])))
    xmin <- bottomleft_cell[i, 1]; ymin <- bottomleft_cell[i, 2]
    xmax <- topright_cell[i, 1]; ymax <- topright_cell[i, 2]
    kk <- genGrid(n_side, n_side, xmin, ymin, xmax, ymax)
    coords_knot <- rbind(coords_knot, kk)
    cell_id_knot <- c(cell_id_knot, rep(i, n_side^2))
  }
  colnames(coords_knot) <- c("lon", "lat")
  n_knot <- nrow(coords_knot)

  graph0 <- dentrigraph(coords_knot, threshold = threshold_graph)
  E(graph0)$eid <- seq_len(ecount(graph0))
  V(graph0)$vid <- seq_len(vcount(graph0))
  fl_res <- genlasso::fusedlasso2d(matrix(theta_cell, nrow = length(y_cutoff) - 1, ncol = length(x_cutoff) - 1))
  df_vec <- fl_res$df
  col_idx <- which(df_vec == k0)
  if (length(col_idx) == 0) col_idx <- which.min(abs(df_vec - k0)) else col_idx <- col_idx[1]
  theta_cell_hat <- fl_res$beta[, col_idx]
  cluster_cell <- as.integer(as.factor(theta_cell_hat))
  cluster <- cluster_cell[cell_id_knot]
  edge_status <- getEdgeStatus(cluster, graph0)
  mstgraph0 <- proposeMST(graph0, edge_status)
  if ("weight" %in% igraph::edge_attr_names(graph0)) graph0 <- igraph::delete_edge_attr(graph0, "weight")
  if ("weight" %in% igraph::edge_attr_names(mstgraph0)) mstgraph0 <- igraph::delete_edge_attr(mstgraph0, "weight")
  if (ecount(mstgraph0) != n_knot - 1) stop("MST graph not connected")
  if (sum(edge_status[E(mstgraph0)$eid] == "b") >= k0) stop("Non-contiguous partition")

  nn_res <- FNN::get.knnx(coords_knot, coords, k = L)
  nearest_knots <- nn_res$nn.index
  proj_prob <- 1 / nn_res$nn.dist
  for (i in seq_len(nrow(proj_prob))) {
    if (is.infinite(proj_prob[i, 1])) proj_prob[i, ] <- c(1, rep(0, L - 1))
    else proj_prob[i, ] <- proj_prob[i, ] / sum(proj_prob[i, ])
  }
  set.seed(seed_proj)
  proj <- randomProject(nearest_knots, proj_prob, L)

  tunning <- c(); tunning["csize_min"] <- 20; tunning["min_n_nngp"] <- 250
  hyperpar <- c()
  hyperpar["c"] <- 0.5; hyperpar["a_phi"] <- 1; hyperpar["b_phi"] <- 1
  hyperpar["a_tau"] <- 2; hyperpar["b_tau"] <- 2; hyperpar["a_sigma"] <- 2; hyperpar["b_sigma"] <- 0.1
  hyperpar["a_lambda"] <- 2; hyperpar["b_lambda"] <- 2; hyperpar["nn"] <- L; hyperpar["nngp"] <- 15
  init_val <- list()
  init_val[["k"]] <- k0; init_val[["coords_knot"]] <- coords_knot; init_val[["graph0"]] <- graph0
  init_val[["tree"]] <- mstgraph0; init_val[["cluster"]] <- cluster
  init_val[["nearest_knots"]] <- nearest_knots; init_val[["proj"]] <- proj
  init_val[["tau"]] <- rep(1, k0); init_val[["phi"]] <- rep(1, k0); init_val[["sigmasq"]] <- rep(1, k0)
  init_val[["beta"]] <- 0; init_val[["lambda"]] <- 1

  list(coords_knot = coords_knot, graph0 = graph0, init_val = init_val, hyperpar = hyperpar, tunning = tunning)
}
