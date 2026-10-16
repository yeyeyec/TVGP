// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]
// Worker failures are reported on the R thread, never through Armadillo streams.
#define ARMA_WARN_LEVEL 0
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>

using namespace arma;
using namespace Rcpp;

namespace {
int tvgp_threads = 1;
template <typename F>
struct RowWorker : public RcppParallel::Worker {
    F& calculate;
    std::vector<std::string>& errors;
    RowWorker(F& calculate_, std::vector<std::string>& errors_)
        : calculate(calculate_), errors(errors_) {}
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            try { calculate(static_cast<int>(i)); }
            catch (const std::exception& e) { errors[i] = e.what(); }
            catch (...) { errors[i] = "unknown_row_evaluation_failure"; }
        }
    }
};
}

// [[Rcpp::export]]
int TVGP_set_compute_threads(int threads = 1) {
    if (threads < 1 || threads > 16) stop("threads must be between 1 and 16.");
    int previous = tvgp_threads;
    tvgp_threads = threads;
    return previous;
}

// -----------------------------------------------------------------------------
// 1. Matérn Kernel (1D)
// -----------------------------------------------------------------------------
struct Kernel1D {
    double val;
    double d_log_phi;
    double d2_log_phi;
};

Kernel1D matern_1d(double d, double phi, double nu) {
    Kernel1D res;
    if (d < 1e-12) {
        res.val = 1.0;
        res.d_log_phi = 0.0;
        res.d2_log_phi = 0.0;
        return res;
    }
    double r = d / phi;
    double exp_r = exp(-r);

    if (std::abs(nu - 0.5) < 1e-8) {
        res.val = exp_r;
        res.d_log_phi = r * exp_r;
        res.d2_log_phi = (r * r - r) * exp_r;
    } else if (std::abs(nu - 1.5) < 1e-8) {
        res.val = (1.0 + r) * exp_r;
        res.d_log_phi = r * r * exp_r;
        res.d2_log_phi = r * r * (r - 2.0) * exp_r;
    } else if (std::abs(nu - 2.5) < 1e-8) {
        double r2 = r * r;
        res.val = (1.0 + r + r2 / 3.0) * exp_r;
        res.d_log_phi = r2 * (1.0 + r) * exp_r / 3.0;
        res.d2_log_phi = r2 * (r2 - 2.0 * r - 2.0) * exp_r / 3.0;
    } else {
        stop("Analytic derivatives are implemented for Matern nu in {0.5, 1.5, 2.5}.");
    }
    return res;
}

void separable_matern(const rowvec& x1, const rowvec& x2,
                      const vec& phi, double nu,
                      bool calc_grad, bool calc_hessian,
                      double& value, vec& gradient, mat& hessian) {
    int d_dim = phi.n_elem;
    vec component(d_dim);
    vec component_grad(d_dim, fill::zeros);
    vec component_hessian(d_dim, fill::zeros);

    value = 1.0;
    for(int d = 0; d < d_dim; d++) {
        Kernel1D kd = matern_1d(std::abs(x1(d) - x2(d)), phi(d), nu);
        component(d) = kd.val;
        component_grad(d) = kd.d_log_phi;
        component_hessian(d) = kd.d2_log_phi;
        value *= kd.val;
    }

    if(!calc_grad) return;
    gradient.zeros(d_dim);
    for(int d = 0; d < d_dim; d++) {
        double other_product = 1.0;
        for(int j = 0; j < d_dim; j++) if(j != d) other_product *= component(j);
        gradient(d) = component_grad(d) * other_product;
    }

    if(!calc_hessian) return;
    hessian.zeros(d_dim, d_dim);
    for(int d = 0; d < d_dim; d++) {
        double other_product = 1.0;
        for(int j = 0; j < d_dim; j++) if(j != d) other_product *= component(j);
        hessian(d, d) = component_hessian(d) * other_product;
        for(int ell = 0; ell < d; ell++) {
            double remaining_product = 1.0;
            for(int j = 0; j < d_dim; j++) {
                if(j != d && j != ell) remaining_product *= component(j);
            }
            double cross = component_grad(d) * component_grad(ell) * remaining_product;
            hessian(d, ell) = cross;
            hessian(ell, d) = cross;
        }
    }
}

// -----------------------------------------------------------------------------
// 2. Helpers
// -----------------------------------------------------------------------------
bool robust_inv(const mat& A, mat& A_inv) {
    // A silent jitter changes the objective without changing its derivatives.
    // Treat a non-SPD conditioning matrix as a failed covariance evaluation.
    return inv_sympd(A_inv, A);
}

uvec find_nn_past(int i, int m, const mat& X) {
    if (i == 0) return uvec();
    int k = std::min(i, m);
    std::vector<std::pair<double, unsigned int> > dists;
    dists.reserve(i);
    for(int j = 0; j < i; j++) {
        dists.push_back(std::make_pair(sum(square(X.row(i) - X.row(j))), (unsigned int)j));
    }
    if(k < i) {
        std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    std::sort(dists.begin(), dists.begin() + k,
              [](const auto& a, const auto& b) { return a.first < b.first; });

    uvec out(k);
    for(int j = 0; j < k; j++) out(j) = dists[j].second;
    return out;
}

// [[Rcpp::export]]
uvec get_maximin_order(mat X) {
    int n = X.n_rows;
    uvec order(n);
    vec min_dists(n);
    min_dists.fill(datum::inf);
    order(0) = 0;
    vec used = zeros(n); used(0) = 1;
    for(int i = 1; i < n; i++) min_dists(i) = sum(square(X.row(i) - X.row(0)));
    for(int k = 1; k < n; k++) {
        uword best_idx = 0;
        double max_min_dist = -1.0;
        for(int i = 0; i < n; i++) {
            if(used(i) == 0 && min_dists(i) > max_min_dist) {
                max_min_dist = min_dists(i); best_idx = i;
            }
        }
        order(k) = best_idx; used(best_idx) = 1;
        for(int i = 0; i < n; i++) {
            if(used(i) == 0) {
                double d = sum(square(X.row(i) - X.row(best_idx)));
                if(d < min_dists(i)) min_dists(i) = d;
            }
        }
    }
    return order;
}

// [[Rcpp::export]]
umat get_nn_matrix(mat X, int m) {
    if (m < 0 || !X.is_finite()) stop("Invalid nearest-neighbor inputs.");
    int n = X.n_rows;
    umat nn_mat(n, m);
    nn_mat.fill(0); 
    std::vector<std::string> errors(n);
    auto calculate = [&](int i) {
        uvec nn = find_nn_past(i, m, X);
        for(unsigned int j=0; j < nn.n_elem; j++) nn_mat(i, j) = nn(j);
    };
    RowWorker<decltype(calculate)> worker(calculate, errors);
    if (tvgp_threads == 1 || n < 32) worker(0, n);
    else RcppParallel::parallelFor(0, n, worker, 16, tvgp_threads);
    for (int i = 0; i < n; ++i) {
        if (!errors[i].empty()) stop(errors[i]);
    }
    return nn_mat;
}

// [[Rcpp::export]]
List build_local_nn_from_global_pool(const mat& X,
                                     const uvec& local_order,
                                     const umat& global_nn_pool_orig,
                                     const uvec& global_nn_count,
                                     const uvec& order_rank,
                                     int m,
                                     bool exact_fallback = false) {
    int n = local_order.n_elem;
    int N = X.n_rows;
    umat nn_mat(n, m);
    nn_mat.fill(0);
    uvec nn_count(n);
    nn_count.fill(0);

    std::vector<int> local_pos(N, -1);
    for(int i = 0; i < n; i++) local_pos[local_order(i)] = i;

    std::vector<std::string> errors(n);
    auto calculate = [&](int i) {
        int needed = std::min(i, m);
        if(needed == 0) return;

        std::vector<unsigned int> chosen;
        chosen.reserve(needed);
        unsigned int gpos = order_rank(local_order(i));
        unsigned int pool_n = global_nn_count(gpos);

        for(unsigned int pp = 0; pp < pool_n && (int)chosen.size() < needed; pp++) {
            unsigned int orig = global_nn_pool_orig(gpos, pp);
            if(orig >= (unsigned int)N) continue;
            int lp = local_pos[orig];
            if(lp < 0 || lp >= i) continue;

            bool seen = false;
            for(unsigned int old : chosen) {
                if(old == (unsigned int)lp) { seen = true; break; }
            }
            if(!seen) chosen.push_back((unsigned int)lp);
        }

        if(exact_fallback && (int)chosen.size() < needed) {
            std::vector<std::pair<double, unsigned int> > dists;
            dists.reserve(i);
            for(int j = 0; j < i; j++) {
                bool seen = false;
                for(unsigned int old : chosen) {
                    if(old == (unsigned int)j) { seen = true; break; }
                }
                if(seen) continue;
                double dist = sum(square(X.row(local_order(i)) - X.row(local_order(j))));
                dists.push_back(std::make_pair(dist, (unsigned int)j));
            }

            int add_n = std::min((int)dists.size(), needed - (int)chosen.size());
            if(add_n > 0) {
                if(add_n < (int)dists.size()) {
                    std::nth_element(dists.begin(), dists.begin() + add_n, dists.end(),
                                     [](const auto& a, const auto& b) { return a.first < b.first; });
                }
                std::sort(dists.begin(), dists.begin() + add_n,
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                for(int a = 0; a < add_n; a++) chosen.push_back(dists[a].second);
            }
        }

        nn_count(i) = chosen.size();
        for(unsigned int j = 0; j < chosen.size(); j++) nn_mat(i, j) = chosen[j];
    };
    RowWorker<decltype(calculate)> worker(calculate, errors);
    if (tvgp_threads == 1 || n < 32) worker(0, n);
    else RcppParallel::parallelFor(0, n, worker, 16, tvgp_threads);
    for (int i = 0; i < n; ++i) {
        if (!errors[i].empty()) stop(errors[i]);
    }

    return List::create(Named("nn_mat") = nn_mat, Named("nn_count") = nn_count);
}

// -----------------------------------------------------------------------------
// 3. Proper-NIG integrated likelihood, gradient, and Hessian
// -----------------------------------------------------------------------------
struct CovPriorEval {
    double nll;
    vec grad;
    mat hessian;
};

CovPriorEval covariance_prior_nll_grad_impl(const vec& theta,
                                             int cov_prior_code,
                                             const vec& prior_params,
                                             bool calc_grad,
                                             bool calc_hessian) {
    int D_dim = theta.n_elem - 1;
    if(D_dim < 1) stop("theta must contain at least one log range and one log nugget ratio.");

    vec phi(D_dim);
    for(int j = 0; j < D_dim; j++) phi(j) = exp(theta(j));
    double g = exp(theta(D_dim));
    CovPriorEval out;
    out.nll = 0.0;
    out.grad = zeros(D_dim + 1);
    out.hessian = zeros(D_dim + 1, D_dim + 1);

    if(cov_prior_code == 0) {
        if(prior_params.n_elem != 4) {
            stop("The inverse-gamma covariance prior requires four parameters.");
        }
        double a_phi = prior_params(0);
        double b_phi = prior_params(1);
        double a_g = prior_params(2);
        double b_g = prior_params(3);
        if(a_phi <= 0.0 || b_phi <= 0.0 || a_g <= 0.0 || b_g <= 0.0) {
            stop("Inverse-gamma covariance-prior parameters must be positive.");
        }

        // The density is transformed from (phi, g) to (log(phi), log(g)).
        out.nll = a_g * log(g) + b_g / g;
        if(calc_grad) out.grad(D_dim) = a_g - b_g / g;
        if(calc_hessian) out.hessian(D_dim, D_dim) = b_g / g;
        for(int j = 0; j < D_dim; j++) {
            out.nll += a_phi * log(phi(j)) + b_phi / phi(j);
            if(calc_grad) out.grad(j) = a_phi - b_phi / phi(j);
            if(calc_hessian) out.hessian(j, j) = b_phi / phi(j);
        }
        return out;
    }

    if(cov_prior_code == 1) {
        if(prior_params.n_elem != (uword)(D_dim + 2)) {
            stop("The jointly robust covariance prior requires a, b, and one C_j per input.");
        }
        double a = prior_params(0);
        double b = prior_params(1);
        vec C = prior_params.subvec(2, D_dim + 1);
        if(a <= -(D_dim + 1.0) || b <= 0.0 || any(C <= 0.0)) {
            stop("Invalid jointly robust covariance-prior parameters.");
        }

        // Gu (2019), Eq. (3.2), with beta_j=1/phi_j and eta=g:
        // (sum C_j beta_j + eta)^a exp[-b(sum C_j beta_j + eta)].
        // The final terms are the Jacobian for
        // (beta, eta) -> (log(phi), log(g)).
        vec weighted_beta = C / phi;
        double t = sum(weighted_beta);
        double u = t + g;
        if(!std::isfinite(u) || u <= 0.0) stop("Invalid jointly robust prior scale.");
        out.nll = -a * log(u) + b * u + sum(log(phi)) - log(g);
        if(calc_grad) {
            double c = a / u - b;
            for(int j = 0; j < D_dim; j++) {
                out.grad(j) = 1.0 + c * weighted_beta(j);
            }
            out.grad(D_dim) = (b - a / u) * g - 1.0;
        }
        if(calc_hessian) {
            vec du(D_dim + 1, fill::zeros);
            mat ddu(D_dim + 1, D_dim + 1, fill::zeros);
            for(int j = 0; j < D_dim; j++) {
                du(j) = -weighted_beta(j);
                ddu(j, j) = weighted_beta(j);
            }
            du(D_dim) = g;
            ddu(D_dim, D_dim) = g;
            out.hessian = (a / (u * u)) * (du * du.t()) +
                (b - a / u) * ddu;
        }
        return out;
    }

    if(cov_prior_code == 2) {
        if(prior_params.n_elem != (uword)(D_dim + 2)) {
            stop("The legacy covariance prior requires a, b, and one C_j per input.");
        }
        double a = prior_params(0);
        double b = prior_params(1);
        vec C = prior_params.subvec(2, D_dim + 1);
        if(a <= -D_dim || b <= 0.0 || any(C <= 0.0)) {
            stop("Invalid legacy covariance-prior parameters.");
        }
        vec weighted_beta = C / phi;
        double t = sum(weighted_beta);
        if(!std::isfinite(t) || t <= 0.0) stop("Invalid legacy prior scale.");
        out.nll = -a * log(t) + b * (t + g) + sum(log(phi)) - log(g);
        if(calc_grad) {
            double c = a / t - b;
            for(int j = 0; j < D_dim; j++) {
                out.grad(j) = 1.0 + c * weighted_beta(j);
            }
            out.grad(D_dim) = b * g - 1.0;
        }
        if(calc_hessian) {
            vec dt(D_dim, fill::zeros);
            mat ddt(D_dim, D_dim, fill::zeros);
            for(int j = 0; j < D_dim; j++) {
                dt(j) = -weighted_beta(j);
                ddt(j, j) = weighted_beta(j);
            }
            out.hessian.submat(0, 0, D_dim - 1, D_dim - 1) =
                (a / (t * t)) * (dt * dt.t()) + (b - a / t) * ddt;
            out.hessian(D_dim, D_dim) = b * g;
        }
        return out;
    }

    stop("cov_prior_code must be 0 (inverse_gamma), 1 (Gu 2019 jointly robust), or 2 (legacy).");
    return out;
}

List vecchia_nll_grad_impl(const vec& theta, const vec& y, const mat& X,
                           const umat& nn_mat, const uvec& nn_count,
                           double nu, int m, int cov_prior_code,
                           vec prior_params, vec leaf_prior_params,
                           bool calc_grad, bool calc_hessian) {
    
    int n = y.n_elem;
    int D_dim = X.n_cols;
    if (!(std::abs(nu - 0.5) < 1e-8 || std::abs(nu - 1.5) < 1e-8 ||
          std::abs(nu - 2.5) < 1e-8)) {
        stop("Analytic derivatives require Matern nu in {0.5, 1.5, 2.5}.");
    }
    if (theta.n_elem != static_cast<unsigned>(D_dim + 1) ||
        X.n_rows != y.n_elem || nn_mat.n_rows != y.n_elem ||
        nn_count.n_elem != y.n_elem || m < 0 ||
        !theta.is_finite() || !X.is_finite() || !y.is_finite()) {
        stop("Invalid Vecchia likelihood dimensions or nonfinite inputs.");
    }
    for (int i = 0; i < n; ++i) {
        arma::uword k = std::min(static_cast<arma::uword>(m), std::min(nn_count(i),
            static_cast<arma::uword>(i)));
        if (k > nn_mat.n_cols) stop("Neighbor count exceeds matrix width.");
        for (unsigned j = 0; j < k; ++j) {
            if (nn_mat(i, j) >= static_cast<unsigned>(i))
                stop("Training neighbors must precede the current observation.");
        }
    }
    if(leaf_prior_params.n_elem != 4) {
        stop("The proper NIG leaf prior requires (m0, V0, a0, b0).");
    }
    double m0 = leaf_prior_params(0);
    double V0 = leaf_prior_params(1);
    double a0 = leaf_prior_params(2);
    double b0 = leaf_prior_params(3);
    if(!std::isfinite(m0) || !std::isfinite(V0) ||
       !std::isfinite(a0) || !std::isfinite(b0) ||
       V0 <= 0.0 || a0 <= 0.0 || b0 <= 0.0) {
        stop("Invalid proper NIG leaf-prior parameters.");
    }
    double P0 = 1.0 / V0;
    double a_n = a0 + 0.5 * (double)n;
    
    vec phi(D_dim); for(int j=0; j<D_dim; j++) phi(j) = exp(theta(j));
    double g = exp(theta(D_dim));
    
    if(calc_hessian) calc_grad = true;
    CovPriorEval prior = covariance_prior_nll_grad_impl(
        theta, cov_prior_code, prior_params, calc_grad, calc_hessian
    );
    double neg_log_prior = prior.nll;
    vec grad_out = prior.grad;
    mat hessian_out = prior.hessian;

    int q = D_dim + 1;
    auto failure_result = [&](const std::string& reason) {
        mat failed_hessian(q, q);
        failed_hessian.fill(datum::nan);
        return List::create(
            Named("nll") = 1e100,
            Named("grad") = zeros(q),
            Named("hessian") = failed_hessian,
            Named("valid") = false,
            Named("failure_reason") = reason,
            Named("gamma_mean") = m0,
            Named("gamma_var_factor") = V0,
            Named("sigma2_shape") = a_n,
            Named("sigma2_scale") = b0,
            Named("sigma2_mean") = (a_n > 1.0) ? b0 / (a_n - 1.0) : R_PosInf,
            Named("sigma2_hat") = (a_n > 1.0) ? b0 / (a_n - 1.0) : R_PosInf,
            Named("S0") = NA_REAL
        );
    };

    vec By(n, fill::zeros);
    vec B1(n, fill::zeros);
    vec D_vec(n, fill::zeros);
    mat dBy, dB1, dD;
    cube ddBy, ddB1, ddD;
    if(calc_grad) {
        dBy.zeros(n, q);
        dB1.zeros(n, q);
        dD.zeros(n, q);
    }
    if(calc_hessian) {
        ddBy.zeros(n, q, q);
        ddB1.zeros(n, q, q);
        ddD.zeros(n, q, q);
    }

    // Rows write disjoint entries; all reductions retain the serial summation order.
    std::vector<std::string> row_errors(n);
    auto calculate_row = [&](int i) {
        int k = std::min((int)nn_count(i), std::min(i, m));
        double R_ii = 1.0 + g;

        if (k == 0) {
            D_vec(i) = R_ii; By(i) = y(i); B1(i) = 1.0;
            if(calc_grad) dD(i, D_dim) = g;
            if(calc_hessian) ddD(i, D_dim, D_dim) = g;
        } else {
            uvec nn(k); for(int j=0; j<k; j++) nn(j) = nn_mat(i, j);
            mat R_NN(k, k, fill::zeros);
            vec r_Ni(k, fill::zeros);
            Cube<double> dR_NN;
            mat dr_Ni;
            std::vector<mat> ddR_NN;
            cube ddr_Ni;
            if(calc_grad) {
                dR_NN.zeros(k, k, q);
                dr_Ni.zeros(k, q);
            }
            if(calc_hessian) {
                ddR_NN.reserve(q * q);
                for(int idx = 0; idx < q * q; idx++) {
                    ddR_NN.push_back(mat(k, k, fill::zeros));
                }
                ddr_Ni.zeros(k, q, q);
            }

            for(int r_idx=0; r_idx<k; r_idx++) {
                int r_global = nn(r_idx);
                double val_prod = 0.0;
                vec pair_grad(D_dim, fill::zeros);
                mat pair_hessian(D_dim, D_dim, fill::zeros);
                separable_matern(
                    X.row(i), X.row(r_global), phi, nu,
                    calc_grad, calc_hessian, val_prod, pair_grad, pair_hessian
                );
                r_Ni(r_idx) = val_prod;
                if(calc_grad) {
                    for(int d=0; d<D_dim; d++) dr_Ni(r_idx, d) = pair_grad(d);
                }
                if(calc_hessian) {
                    for(int p=0; p<D_dim; p++) {
                        for(int ell=0; ell<D_dim; ell++) {
                            ddr_Ni(r_idx, p, ell) = pair_hessian(p, ell);
                        }
                    }
                }
                R_NN(r_idx, r_idx) = 1.0 + g;
                if(calc_grad) dR_NN(r_idx, r_idx, D_dim) = g;
                if(calc_hessian) ddR_NN[D_dim * q + D_dim](r_idx, r_idx) = g;
                for(int c_idx=0; c_idx<r_idx; c_idx++) {
                    int c_global = nn(c_idx);
                    double val_sub = 0.0;
                    vec pair_grad_sub(D_dim, fill::zeros);
                    mat pair_hessian_sub(D_dim, D_dim, fill::zeros);
                    separable_matern(
                        X.row(r_global), X.row(c_global), phi, nu,
                        calc_grad, calc_hessian,
                        val_sub, pair_grad_sub, pair_hessian_sub
                    );
                    R_NN(r_idx, c_idx) = R_NN(c_idx, r_idx) = val_sub;
                    if(calc_grad) {
                        for(int d=0; d<D_dim; d++) {
                            dR_NN(r_idx, c_idx, d) =
                                dR_NN(c_idx, r_idx, d) = pair_grad_sub(d);
                        }
                    }
                    if(calc_hessian) {
                        for(int p=0; p<D_dim; p++) {
                            for(int ell=0; ell<D_dim; ell++) {
                                double value = pair_hessian_sub(p, ell);
                                ddR_NN[p * q + ell](r_idx, c_idx) = value;
                                ddR_NN[p * q + ell](c_idx, r_idx) = value;
                            }
                        }
                    }
                }
            }
            mat U;
            if (!chol(U, R_NN)) {
                row_errors[i] = "conditioning_covariance_not_positive_definite";
                return;
            }
            auto solve_R = [&](const vec& rhs) -> vec {
                vec tmp = solve(trimatl(U.t()), rhs, solve_opts::fast);
                return solve(trimatu(U), tmp, solve_opts::fast);
            };
            vec whitened_cross = solve(trimatl(U.t()), r_Ni, solve_opts::fast);
            vec b_i = -solve(trimatu(U), whitened_cross, solve_opts::fast);
            long double schur = R_ii;
            for (int j = 0; j < k; ++j) {
                schur -= static_cast<long double>(whitened_cross(j)) * whitened_cross(j);
            }
            D_vec(i) = static_cast<double>(schur);
            if(!std::isfinite(D_vec(i)) || D_vec(i) <= 1e-14) {
                row_errors[i] = "nonpositive_vecchia_conditional_variance";
                return;
            }
            By(i) = y(i) + dot(b_i, y.elem(nn));
            B1(i) = 1.0 + dot(b_i, ones(k));
            if(calc_grad) {
                mat db(k, q, fill::zeros);
                for(int p=0; p<=D_dim; p++) {
                    db.col(p) = -solve_R(dR_NN.slice(p) * b_i + dr_Ni.col(p));
                    double dR_ii_p = (p == D_dim) ? g : 0.0;
                    dD(i, p) = dR_ii_p + 2.0 * dot(dr_Ni.col(p), b_i) +
                        dot(b_i, dR_NN.slice(p) * b_i);
                    dBy(i, p) = dot(db.col(p), y.elem(nn));
                    dB1(i, p) = dot(db.col(p), ones(k));
                }

                if(calc_hessian) {
                    cube ddb(k, q, q, fill::zeros);
                    for(int p=0; p<q; p++) {
                        for(int ell=0; ell<q; ell++) {
                            vec rhs = ddR_NN[p * q + ell] * b_i +
                                dR_NN.slice(p) * db.col(ell) +
                                dR_NN.slice(ell) * db.col(p) +
                                ddr_Ni.slice(ell).col(p);
                            ddb.slice(ell).col(p) = -solve_R(rhs);

                            double d2R_ii = (p == D_dim && ell == D_dim) ? g : 0.0;
                            ddD(i, p, ell) = d2R_ii +
                                2.0 * dot(ddr_Ni.slice(ell).col(p), b_i) +
                                dot(b_i, ddR_NN[p * q + ell] * b_i) +
                                dot(db.col(ell), dR_NN.slice(p) * b_i + dr_Ni.col(p)) +
                                dot(db.col(p), dR_NN.slice(ell) * b_i + dr_Ni.col(ell));
                            ddBy(i, p, ell) = dot(ddb.slice(ell).col(p), y.elem(nn));
                            ddB1(i, p, ell) = dot(ddb.slice(ell).col(p), ones(k));
                        }
                    }
                }
            }
        }
    };
    RowWorker<decltype(calculate_row)> worker(calculate_row, row_errors);
    if (tvgp_threads == 1 || n < 32) worker(0, n);
    else RcppParallel::parallelFor(0, n, worker, 16, tvgp_threads);
    for (int i = 0; i < n; ++i) {
        if (!row_errors[i].empty()) return failure_result(row_errors[i]);
    }

    vec D_inv = 1.0 / D_vec;
    double HTQH = dot(B1, D_inv % B1);
    double yTQy = dot(By, D_inv % By);
    double yTQH = dot(By, D_inv % B1);
    double M = P0 + HTQH;
    double h = P0 * m0 + yTQH;
    double gamma_mean = h / M;
    vec Be = By - gamma_mean * B1;
    double S0 = dot(Be, D_inv % Be) + P0 * std::pow(gamma_mean - m0, 2);
    if(!std::isfinite(S0) || S0 < 0.0 || !std::isfinite(M) || M <= 0.0) {
        return failure_result("invalid_proper_nig_quadratic_form");
    }

    double b_n = b0 + 0.5 * S0;
    double gamma_var_factor = 1.0 / M;
    double sigma2_mean = (a_n > 1.0) ? b_n / (a_n - 1.0) : R_PosInf;
    double neg_loglik = 0.5 * sum(log(D_vec)) + 0.5 * log(M) +
                        a_n * log(b_n) + neg_log_prior;

    if(calc_grad) {
        vec d_HTQH(q, fill::zeros);
        mat dd_HTQH(q, q, fill::zeros);
        mat dBe = dBy - gamma_mean * dB1;
        cube ddBe;
        if (calc_hessian) ddBe = ddBy - gamma_mean * ddB1;

        auto quad_first = [&](const vec& u, const vec& v,
                              const mat& du, const mat& dv, int p) {
            vec dE = -dD.col(p) % square(D_inv);
            return dot(du.col(p), D_inv % v) +
                dot(u, D_inv % dv.col(p)) + dot(u, dE % v);
        };
        auto quad_second = [&](const vec& u, const vec& v,
                               const mat& du, const mat& dv,
                               const cube& ddu, const cube& ddv,
                               int p, int ell) {
            vec dE_p = -dD.col(p) % square(D_inv);
            vec dE_ell = -dD.col(ell) % square(D_inv);
            vec ddE = 2.0 * dD.col(p) % dD.col(ell) % pow(D_inv, 3) -
                ddD.slice(ell).col(p) % square(D_inv);
            return
                dot(ddu.slice(ell).col(p), D_inv % v) +
                dot(du.col(p), D_inv % dv.col(ell)) +
                dot(du.col(ell), D_inv % dv.col(p)) +
                dot(u, D_inv % ddv.slice(ell).col(p)) +
                dot(du.col(p), dE_ell % v) +
                dot(u, dE_ell % dv.col(p)) +
                dot(du.col(ell), dE_p % v) +
                dot(u, dE_p % dv.col(ell)) +
                dot(u, ddE % v);
        };

        for(int p=0; p<q; p++) {
            d_HTQH(p) = quad_first(B1, B1, dB1, dB1, p);
        }
        if(calc_hessian) {
            for(int p=0; p<q; p++) {
                for(int ell=0; ell<q; ell++) {
                    dd_HTQH(p, ell) = quad_second(
                        B1, B1, dB1, dB1, ddB1, ddB1, p, ell
                    );
                }
            }
        }

        vec dS0(q, fill::zeros);
        mat ddS0(q, q, fill::zeros);
        vec z(q, fill::zeros);
        for(int p=0; p<q; p++) {
            // Differentiate with the mean fixed, then apply the Schur-complement correction.
            dS0(p) = quad_first(Be, Be, dBe, dBe, p);
            z(p) = quad_first(B1, Be, dB1, dBe, p);
            grad_out(p) += 0.5 * sum(D_inv % dD.col(p)) +
                0.5 * d_HTQH(p) / M + 0.5 * a_n * dS0(p) / b_n;
        }

        if(calc_hessian) {
            for(int p=0; p<q; p++) {
                for(int ell=0; ell<q; ell++) {
                    ddS0(p, ell) = quad_second(
                        Be, Be, dBe, dBe, ddBe, ddBe, p, ell
                    ) - 2.0 * z(p) * z(ell) / M;

                    double log_D_hessian = sum(
                        D_inv % ddD.slice(ell).col(p) -
                        square(D_inv) % dD.col(p) % dD.col(ell)
                    );
                    double log_M_hessian = dd_HTQH(p, ell) / M -
                        d_HTQH(p) * d_HTQH(ell) / (M * M);
                    double log_b_hessian = 0.5 * ddS0(p, ell) / b_n -
                        0.25 * dS0(p) * dS0(ell) / (b_n * b_n);
                    hessian_out(p, ell) += 0.5 * log_D_hessian +
                        0.5 * log_M_hessian + a_n * log_b_hessian;
                }
            }
            hessian_out = 0.5 * (hessian_out + hessian_out.t());
        }
    }

    return List::create(
        Named("nll") = neg_loglik, 
        Named("grad") = grad_out, 
        Named("hessian") = hessian_out,
        Named("valid") = true,
        Named("failure_reason") = NA_STRING,
        Named("gamma_mean") = gamma_mean,
        Named("gamma_var_factor") = gamma_var_factor,
        Named("sigma2_shape") = a_n,
        Named("sigma2_scale") = b_n,
        Named("sigma2_mean") = sigma2_mean,
        Named("sigma2_hat") = sigma2_mean,
        Named("S0") = S0
    );
}

// [[Rcpp::export]]
List covariance_prior_nll_grad(vec theta, int cov_prior_code, vec prior_params) {
    CovPriorEval out = covariance_prior_nll_grad_impl(
        theta, cov_prior_code, prior_params, true, false
    );
    return List::create(Named("nll") = out.nll, Named("grad") = out.grad);
}

// [[Rcpp::export]]
List covariance_prior_nll_grad_hessian(vec theta, int cov_prior_code,
                                       vec prior_params) {
    CovPriorEval out = covariance_prior_nll_grad_impl(
        theta, cov_prior_code, prior_params, true, true
    );
    return List::create(
        Named("nll") = out.nll,
        Named("grad") = out.grad,
        Named("hessian") = out.hessian
    );
}

// [[Rcpp::export]]
List vecchia_nll_grad(const vec& theta, const vec& y, const mat& X,
                      const umat& nn_mat, double nu, int m,
                      int cov_prior_code, vec prior_params,
                      vec leaf_prior_params, bool calc_grad) {
    int n = y.n_elem;
    uvec nn_count(n);
    for(int i = 0; i < n; i++) nn_count(i) = std::min(i, m);
    return vecchia_nll_grad_impl(
        theta, y, X, nn_mat, nn_count, nu, m,
        cov_prior_code, prior_params, leaf_prior_params, calc_grad, false
    );
}

// [[Rcpp::export]]
List vecchia_nll_grad_counts(const vec& theta, const vec& y, const mat& X,
                             const umat& nn_mat, const uvec& nn_count,
                             double nu, int m, int cov_prior_code,
                             vec prior_params, vec leaf_prior_params,
                             bool calc_grad) {
    return vecchia_nll_grad_impl(
        theta, y, X, nn_mat, nn_count, nu, m,
        cov_prior_code, prior_params, leaf_prior_params, calc_grad, false
    );
}

// [[Rcpp::export]]
List vecchia_nll_grad_hessian_counts(const vec& theta, const vec& y, const mat& X,
                                     const umat& nn_mat, const uvec& nn_count,
                                     double nu, int m,
                                     int cov_prior_code, vec prior_params,
                                     vec leaf_prior_params) {
    return vecchia_nll_grad_impl(
        theta, y, X, nn_mat, nn_count, nu, m,
        cov_prior_code, prior_params, leaf_prior_params, true, true
    );
}

// -----------------------------------------------------------------------------
// 4. Proper-NIG posterior predictive Student-t parameters and moments
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
List vecchia_prediction_moments(vec y_train, mat X_train, mat X_test, vec theta,
                                double gamma_mean, double gamma_var_factor,
                                double sigma2_shape, double sigma2_scale,
                                double nu, int m,
                                bool predict_obs) {
    int n = X_train.n_rows;
    int np = X_test.n_rows;
    int D_dim = X_train.n_cols;
    vec phi(D_dim); for(int j=0; j<D_dim; j++) phi(j) = exp(theta(j));
    double g = exp(theta(D_dim));
    
    // The R layer supplies the order used when the leaf was fitted.
    vec y_train_ord = y_train;
    mat X_train_ord = X_train;

    if(!std::isfinite(gamma_mean) || !std::isfinite(gamma_var_factor) ||
       !std::isfinite(sigma2_shape) || !std::isfinite(sigma2_scale) ||
       gamma_var_factor <= 0.0 || sigma2_shape <= 1.0 || sigma2_scale <= 0.0) {
        stop("Invalid proper-NIG posterior parameters supplied to prediction.");
    }
    double sigma2_mean = sigma2_scale / (sigma2_shape - 1.0);
    double predictive_df = 2.0 * sigma2_shape;
    vec residuals = y_train_ord - gamma_mean;

    vec mu_pred(np);
    vec var_pred(np);
    vec scale2_pred(np);
    vec df_pred(np);
    for(int i = 0; i < np; i++) {
        vec dists(n);
        for(int j=0; j<n; j++) dists(j) = sum(square(X_test.row(i) - X_train_ord.row(j))); 
        uvec sorted_idx = sort_index(dists, "ascend");
        int k = std::min(m, n);
        uvec nn = sorted_idx.head(k);
        mat R_NN(k,k); vec r_Ni(k);
        for(int r=0; r<k; r++) {
            double val_prod = 1.0;
            for(int d=0; d<D_dim; d++) val_prod *= matern_1d(std::abs(X_test(i,d) - X_train_ord(nn(r),d)), phi(d), nu).val;
            r_Ni(r) = val_prod;
            R_NN(r,r) = 1.0 + g;
            for(int c=0; c<r; c++) {
                double val_sub = 1.0;
                for(int d=0; d<D_dim; d++) val_sub *= matern_1d(std::abs(X_train_ord(nn(r),d) - X_train_ord(nn(c),d)), phi(d), nu).val;
                R_NN(r,c) = R_NN(c,r) = val_sub;
            }
        }
        mat U;
        if(!chol(U, R_NN)) stop("Prediction covariance solve failed.");
        vec whitened_cross = solve(trimatl(U.t()), r_Ni, solve_opts::fast);
        vec b_i = -solve(trimatu(U), whitened_cross, solve_opts::fast);
        double mean_coefficient = 1.0 + sum(b_i);
        mu_pred(i) = gamma_mean - dot(b_i, residuals.elem(nn));
        
        double R_pp = (predict_obs) ? (1.0 + g) : 1.0;
        long double conditional_variance = R_pp;
        for (int j = 0; j < k; ++j) {
            conditional_variance -= static_cast<long double>(whitened_cross(j)) *
                                    whitened_cross(j);
        }
        double unscaled_var = static_cast<double>(conditional_variance) +
                              mean_coefficient * mean_coefficient * gamma_var_factor;
        if(!std::isfinite(unscaled_var) || unscaled_var < -1e-10) {
            stop("Invalid posterior predictive variance factor.");
        }
        unscaled_var = std::max(unscaled_var, 0.0);
        scale2_pred(i) = (sigma2_scale / sigma2_shape) * unscaled_var;
        var_pred(i) = sigma2_mean * unscaled_var;
        df_pred(i) = predictive_df;
    }
    return List::create(
        Named("mu") = mu_pred,
        Named("variance") = var_pred,
        Named("scale2") = scale2_pred,
        Named("df") = df_pred,
        Named("gamma_mean") = gamma_mean,
        Named("prediction_target") = predict_obs ? "observation" : "latent_function"
    );
}

// [[Rcpp::export]]
List vecchia_prediction_joint(vec y_train, mat X_train, mat X_test, vec theta,
                              double gamma_mean, double gamma_var_factor,
                              double sigma2_shape, double sigma2_scale,
                              double nu, int m, bool predict_obs,
                              bool return_covariance = true) {
    int n = X_train.n_rows;
    int np = X_test.n_rows;
    int d_dim = X_train.n_cols;
    if((int)y_train.n_elem != n || X_test.n_cols != (uword)d_dim) {
        stop("Prediction inputs have incompatible dimensions.");
    }
    if((int)theta.n_elem != d_dim + 1 || m < 1) {
        stop("Invalid covariance parameters or neighbor count for joint prediction.");
    }
    if(!std::isfinite(gamma_mean) || !std::isfinite(gamma_var_factor) ||
       !std::isfinite(sigma2_shape) || !std::isfinite(sigma2_scale) ||
       gamma_var_factor <= 0.0 || sigma2_shape <= 1.0 || sigma2_scale <= 0.0) {
        stop("Invalid proper-NIG posterior parameters supplied to joint prediction.");
    }

    vec phi(d_dim);
    for(int j = 0; j < d_dim; j++) phi(j) = exp(theta(j));
    double g = exp(theta(d_dim));
    double test_diagonal = predict_obs ? 1.0 + g : 1.0;

    auto kernel_value = [&](const rowvec& x1, const rowvec& x2) {
        double out = 1.0;
        for(int d = 0; d < d_dim; d++) {
            out *= matern_1d(std::abs(x1(d) - x2(d)), phi(d), nu).val;
        }
        return out;
    };
    auto predecessor_row = [&](unsigned int code) {
        return code < (unsigned int)n ? X_train.row(code) : X_test.row(code - n);
    };
    auto predecessor_covariance = [&](unsigned int first, unsigned int second) {
        if(first == second) {
            return first < (unsigned int)n ? 1.0 + g : test_diagonal;
        }
        return kernel_value(predecessor_row(first), predecessor_row(second));
    };

    vec mu(np, fill::zeros);
    vec mean_loading(np, fill::zeros);
    vec D_star(np, fill::zeros);
    int max_test_parents = std::min(m, std::max(0, np - 1));
    imat test_parent_index(np, max_test_parents);
    test_parent_index.fill(-1);
    mat test_parent_coefficient(np, max_test_parents, fill::zeros);
    uvec test_parent_count(np, fill::zeros);

    for(int i = 0; i < np; i++) {
        int n_predecessors = n + i;
        int k = std::min(m, n_predecessors);
        if(k < 1) stop("Joint prediction requires at least one predecessor.");

        std::vector<std::pair<double, unsigned int> > distances;
        distances.reserve(n_predecessors);
        for(int j = 0; j < n; j++) {
            distances.push_back(std::make_pair(
                sum(square(X_test.row(i) - X_train.row(j))),
                (unsigned int)j
            ));
        }
        for(int j = 0; j < i; j++) {
            distances.push_back(std::make_pair(
                sum(square(X_test.row(i) - X_test.row(j))),
                (unsigned int)(n + j)
            ));
        }
        auto distance_order = [](const std::pair<double, unsigned int>& a,
                                 const std::pair<double, unsigned int>& b) {
            if(a.first < b.first) return true;
            if(a.first > b.first) return false;
            return a.second < b.second;
        };
        if(k < n_predecessors) {
            std::nth_element(
                distances.begin(), distances.begin() + k, distances.end(),
                distance_order
            );
        }
        std::sort(distances.begin(), distances.begin() + k, distance_order);

        std::vector<unsigned int> parent(k);
        for(int r = 0; r < k; r++) parent[r] = distances[r].second;

        mat R_NN(k, k, fill::zeros);
        vec r_star(k, fill::zeros);
        for(int r = 0; r < k; r++) {
            r_star(r) = kernel_value(
                X_test.row(i), predecessor_row(parent[r])
            );
            R_NN(r, r) = predecessor_covariance(parent[r], parent[r]);
            for(int c = 0; c < r; c++) {
                double value = predecessor_covariance(parent[r], parent[c]);
                R_NN(r, c) = value;
                R_NN(c, r) = value;
            }
        }

        // The last Cholesky pivot is the conditional standard deviation.
        // An explicit inverse loses accuracy when latent parents are close.
        mat augmented(k + 1, k + 1, fill::zeros);
        augmented.submat(0, 0, k - 1, k - 1) = R_NN;
        augmented.submat(0, k, k - 1, k) = r_star;
        augmented.submat(k, 0, k, k - 1) = r_star.t();
        augmented(k, k) = test_diagonal;
        mat L;
        if(!chol(L, augmented, "lower")) {
            stop("Joint covariance Cholesky failed at ordered test row %d (parents=%d).",
                 i + 1, k);
        }
        vec transformed_cross = L.submat(k, 0, k, k - 1).t();
        mat parent_factor = L.submat(0, 0, k - 1, k - 1);
        vec coefficient;
        if(!solve(coefficient, trimatu(parent_factor.t()), transformed_cross,
                  solve_opts::fast) || !coefficient.is_finite()) {
            stop("Joint triangular solve failed at ordered test row %d.", i + 1);
        }
        double conditional_variance = L(k, k) * L(k, k);
        if(!std::isfinite(conditional_variance) || conditional_variance <= 0.0) {
            stop("Invalid Cholesky conditional variance at ordered test row %d.", i + 1);
        }
        D_star(i) = conditional_variance;

        double current_mean = gamma_mean;
        double current_loading = 1.0;
        int stored_test_parents = 0;
        for(int r = 0; r < k; r++) {
            double a = coefficient(r);
            unsigned int code = parent[r];
            if(code < (unsigned int)n) {
                current_mean += a * (y_train(code) - gamma_mean);
                current_loading -= a;
            } else {
                int previous = code - n;
                current_mean += a * (mu(previous) - gamma_mean);
                current_loading += a * (mean_loading(previous) - 1.0);
                test_parent_index(i, stored_test_parents) = previous;
                test_parent_coefficient(i, stored_test_parents) = a;
                stored_test_parents++;
            }
        }
        mu(i) = current_mean;
        mean_loading(i) = current_loading;
        test_parent_count(i) = stored_test_parents;
    }

    List out = List::create(
        Named("mu") = mu,
        Named("D_star") = D_star,
        Named("mean_loading") = mean_loading,
        Named("test_parent_index") = test_parent_index,
        Named("test_parent_coefficient") = test_parent_coefficient,
        Named("test_parent_count") = test_parent_count,
        Named("gamma_var_factor") = gamma_var_factor,
        Named("scale_factor") = sigma2_scale / sigma2_shape,
        Named("variance_factor") = sigma2_scale / (sigma2_shape - 1.0),
        Named("df") = 2.0 * sigma2_shape,
        Named("prediction_target") = predict_obs ? "observation" : "latent_function"
    );

    if(return_covariance) {
        mat omega(np, np, fill::zeros);
        for(int i = 0; i < np; i++) {
            int count = test_parent_count(i);
            for(int ell = 0; ell < i; ell++) {
                double value = 0.0;
                for(int r = 0; r < count; r++) {
                    int previous = test_parent_index(i, r);
                    value += test_parent_coefficient(i, r) * omega(previous, ell);
                }
                omega(i, ell) = value;
                omega(ell, i) = value;
            }
            double value = D_star(i);
            for(int r = 0; r < count; r++) {
                int previous = test_parent_index(i, r);
                value += test_parent_coefficient(i, r) * omega(i, previous);
            }
            omega(i, i) = value;
        }
        mat lambda = omega + gamma_var_factor * (mean_loading * mean_loading.t());
        lambda = 0.5 * (lambda + lambda.t());
        out["scale_matrix"] = (sigma2_scale / sigma2_shape) * lambda;
        out["covariance"] = (sigma2_scale / (sigma2_shape - 1.0)) * lambda;
    }
    return out;
}
