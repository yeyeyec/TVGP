// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

namespace dense_nig {
using namespace arma;
using namespace Rcpp;

constexpr uword max_n = 2000;
constexpr uword max_d = 2;

void check_design(const vec& theta, const vec& y, const mat& X, double nu) {
    if (X.n_rows > max_n) stop("DENSE NIG supports at most 2000 training rows per leaf.");
    if (X.n_cols < 1 || X.n_cols > max_d)
        stop("DENSE NIG ablation supports one or two input dimensions.");
    if (X.n_rows < 1 || X.n_rows != y.n_elem || theta.n_elem != X.n_cols + 1)
        stop("DENSE NIG requires nonempty, conformable theta, y, and X.");
    if (!X.is_finite() || !y.is_finite()) stop("DENSE NIG requires finite X and y.");
    if (!(std::abs(nu - 0.5) < 1e-8 || std::abs(nu - 1.5) < 1e-8 ||
          std::abs(nu - 2.5) < 1e-8))
        stop("Analytic derivatives require Matern nu in {0.5, 1.5, 2.5}.");
}

struct Kernel { double value, first, second; };

Kernel kernel(double distance, double phi, double nu) {
    // This is the core's range convention: r = distance / phi, no sqrt(2 nu).
    if (distance < 1e-12) return {1.0, 0.0, 0.0};
    const double r = distance / phi;
    if (!std::isfinite(r)) return {0.0, 0.0, 0.0};
    const double e = std::exp(-r);
    if (e == 0.0) return {0.0, 0.0, 0.0};
    const double r2 = r * r;
    if (std::abs(nu - 0.5) < 1e-8)
        return {e, r * e, (r2 - r) * e};
    if (std::abs(nu - 1.5) < 1e-8)
        return {(1.0 + r) * e, r2 * e, r2 * (r - 2.0) * e};
    return {(1.0 + r + r2 / 3.0) * e, r2 * (1.0 + r) * e / 3.0,
            r2 * (r2 - 2.0 * r - 2.0) * e / 3.0};
}

// p = -1 builds K; ell = -1 builds K_p; otherwise builds K_{p,ell}.
// Only one derivative matrix is materialized at a time.
void covariance(mat& out, const mat& X, const vec& phi, double g, double nu,
                int p = -1, int ell = -1) {
    const int d = X.n_cols;
    out.zeros(X.n_rows, X.n_rows);
    if (p == -1) out.diag().fill(1.0 + g);
    if (p == d && (ell == -1 || ell == d)) out.diag().fill(g);
    if (p == d || ell == d) return;
    for (uword i = 0; i < X.n_rows; ++i) {
        for (uword j = 0; j < i; ++j) {
            double value = 1.0;
            for (int k = 0; k < d; ++k) {
                const Kernel z = kernel(std::abs(X(i, k) - X(j, k)), phi(k), nu);
                value *= (p == k && ell == k) ? z.second :
                    ((p == k || ell == k) ? z.first : z.value);
            }
            out(i, j) = out(j, i) = value;
        }
        if (i % 128 == 0) checkUserInterrupt();
    }
}

struct Prior { double nll; vec grad; mat hessian; };

Prior prior(const vec& theta, int code, const vec& pars, bool gradient, bool hessian) {
    const uword d = theta.n_elem - 1;
    const vec phi = exp(theta.head(d));
    const double g = std::exp(theta(d));
    Prior out{0.0, vec(d + 1, fill::zeros), mat(d + 1, d + 1, fill::zeros)};
    if (!pars.is_finite()) stop("Covariance-prior parameters must be finite.");
    if (code == 0) {
        if (pars.n_elem != 4 || any(pars <= 0.0))
            stop("The inverse-gamma covariance prior requires four positive parameters.");
        for (uword j = 0; j <= d; ++j) {
            const double a = pars(j == d ? 2 : 0), b = pars(j == d ? 3 : 1);
            const double inverse = std::exp(-theta(j));
            out.nll += a * theta(j) + b * inverse;
            if (gradient) out.grad(j) = a - b * inverse;
            if (hessian) out.hessian(j, j) = b * inverse;
        }
    } else if (code == 1 || code == 2) {
        if (pars.n_elem != d + 2) stop("The jointly robust prior requires a, b, and C_j.");
        const double a = pars(0), b = pars(1);
        const vec C = pars.tail(d);
        if (a <= -(double(d) + (code == 1 ? 1.0 : 0.0)) || b <= 0.0 || any(C <= 0.0))
            stop("Invalid jointly robust covariance-prior parameters.");
        const vec beta = C / phi;
        const double t = sum(beta), u = t + (code == 1 ? g : 0.0);
        out.nll = -a * std::log(u) + b * (t + g) + sum(theta.head(d)) - theta(d);
        vec du(d + 1, fill::zeros), diagonal(d + 1, fill::zeros);
        du.head(d) = -beta;
        diagonal.head(d) = beta;
        if (code == 1) { du(d) = g; diagonal(d) = g; }
        if (gradient) {
            out.grad = (b - a / u) * du;
            out.grad.head(d) += 1.0;
            out.grad(d) = (code == 1 ? (b - a / u) * g : b * g) - 1.0;
        }
        if (hessian) {
            out.hessian = (a / (u * u)) * (du * du.t()) +
                (b - a / u) * diagmat(diagonal);
            if (code == 2) out.hessian(d, d) = b * g;
        }
    } else stop("cov_prior_code must be 0 (IG), 1 (JR), or 2 (legacy JR).");
    return out;
}

mat lower_solve(const mat& L, const mat& rhs) {
    mat out;
    if (!solve(out, trimatl(L), rhs, solve_opts::fast + solve_opts::no_approx))
        stop("DENSE NIG triangular solve failed.");
    return out;
}

mat upper_solve(const mat& L, const mat& rhs) {
    mat out;
    if (!solve(out, trimatu(L.t()), rhs, solve_opts::fast + solve_opts::no_approx))
        stop("DENSE NIG triangular solve failed.");
    return out;
}

List evaluate(const vec& theta, const vec& y, const mat& X, double nu,
              int prior_code, const vec& prior_params, const vec& leaf,
              bool gradient, bool hessian) {
    check_design(theta, y, X, nu);
    if (leaf.n_elem != 4 || !leaf.is_finite() || any(leaf.tail(3) <= 0.0))
        stop("The proper NIG leaf prior requires finite (m0, V0, a0, b0), V0,a0,b0 > 0.");
    gradient = gradient || hessian;
    const uword n = y.n_elem, q = theta.n_elem;
    const double m0 = leaf(0), V0 = leaf(1), P0 = 1.0 / V0;
    const double a = leaf(2) + 0.5 * n, b0 = leaf(3);
    auto failure = [&](const std::string& reason) {
        mat H(q, q); H.fill(datum::nan);
        const double mean = a > 1.0 ? b0 / (a - 1.0) : R_PosInf;
        return List::create(_["nll"] = 1e100, _["grad"] = vec(q, fill::zeros),
            _["hessian"] = H, _["valid"] = false, _["failure_reason"] = reason,
            _["gamma_mean"] = m0, _["gamma_var_factor"] = V0,
            _["sigma2_shape"] = a, _["sigma2_scale"] = b0,
            _["sigma2_mean"] = mean, _["sigma2_hat"] = mean, _["S0"] = NA_REAL,
            _["value"] = 1e100, _["df"] = 2.0 * a);
    };
    const vec natural = exp(theta);
    if (!theta.is_finite() || !natural.is_finite() || any(natural <= 0.0))
        return failure("invalid_covariance_parameters");
    const vec phi = natural.head(q - 1);
    const double g = natural(q - 1);
    Prior cp = prior(theta, prior_code, prior_params, gradient, hessian);
    mat work, L;
    covariance(work, X, phi, g, nu);
    // Exactly one Cholesky. No inv_sympd, jitter retry, or eigenvalue repair.
    if (!chol(L, work, "lower")) return failure("covariance_cholesky_failed");
    work.reset();
    mat rhs(n, 2);
    rhs.col(0) = y - m0;
    rhs.col(1).ones();
    const mat whitened = lower_solve(L, rhs);
    const vec z = whitened.col(0), u = whitened.col(1);
    const double M = P0 + dot(u, u), delta = dot(u, z) / M;
    const vec residual = z - u * delta;
    // Stable residual Schur complement, avoiding subtraction of large y'Qy terms.
    const double S = dot(residual, residual) + P0 * delta * delta;
    const double b = b0 + 0.5 * S;
    const double value = sum(log(L.diag())) + 0.5 * std::log(M) +
        a * std::log(b) + cp.nll;
    if (!std::isfinite(value) || !std::isfinite(S) || !std::isfinite(M) || M <= 0.0)
        return failure("invalid_proper_nig_quadratic_form");

    if (gradient) {
        const vec alpha = upper_solve(L, residual);
        const vec w = upper_solve(L, u);
        mat P;
        {
            const mat inverse_L = lower_solve(L, eye<mat>(n, n));
            P = inverse_L.t() * inverse_L;
        }
        P -= (w * w.t()) / M;
        L.reset();
        // P = (K + V0 11')^-1. B_p = P K_p; no second-derivative cube.
        std::vector<mat> B(hessian ? q : 0);
        mat v(n, q, fill::zeros), Pv;
        vec dS(q);
        for (uword p = 0; p < q; ++p) {
            if (p == q - 1) {
                v.col(p) = g * alpha;
                dS(p) = -dot(alpha, v.col(p));
                cp.grad(p) += 0.5 * g * trace(P) + 0.5 * a * dS(p) / b;
                if (hessian) B[p] = g * P;
                continue;
            }
            covariance(work, X, phi, g, nu, p);
            v.col(p) = work * alpha;
            dS(p) = -dot(alpha, v.col(p));
            cp.grad(p) += 0.5 * accu(P % work) + 0.5 * a * dS(p) / b;
            if (hessian) B[p] = P * work;
            checkUserInterrupt();
        }
        if (hessian) {
            Pv = P * v;
            for (uword p = 0; p < q; ++p) {
                for (uword ell = 0; ell <= p; ++ell) {
                    covariance(work, X, phi, g, nu, p, ell);
                    const double ddlogdet = accu(P % work) - accu(B[p] % B[ell].t());
                    const double ddS = 2.0 * dot(v.col(p), Pv.col(ell)) -
                        as_scalar(alpha.t() * work * alpha);
                    const double entry = 0.5 * ddlogdet + a *
                        (0.5 * ddS / b - 0.25 * dS(p) * dS(ell) / (b * b));
                    cp.hessian(p, ell) += entry;
                    if (p != ell) cp.hessian(ell, p) += entry;
                }
                checkUserInterrupt();
            }
        }
        if (!cp.grad.is_finite() || !cp.hessian.is_finite())
            return failure("nonfinite_analytic_derivatives");
    }
    const double mean = a > 1.0 ? b / (a - 1.0) : R_PosInf;
    return List::create(_["nll"] = value, _["grad"] = cp.grad,
        _["hessian"] = cp.hessian, _["valid"] = true, _["failure_reason"] = NA_STRING,
        _["gamma_mean"] = m0 + delta, _["gamma_var_factor"] = 1.0 / M,
        _["sigma2_shape"] = a, _["sigma2_scale"] = b,
        _["sigma2_mean"] = mean, _["sigma2_hat"] = mean, _["S0"] = S,
        _["value"] = value, _["df"] = 2.0 * a);
}
} // namespace dense_nig

// [[Rcpp::export]]
Rcpp::List dense_nig_nll_grad(const arma::vec& theta, const arma::vec& y,
                            const arma::mat& X, double nu, int cov_prior_code,
                            const arma::vec& prior_params, const arma::vec& leaf_prior_params,
                            bool calc_grad = true) {
    return dense_nig::evaluate(theta, y, X, nu, cov_prior_code, prior_params,
                               leaf_prior_params, calc_grad, false);
}

// [[Rcpp::export]]
Rcpp::List dense_nig_nll_grad_hessian(const arma::vec& theta, const arma::vec& y,
                                    const arma::mat& X, double nu, int cov_prior_code,
                                    const arma::vec& prior_params,
                                    const arma::vec& leaf_prior_params) {
    return dense_nig::evaluate(theta, y, X, nu, cov_prior_code, prior_params,
                               leaf_prior_params, true, true);
}

// [[Rcpp::export]]
Rcpp::List dense_nig_prediction_moments(
    const arma::vec& y_train, const arma::mat& X_train, const arma::mat& X_test,
    const arma::vec& theta, double gamma_mean, double gamma_var_factor,
    double sigma2_shape, double sigma2_scale, double nu, int chunk_size = 128) {
    using namespace dense_nig;
    check_design(theta, y_train, X_train, nu);
    if (X_test.n_cols != X_train.n_cols || !X_test.is_finite())
        stop("Prediction X_test must be finite with the training design's columns.");
    if (chunk_size < 1 || chunk_size > 2000) stop("chunk_size must be in [1, 2000].");
    const vec natural = exp(theta);
    if (!theta.is_finite() || !natural.is_finite() || any(natural <= 0.0))
        stop("Invalid dense prediction covariance parameters.");
    if (!std::isfinite(gamma_mean) || !std::isfinite(gamma_var_factor) ||
        !std::isfinite(sigma2_shape) || !std::isfinite(sigma2_scale) ||
        gamma_var_factor <= 0.0 || sigma2_shape <= 0.0 || sigma2_scale <= 0.0)
        stop("Invalid proper-NIG posterior prediction parameters.");
    const uword n = X_train.n_rows, nt = X_test.n_rows, d = X_train.n_cols;
    const vec phi = natural.head(d);
    mat K, L;
    covariance(K, X_train, phi, natural(d), nu);
    if (!chol(L, K, "lower")) stop("Dense prediction covariance Cholesky failed.");
    K.reset();
    mat rhs(n, 2);
    rhs.col(0) = y_train - gamma_mean;
    rhs.col(1).ones();
    const mat whitened = lower_solve(L, rhs);
    vec mu(nt), variance(nt), scale2(nt), df(nt);
    df.fill(2.0 * sigma2_shape);
    for (uword start = 0; start < nt; start += chunk_size) {
        const uword count = std::min<uword>(chunk_size, nt - start);
        mat cross(n, count);
        for (uword j = 0; j < count; ++j) {
            for (uword i = 0; i < n; ++i) {
                double value = 1.0;
                for (uword k = 0; k < d; ++k)
                    value *= kernel(std::abs(X_train(i, k) - X_test(start + j, k)), phi(k), nu).value;
                cross(i, j) = value;
            }
        }
        const mat solved = lower_solve(L, cross);
        for (uword j = 0; j < count; ++j) {
            const double loading = 1.0 - dot(solved.col(j), whitened.col(1));
            double factor = 1.0 - dot(solved.col(j), solved.col(j)) +
                gamma_var_factor * loading * loading;
            if (!std::isfinite(factor) || factor < -1e-10)
                stop("Invalid dense posterior predictive variance factor.");
            factor = std::max(0.0, factor); // Only roundoff; no positive variance floor.
            mu(start + j) = gamma_mean + dot(solved.col(j), whitened.col(0));
            scale2(start + j) = sigma2_scale / sigma2_shape * factor;
            variance(start + j) = sigma2_shape > 1.0 ?
                sigma2_scale / (sigma2_shape - 1.0) * factor :
                (factor == 0.0 ? 0.0 : R_PosInf);
        }
        checkUserInterrupt();
    }
    return List::create(_["mu"] = mu, _["variance"] = variance, _["scale2"] = scale2,
        _["df"] = df, _["gamma_mean"] = gamma_mean,
        _["prediction_target"] = "latent_function", _["prediction_mode"] = "pointwise");
}
