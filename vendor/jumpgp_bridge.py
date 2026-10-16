#!/usr/bin/env python3
"""File-only R bridge to the pinned public CEM model and external UK adapter.

No old experiment driver is imported. Truth/test responses are never accepted.
"""

import argparse
import csv
import os
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "tvgp_baseline_mpl"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "JumpGaussianProcess"))


def check_dependencies():
    import numpy as np
    import scipy
    import matplotlib
    if np.__version__ != "1.26.4" or scipy.__version__ != "1.13.1" or matplotlib.__version__ != "3.8.4":
        raise RuntimeError("JumpGP requires its tested environment: numpy==1.26.4, "
                           "scipy==1.13.1, matplotlib==3.8.4; found {}, {}, {}".format(
                               np.__version__, scipy.__version__, matplotlib.__version__))
    from JumpGP_LD import JumpGP_LD
    from JumpGP_QD import JumpGP_QD
    from jumpgp_prediction_adapter import compute_external_predictions
    return np, JumpGP_LD, JumpGP_QD, compute_external_predictions


def initialize_worker(x, y, neighbors, seed):
    global _X, _Y, _NEIGHBORS, _SEED
    _X, _Y, _NEIGHBORS, _SEED = x, y, neighbors, seed


def fit_one(task):
    np, linear, quadratic, _ = check_dependencies()
    row, xt, nn = task
    # A local seed makes results invariant to core count and batch scheduling.
    np.random.seed((_SEED + row) % (2**32))
    model_fun = quadratic if _X.shape[1] <= 3 else linear
    try:
        _, _, model, _ = model_fun(_X[nn, :], _Y[nn].reshape(-1, 1),
                                   xt.reshape(1, -1), "CEM", False)
        for name in ("mu_t", "sig2_t", "logtheta"):
            if not np.all(np.isfinite(np.asarray(model[name], dtype=float))):
                raise ValueError("Non-finite public field " + name)
        return row, model, ""
    except Exception as error:
        return row, None, "{}: {}".format(type(error).__name__, error)


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    np, linear, quadratic, adapt = check_dependencies()
    rng = np.random.RandomState(314)
    x = rng.uniform(0.05, 0.95, (32, 2))
    y = (np.sin(4 * x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, 0.05, 32)).reshape(-1, 1)
    for fun in (linear, quadratic):
        _, _, model, _ = fun(x, y, np.array([[0.45, 0.55]]), "CEM", False)
        p = adapt(model)
        if not all(np.isfinite(v) for v in p.values()):
            raise AssertionError("Non-finite public/external UK prediction")
        if abs(p["uk_response_variance"] - p["uk_latent_variance"] - p["noise_variance"]) > 1e-9:
            raise AssertionError("Latent/noise decomposition failed")
        if p["public_reconstruction_checked"] != 1:
            raise AssertionError("Smoke model did not reconstruct public prediction")
    print("PASS public LD/QD CEM and external UK, Python {}".format(sys.version.split()[0]))


def run(args):
    np, _, _, adapt = check_dependencies()
    from scipy.spatial import cKDTree
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train = np.loadtxt(args.train, delimiter=",", ndmin=2)
    test = np.loadtxt(args.test, delimiter=",", ndmin=2)
    x, y = train[:, :-1], train[:, -1]
    if (x.shape[1] != test.shape[1] or not np.all(np.isfinite(train))
            or not np.all(np.isfinite(test))):
        raise ValueError("Invalid train/test dimensions or non-finite input")
    dim = x.shape[1]
    n_features = 1 + dim + (dim * (dim + 1) // 2 if dim <= 3 else 0)
    neighbors = min(args.neighbors, len(x))
    if neighbors <= n_features:
        raise ValueError("CEM neighborhood must exceed boundary feature count {}".format(n_features))
    center, scale = float(np.mean(y)), float(np.std(y, ddof=1))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    standardized = (y - center) / scale
    tic = time.perf_counter()
    tree = cKDTree(x)
    setup_fit_s = time.perf_counter() - tic
    executor = None
    if args.cores > 1:
        executor = ProcessPoolExecutor(max_workers=args.cores, initializer=initialize_worker,
            initargs=(x, standardized, neighbors, args.seed_fit))
    else:
        initialize_worker(x, standardized, neighbors, args.seed_fit)

    def predict(points, name):
        if points.shape[1] != dim or not np.all(np.isfinite(points)):
            raise ValueError("Invalid " + name + " inputs")
        fit_s = pred_s = 0.0
        rows = []
        for start in range(0, len(points), 64):
            end = min(start + 64, len(points))
            tic = time.perf_counter()
            nn = tree.query(points[start:end], k=neighbors)[1]
            tasks = [(i, points[i], nn[i - start]) for i in range(start, end)]
            pred_s += time.perf_counter() - tic
            tic = time.perf_counter()
            fitted = list(executor.map(fit_one, tasks)) if executor else list(map(fit_one, tasks))
            fit_s += time.perf_counter() - tic
            tic = time.perf_counter()
            for row, model, error in fitted:
                if error:
                    write_csv(outdir / (name + "_failure.csv"), [{"row": row + 1, "phase": "public CEM", "error": error}])
                    raise RuntimeError("Public CEM failed at {} row {}: {}".format(name, row + 1, error))
                try:
                    values = adapt(model)
                    if not all(np.isfinite(v) for v in values.values()):
                        raise ValueError("Non-finite external prediction/diagnostic")
                except Exception as error:
                    write_csv(outdir / (name + "_failure.csv"), [{"row": row + 1, "phase": "external UK", "error": str(error)}])
                    raise RuntimeError("External UK failed at {} row {}: {}".format(name, row + 1, error)) from error
                record = {"row": row + 1}
                # Preserve untouched standardized public/UK fields as well as original units.
                record.update({key + "_standardized": value for key, value in values.items()})
                for key in ("public_mean", "uk_mean", "sample_mean", "gls_mean", "plugin_mean_external"):
                    record[key] = center + scale * values[key]
                for key in ("public_response_variance", "uk_latent_variance", "uk_response_variance",
                            "noise_variance", "mean_uncertainty", "plugin_latent_variance_external",
                            "plugin_response_variance_external"):
                    record[key] = scale**2 * values[key]
                record["y_center"], record["y_scale"] = center, scale
                for j, value in enumerate(np.asarray(model["logtheta"]).reshape(-1)):
                    record["logtheta_{}".format(j + 1)] = float(value)
                rows.append(record)
            pred_s += time.perf_counter() - tic
        write_csv(outdir / (name + ".csv"), rows)
        return fit_s, pred_s

    try:
        fit_s, pred_s = predict(test, "predictions")
        surface_s = 0.0
        if args.surface:
            tic = time.perf_counter()
            predict(np.loadtxt(args.surface, delimiter=",", ndmin=2), "surface_predictions")
            surface_s = time.perf_counter() - tic
        write_csv(outdir / "timing.csv", [{"fit_s": fit_s + setup_fit_s, "pred_s": pred_s,
                                           "surface_s": surface_s}])
    finally:
        if executor:
            executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    for name in ("train", "test", "outdir", "surface"):
        parser.add_argument("--" + name)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=25)
    parser.add_argument("--seed-fit", type=int, default=50001)
    parser.add_argument("--seed-predict", type=int, default=21001)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.check:
        check_dependencies()
        print("JumpGP pinned Python dependencies and public CEM imports OK")
    else:
        if not all((args.train, args.test, args.outdir)) or args.cores < 1:
            parser.error("--train, --test, --outdir and positive --cores are required")
        run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
