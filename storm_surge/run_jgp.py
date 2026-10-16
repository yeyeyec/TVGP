#!/usr/bin/env python3
"""Run Park's local Jump GP on one fixed manuscript dataset."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.special import ndtr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "JumpGaussianProcess"
sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(PROJECT_DIR / "vendor"))

from jumpgp_prediction_adapter import compute_external_predictions

UPSTREAM_COMMIT = "2eccbc30ceb4d38521a88137856b65761f4a9030"

_WORK_X = None
_WORK_Y = None
_WORK_XT = None
_WORK_NEIGHBORS = None
_WORK_BOUNDARY = None
_WORK_MODE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--rep", required=True, type=int)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument(
        "--boundary", choices=("auto", "linear", "quadratic"), default="auto"
    )
    parser.add_argument("--neighbors", type=int, default=0)
    parser.add_argument("--mode", choices=("CEM", "VEM"), default="CEM")
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--prediction-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--allow-failures", type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Dataset has no header: {}".format(path))
        input_names = [name for name in reader.fieldnames if name.startswith("x")]
        required = {"y_obs", "truth"}
        missing = required.difference(reader.fieldnames)
        if missing or not input_names:
            raise ValueError(
                "Invalid dataset {}: missing {}, input columns {}".format(
                    path, sorted(missing), input_names
                )
            )
        rows = list(reader)
    x = np.asarray(
        [[float(row[name]) for name in input_names] for row in rows], dtype=float
    )
    y = np.asarray([float(row["y_obs"]) for row in rows], dtype=float)
    truth = np.asarray([float(row["truth"]) for row in rows], dtype=float)
    sigma = np.asarray(
        [float(row.get("sigma", "nan")) for row in rows], dtype=float
    )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite training values in {}".format(path))
    if not np.all(np.isfinite(truth)):
        raise ValueError("Non-finite truth values in {}".format(path))
    return {
        "x": x,
        "y": y,
        "truth": truth,
        "sigma": sigma,
        "input_names": input_names,
    }


def choose_settings(dimension, boundary, neighbors):
    if boundary == "auto":
        boundary = "quadratic" if dimension <= 3 else "linear"
    if neighbors <= 0:
        neighbors = 25
    return boundary, int(neighbors)


def initialize_worker(x, y, xt, neighbors, boundary, mode):
    global _WORK_X, _WORK_Y, _WORK_XT, _WORK_NEIGHBORS
    global _WORK_BOUNDARY, _WORK_MODE
    _WORK_X = x
    _WORK_Y = y
    _WORK_XT = xt
    _WORK_NEIGHBORS = neighbors
    _WORK_BOUNDARY = boundary
    _WORK_MODE = mode


def fit_test_index(index):
    try:
        local_index = _WORK_NEIGHBORS[index]
        x_local = _WORK_X[local_index, :]
        y_local = _WORK_Y[local_index].reshape(-1, 1)
        x_test = _WORK_XT[index, :].reshape(1, -1)

        # The simulator contains large exact-zero regions.  A constant local
        # sample is the limiting local GP fit and avoids a zero-variance MLE.
        if float(np.ptp(y_local)) <= 1e-12:
            value = float(y_local.reshape(-1)[0])
            return index, value, 0.0, 0.0, len(y_local), ""

        if _WORK_BOUNDARY == "quadratic":
            from JumpGP_QD import JumpGP_QD

            mean, variance, model, _ = JumpGP_QD(
                x_local, y_local, x_test, _WORK_MODE, False
            )
        else:
            from JumpGP_LD import JumpGP_LD

            mean, variance, model, _ = JumpGP_LD(
                x_local, y_local, x_test, _WORK_MODE, False
            )

        prediction = compute_external_predictions(model)
        mean = float(prediction["uk_mean"])
        total_variance = float(prediction["uk_response_variance"])
        noise_variance = float(prediction["noise_variance"])
        selected = int(prediction["selected_neighbors"])
        values = (mean, total_variance, noise_variance)
        if not all(np.isfinite(value) for value in values):
            raise FloatingPointError("non-finite JGP prediction")
        return index, mean, total_variance, noise_variance, selected, ""
    except Exception as error:
        detail = "{}: {}".format(type(error).__name__, error)
        return index, np.nan, np.nan, np.nan, -1, detail


def atomic_checkpoint(path, mean, total_variance, noise_variance, selected, done,
                      elapsed_seconds):
    temporary = Path(str(path) + ".tmp.npz")
    np.savez_compressed(
        temporary,
        mean=mean,
        total_variance=total_variance,
        noise_variance=noise_variance,
        selected=selected,
        done=done,
        elapsed_seconds=np.asarray([elapsed_seconds], dtype=float),
    )
    os.replace(str(temporary), str(path))


def load_checkpoint(path, n_test):
    if not path.exists():
        return (
            np.full(n_test, np.nan),
            np.full(n_test, np.nan),
            np.full(n_test, np.nan),
            np.full(n_test, -1, dtype=int),
            np.zeros(n_test, dtype=bool),
            0.0,
        )
    saved = np.load(path)
    if len(saved["mean"]) != n_test:
        raise ValueError("Checkpoint size does not match the test set.")
    elapsed_values = (
        saved["elapsed_seconds"]
        if "elapsed_seconds" in saved.files
        else np.asarray([0.0])
    )
    elapsed = float(np.asarray(elapsed_values)[0])
    return (
        saved["mean"].astype(float),
        saved["total_variance"].astype(float),
        saved["noise_variance"].astype(float),
        saved["selected"].astype(int),
        saved["done"].astype(bool),
        elapsed,
    )


def gaussian_crps(y, mean, sd):
    y = np.asarray(y, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    output = np.abs(y - mean)
    positive = sd > 0
    z = (y[positive] - mean[positive]) / sd[positive]
    output[positive] = sd[positive] * (
        z * (2 * ndtr(z) - 1)
        + 2 * np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        - 1 / math.sqrt(math.pi)
    )
    return output


def sample_crps(y, draws):
    """CRPS of each row of predictive draws, matching the manuscript R code."""
    ordered = np.sort(np.asarray(draws, dtype=float), axis=1)
    n_draws = ordered.shape[1]
    coefficients = 2 * np.arange(1, n_draws + 1) - n_draws - 1
    pair_term = np.sum(ordered * coefficients[np.newaxis, :], axis=1)
    pair_term = pair_term / float(n_draws * n_draws)
    return np.mean(np.abs(draws - y[:, np.newaxis]), axis=1) - pair_term


def write_rows_gzip(path, fieldnames, rows):
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path, row):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = read_dataset(args.train)
    test = read_dataset(args.test)
    if train["input_names"] != test["input_names"]:
        raise ValueError("Training and test input columns differ.")
    if train["x"].shape[1] != test["x"].shape[1]:
        raise ValueError("Training and test dimensions differ.")

    dimension = train["x"].shape[1]
    boundary, neighbors = choose_settings(
        dimension, args.boundary, args.neighbors
    )
    neighbors = min(neighbors, train["x"].shape[0])
    cores = max(1, int(args.cores))
    prefix = "jgp_{}_rep{:03d}".format(args.experiment, args.rep)
    checkpoint_path = args.output_dir / (prefix + "_checkpoint.npz")
    error_path = args.output_dir / (prefix + "_errors.csv")

    y_center = float(np.mean(train["y"]))
    y_scale = float(np.std(train["y"], ddof=1))
    if not np.isfinite(y_scale) or y_scale <= 1e-12:
        y_scale = 1.0
    y_standard = (train["y"] - y_center) / y_scale

    start = time.monotonic()
    tree = cKDTree(train["x"])
    try:
        _, neighbor_index = tree.query(
            test["x"], k=neighbors, workers=cores
        )
    except TypeError:
        _, neighbor_index = tree.query(test["x"], k=neighbors)
    if neighbors == 1:
        neighbor_index = neighbor_index[:, np.newaxis]

    (mean_standard, total_variance_standard, noise_variance_standard,
     selected, done, prior_elapsed) = load_checkpoint(
        checkpoint_path, test["x"].shape[0]
    )
    pending = np.flatnonzero(~done)
    errors = {}
    completed_since_checkpoint = 0

    def record_result(result):
        nonlocal completed_since_checkpoint
        index, mean, total_variance, noise_variance, n_selected, error = result
        if error:
            errors[index] = error
        else:
            mean_standard[index] = mean
            total_variance_standard[index] = total_variance
            noise_variance_standard[index] = noise_variance
            selected[index] = n_selected
            done[index] = True
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= max(1, args.checkpoint_every):
            elapsed = prior_elapsed + time.monotonic() - start
            atomic_checkpoint(
                checkpoint_path, mean_standard,
                total_variance_standard, noise_variance_standard,
                selected, done, elapsed,
            )
            completed_since_checkpoint = 0
            print(
                "{}: {}/{} predictions complete".format(
                    prefix, int(np.sum(done)), len(done)
                ),
                flush=True,
            )

    if len(pending) and cores == 1:
        initialize_worker(
            train["x"], y_standard, test["x"], neighbor_index,
            boundary, args.mode,
        )
        for index in pending:
            record_result(fit_test_index(int(index)))
    elif len(pending):
        with ProcessPoolExecutor(
            max_workers=cores,
            initializer=initialize_worker,
            initargs=(
                train["x"], y_standard, test["x"], neighbor_index,
                boundary, args.mode,
            ),
        ) as executor:
            futures = {executor.submit(fit_test_index, int(i)): int(i) for i in pending}
            for future in as_completed(futures):
                record_result(future.result())

    elapsed_seconds = prior_elapsed + time.monotonic() - start
    atomic_checkpoint(
        checkpoint_path, mean_standard, total_variance_standard,
        noise_variance_standard, selected, done, elapsed_seconds,
    )
    if errors:
        with open(error_path, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["test_id", "error"])
            for index in sorted(errors):
                writer.writerow([index + 1, errors[index]])
    elif error_path.exists():
        error_path.unlink()

    n_failed = int(np.sum(~done))
    if n_failed > int(args.allow_failures):
        raise RuntimeError(
            "{} JGP predictions failed; see {}".format(n_failed, error_path)
        )
    valid = done & np.isfinite(mean_standard)
    if not np.any(valid):
        raise RuntimeError("No valid JGP predictions were produced.")

    mean = mean_standard * y_scale + y_center
    total_variance = np.maximum(total_variance_standard, 0.0) * y_scale ** 2
    noise_variance = np.maximum(noise_variance_standard, 0.0) * y_scale ** 2
    latent_variance_raw = total_variance - noise_variance
    latent_variance = np.maximum(latent_variance_raw, 0.0)
    latent_sd = np.sqrt(latent_variance)
    observation_sd = np.sqrt(total_variance)
    truth = test["truth"]
    z975 = 1.959963984540054
    analytic_lower = mean - z975 * latent_sd
    analytic_upper = mean + z975 * latent_sd
    n_draws = max(2, int(args.draws))
    prediction_seed = (
        int(args.prediction_seed)
        if int(args.prediction_seed) > 0
        else int(args.rep) + 29000
    )
    rng = np.random.RandomState(prediction_seed)
    latent_draws = (
        mean[:, np.newaxis]
        + latent_sd[:, np.newaxis] * rng.standard_normal((len(mean), n_draws))
    )
    lower = np.quantile(latent_draws, 0.025, axis=1)
    upper = np.quantile(latent_draws, 0.975, axis=1)
    pointwise_sample_crps = sample_crps(truth, latent_draws)
    pointwise_analytic_crps = gaussian_crps(truth, mean, latent_sd)

    suffix = "Q" if boundary == "quadratic" else "L"
    method = "JGP-{}-{}".format(args.mode, suffix)
    metric_row = {
        "Experiment": args.experiment,
        "Repetition": args.rep,
        "Method": method,
        "RMSE": float(np.sqrt(np.mean((mean[valid] - truth[valid]) ** 2))),
        "CRPS": float(np.mean(pointwise_sample_crps[valid])),
        "Coverage": float(np.mean(
            (truth[valid] >= lower[valid]) & (truth[valid] <= upper[valid])
        )),
        "MeanWidth": float(np.mean(upper[valid] - lower[valid])),
        "Analytic_Gaussian_CRPS": float(np.mean(
            pointwise_analytic_crps[valid]
        )),
        "Analytic_Coverage": float(np.mean(
            (truth[valid] >= analytic_lower[valid])
            & (truth[valid] <= analytic_upper[valid])
        )),
        "Analytic_MeanWidth": float(np.mean(
            analytic_upper[valid] - analytic_lower[valid]
        )),
        "Time_s": elapsed_seconds,
        "Fit_s": 0.0,
        "Pred_s": elapsed_seconds,
        "n_train": train["x"].shape[0],
        "n_test": test["x"].shape[0],
        "n_valid": int(np.sum(valid)),
        "n_failed": n_failed,
        "dimension": dimension,
        "boundary": boundary,
        "mode": args.mode,
        "neighbors": neighbors,
        "predictive_draws": n_draws,
        "prediction_seed": prediction_seed,
        "cores": cores,
        "mean_selected_neighbors": float(np.mean(selected[valid])),
        "mean_estimated_nugget_sd": float(np.mean(np.sqrt(noise_variance[valid]))),
        "latent_variance_clipped_fraction": float(np.mean(
            latent_variance_raw[valid] < 0
        )),
    }
    write_metrics(args.output_dir / (prefix + "_metrics.csv"), metric_row)

    fields = [
        "test_id", "truth", "y_obs", "pred_mean", "pred_sd_latent",
        "pred_sd_observation", "estimated_nugget_sd", "lower95", "upper95",
        "covered95", "analytic_lower95", "analytic_upper95",
        "analytic_covered95", "sample_crps", "analytic_gaussian_crps",
        "selected_neighbors",
    ] + test["input_names"]
    rows = []
    for index in range(len(truth)):
        row = {
            "test_id": index + 1,
            "truth": truth[index],
            "y_obs": test["y"][index],
            "pred_mean": mean[index],
            "pred_sd_latent": latent_sd[index],
            "pred_sd_observation": observation_sd[index],
            "estimated_nugget_sd": math.sqrt(noise_variance[index]),
            "lower95": lower[index],
            "upper95": upper[index],
            "covered95": int(lower[index] <= truth[index] <= upper[index]),
            "analytic_lower95": analytic_lower[index],
            "analytic_upper95": analytic_upper[index],
            "analytic_covered95": int(
                analytic_lower[index] <= truth[index] <= analytic_upper[index]
            ),
            "sample_crps": pointwise_sample_crps[index],
            "analytic_gaussian_crps": pointwise_analytic_crps[index],
            "selected_neighbors": selected[index],
        }
        row.update({
            name: test["x"][index, column]
            for column, name in enumerate(test["input_names"])
        })
        rows.append(row)
    write_rows_gzip(
        args.output_dir / (prefix + "_prediction_diagnostics.csv.gz"),
        fields,
        rows,
    )
    np.savez_compressed(
        args.output_dir / (prefix + "_predictions.npz"),
        mean=mean,
        latent_variance=latent_variance,
        observation_variance=total_variance,
        noise_variance=noise_variance,
        selected_neighbors=selected,
        truth=truth,
        x_test=test["x"],
    )

    run_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "repetition": args.rep,
        "method": method,
        "upstream_repository": "https://github.com/crushonyfg/JumpGaussianProcess",
        "upstream_commit": UPSTREAM_COMMIT,
        "compatibility_patch": "vendor/jumpgp_compatibility.patch",
        "compatibility_patch_sha256": sha256_file(
            PROJECT_DIR / "vendor" / "jumpgp_compatibility.patch"
        ),
        "boundary_rule": "quadratic for d <= 3; linear for d > 3" if args.boundary == "auto" else "explicit",
        "neighbor_rule": "selected once by training-only validation" if args.neighbors > 0 else "default",
        "mode": args.mode,
        "response_transform": {
            "type": "training z-score",
            "center": y_center,
            "scale": y_scale,
        },
        "scoring_target": "latent function",
        "primary_scoring": {
            "rmse": "predictive mean",
            "crps_and_interval": "{} latent Gaussian draws".format(n_draws),
            "prediction_seed": prediction_seed,
        },
        "variance_note": "external universal-kriging latent variance from the fitted public CEM model",
        "train_file": str(args.train.resolve()),
        "test_file": str(args.test.resolve()),
        "train_sha256": sha256_file(args.train),
        "test_sha256": sha256_file(args.test),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cores": cores,
        "elapsed_seconds": elapsed_seconds,
    }
    with open(args.output_dir / (prefix + "_run_record.json"), "w") as stream:
        json.dump(run_record, stream, indent=2, sort_keys=True)
    print(json.dumps(metric_row, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
