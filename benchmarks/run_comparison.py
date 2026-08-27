"""
run_comparison.py
==================
Quantitative benchmark of CASCADE (CWT + physics-constrained Adam) against
classic Levenberg-Marquardt (LM) fitting, addressing reviewer requests for a
direct, quantitative speed/robustness comparison (Analyst review, Referees 1
& 3) rather than only literature-based claims about LM's shortcomings.

Two LM baselines are evaluated (see benchmarks/lm_baseline.py):
  - LM + CWT p0    : scipy.optimize.least_squares(method='lm'), seeded with
                      the EXACT SAME CWT-derived initial guess CASCADE uses.
                      Isolates the optimizer (Adam vs. LM) as the only
                      difference from CASCADE.
  - LM + naive init: scipy LM seeded with a conventional single-scale
                      scipy.signal.find_peaks initialization on the raw
                      spectrum (no wavelet decomposition) — representative
                      of typical unassisted LM/curve_fit practice.

All three methods fit the identical pseudo-Voigt peak model, so any
difference in outcome is attributable to initialization and/or optimizer,
not to a different model class.

Experiment 1 (accuracy/robustness/speed): the SAME three validated
noise/separation/peak-count conditions used for Table 1 of the manuscript
(Ideal, Realistic, Poor; n_peaks in [40,50], matching _run_sweep defaults).

Experiment 2 (computational scaling): fixed "Realistic" noise/separation,
sweeping the true peak count to show how per-spectrum fit time scales with
the number of fitted parameters for each method.

Results are written incrementally to CSV under benchmarks/results/ so
partial progress survives an interruption.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch

from cascade.dataset_utils import RamanDataset
from cascade.tidytorch_utils import init_sweep_context, process_conv_deriv_fit, _match_peaks
from lm_baseline import lm_fit_from_p0, lm_fit_naive_init, naive_initial_guess

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X = np.linspace(300, 1800, 512, dtype=np.float32)
WIDTHS = np.linspace(3, 60, 20, dtype=np.float32)
DEVICE = torch.device("cpu")
SIGMAS = torch.as_tensor(WIDTHS, dtype=torch.float32)
GAMMAS = torch.tensor([5.0], dtype=torch.float32)
X_T = torch.as_tensor(X)

CWT_KW = dict(
    response_threshold=0.02, min_scale_votes=6,
    min_spacing_in=7.0, min_spacing_post=7.0,
    max_peaks=80, max_iter=2000, tol=1e-5, convolution="Lor4",
)


def _scaled_max_nfev(n_params, per_param=60, cap=4000):
    """Give LM a standard-scaled evaluation budget (~ scipy's own default of
    100*(n+1) for method='lm'), capped so a single pathological spectrum
    cannot dominate the whole benchmark's wall-clock time.
    """
    return int(min(per_param * (n_params + 1), cap))

CONDITIONS = {
    "Ideal":     dict(noise_std=0.0,   separability_range=(2.5, 4.0)),
    "Realistic": dict(noise_std=0.015, separability_range=(1.0, 1.5)),
    "Poor":      dict(noise_std=0.10,  separability_range=(0.5, 0.8)),
}


def setup_context():
    init_sweep_context(X, SIGMAS, GAMMAS, DEVICE, WIDTHS)


def fit_cascade(spectrum_t):
    t0 = time.perf_counter()
    params, converged, n_iter, resp, mask, p0 = process_conv_deriv_fit(
        spectrum_t, X_T, SIGMAS, GAMMAS, **CWT_KW
    )
    dt = time.perf_counter() - t0
    return params, p0, dt, n_iter


def run_experiment_1(n_samples=6, seed=42):
    out_path = os.path.join(RESULTS_DIR, "exp1_accuracy_speed.csv")
    fieldnames = [
        "condition", "sample", "n_gt",
        "method", "time_s", "f1", "precision", "recall",
        "mean_amp_err", "mean_ctr_err", "mean_shape_rmse",
        "n_params", "nfev", "lm_success",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for cond_name, ds_kw in CONDITIONS.items():
            ds = RamanDataset(
                x=X, widths=WIDTHS, n_samples=n_samples, n_peaks=(40, 50),
                seed=seed, **ds_kw,
            )
            for i in range(n_samples):
                sample = ds[i]
                spectrum_np = np.asarray(sample[8], dtype=np.float32)
                spectrum_t = torch.as_tensor(spectrum_np)
                n_gt = len(sample[3])
                gt_args = (sample[3], sample[4], sample[7], sample[9])

                # --- CASCADE ---
                params_c, p0, dt_c, n_iter = fit_cascade(spectrum_t)
                stats_c = _match_peaks(*gt_args, params_c, tolerance=15.0,
                                        amp_threshold=1e-2, x_arr=X)
                row = dict(condition=cond_name, sample=i, n_gt=n_gt,
                           method="CASCADE", time_s=dt_c, f1=stats_c["f1"],
                           precision=stats_c["precision"], recall=stats_c["recall"],
                           mean_amp_err=stats_c["mean_amp_err"], mean_ctr_err=stats_c["mean_ctr_err"],
                           mean_shape_rmse=stats_c["mean_shape_rmse"],
                           n_params=int((p0.reshape(-1, 4)[:, 0] > 0).sum().item()) * 4,
                           nfev="", lm_success="")
                writer.writerow(row); fh.flush()
                print(f"[{cond_name} #{i}] CASCADE   t={dt_c:6.2f}s F1={stats_c['f1']:.3f} "
                      f"n_gt={n_gt} n_rec={stats_c['n_rec']}", flush=True)

                p0_np = p0.cpu().numpy()
                n_p0 = int((p0_np.reshape(-1, 4)[:, 0] > 0).sum())
                nfev_budget = _scaled_max_nfev(n_p0 * 4)

                # --- LM + same CWT p0 ---
                t0 = time.perf_counter()
                lm_params, cost, success, nfev = lm_fit_from_p0(spectrum_np, X, p0_np, max_nfev=nfev_budget)
                dt_lm = time.perf_counter() - t0
                stats_lm = _match_peaks(*gt_args, lm_params, tolerance=15.0,
                                         amp_threshold=1e-2, x_arr=X)
                row = dict(condition=cond_name, sample=i, n_gt=n_gt,
                           method="LM+CWT_p0", time_s=dt_lm, f1=stats_lm["f1"],
                           precision=stats_lm["precision"], recall=stats_lm["recall"],
                           mean_amp_err=stats_lm["mean_amp_err"], mean_ctr_err=stats_lm["mean_ctr_err"],
                           mean_shape_rmse=stats_lm["mean_shape_rmse"],
                           n_params=lm_params.size, nfev=nfev, lm_success=success)
                writer.writerow(row); fh.flush()
                print(f"[{cond_name} #{i}] LM+CWT_p0 t={dt_lm:6.2f}s F1={stats_lm['f1']:.3f} "
                      f"nfev={nfev} success={success}", flush=True)

                # --- LM + naive (conventional) init ---
                naive_p0 = naive_initial_guess(spectrum_np, X)
                nfev_budget_naive = _scaled_max_nfev(naive_p0.size)
                t0 = time.perf_counter()
                lm_params2, cost2, success2, nfev2 = lm_fit_naive_init(spectrum_np, X, max_nfev=nfev_budget_naive)
                dt_lm2 = time.perf_counter() - t0
                stats_lm2 = _match_peaks(*gt_args, lm_params2, tolerance=15.0,
                                          amp_threshold=1e-2, x_arr=X)
                row = dict(condition=cond_name, sample=i, n_gt=n_gt,
                           method="LM+naive", time_s=dt_lm2, f1=stats_lm2["f1"],
                           precision=stats_lm2["precision"], recall=stats_lm2["recall"],
                           mean_amp_err=stats_lm2["mean_amp_err"], mean_ctr_err=stats_lm2["mean_ctr_err"],
                           mean_shape_rmse=stats_lm2["mean_shape_rmse"],
                           n_params=lm_params2.size, nfev=nfev2, lm_success=success2)
                writer.writerow(row); fh.flush()
                print(f"[{cond_name} #{i}] LM+naive  t={dt_lm2:6.2f}s F1={stats_lm2['f1']:.3f} "
                      f"nfev={nfev2} success={success2}", flush=True)

    print(f"\nExperiment 1 complete -> {out_path}")
    return out_path


def run_experiment_2(peak_levels=(5, 10, 20, 40, 60), n_reps=3, seed=7):
    out_path = os.path.join(RESULTS_DIR, "exp2_scaling.csv")
    fieldnames = ["n_peaks_true", "rep", "n_params", "method", "time_s", "nfev", "lm_success"]
    ds_kw = dict(noise_std=0.015, separability_range=(1.0, 1.5))

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for k in peak_levels:
            ds = RamanDataset(
                x=X, widths=WIDTHS, n_samples=n_reps, n_peaks=(k, k),
                seed=seed, **ds_kw,
            )
            for r in range(n_reps):
                sample = ds[r]
                spectrum_np = np.asarray(sample[8], dtype=np.float32)
                spectrum_t = torch.as_tensor(spectrum_np)

                params_c, p0, dt_c, n_iter = fit_cascade(spectrum_t)
                n_params = int((p0.reshape(-1, 4)[:, 0] > 0).sum().item()) * 4
                writer.writerow(dict(n_peaks_true=k, rep=r, n_params=n_params,
                                      method="CASCADE", time_s=dt_c, nfev="", lm_success=""))
                fh.flush()
                print(f"[k={k} rep={r}] CASCADE   n_params={n_params:4d} t={dt_c:6.2f}s", flush=True)

                p0_np = p0.cpu().numpy()
                nfev_budget = _scaled_max_nfev(n_params)
                t0 = time.perf_counter()
                lm_params, cost, success, nfev = lm_fit_from_p0(spectrum_np, X, p0_np, max_nfev=nfev_budget)
                dt_lm = time.perf_counter() - t0
                writer.writerow(dict(n_peaks_true=k, rep=r, n_params=lm_params.size,
                                      method="LM+CWT_p0", time_s=dt_lm, nfev=nfev, lm_success=success))
                fh.flush()
                print(f"[k={k} rep={r}] LM+CWT_p0 n_params={lm_params.size:4d} t={dt_lm:6.2f}s "
                      f"nfev={nfev} success={success}", flush=True)

    print(f"\nExperiment 2 complete -> {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    setup_context()
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("1", "both"):
        run_experiment_1()
    if which in ("2", "both"):
        run_experiment_2()
