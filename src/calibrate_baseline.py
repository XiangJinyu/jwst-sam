"""
Calibrate the baseline model parameters to fit z=4-8 UV LF observations.
Uses scipy.optimize to minimize chi-squared at low redshifts.
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig, save_results
from observational_data import get_all_observations
from metrics import compute_all_metrics, interpolate_model_to_obs, chi_squared


def objective(params, obs_data, verbose=False):
    """
    Objective function: chi-squared at z=4,6,8.

    Parameters being optimized:
    params[0] = epsilon_0 (base SFE)
    params[1] = V_SN (SN feedback velocity)
    params[2] = A_UV_0 (dust attenuation)
    """
    epsilon_0, V_SN, A_UV_0 = params

    # Bounds check
    if epsilon_0 < 0.001 or epsilon_0 > 0.5:
        return 1e10
    if V_SN < 30 or V_SN > 500:
        return 1e10
    if A_UV_0 < 0 or A_UV_0 > 5:
        return 1e10

    config = SAMConfig(
        epsilon_0=epsilon_0,
        V_SN=V_SN,
        A_UV_0=A_UV_0,
        use_zdep_sfe=False,
        use_feedback_weakening=False,
        n_mass=150,  # reduce for speed during optimization
    )
    model = SemiAnalyticModel(config)

    M_UV_bins = np.linspace(-24, -14, 40)
    total_chi2 = 0

    for z in [4, 6, 8]:
        M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)

        obs_M_UV = obs_data[z]["M_UV"]
        obs_phi = obs_data[z]["phi"]
        obs_phi_err = obs_data[z]["phi_err"]

        model_phi_interp = interpolate_model_to_obs(M_UV, phi, obs_M_UV)
        chi2, ndof = chi_squared(model_phi_interp, obs_phi, obs_phi_err)

        if np.isinf(chi2):
            return 1e10
        total_chi2 += chi2

    if verbose:
        print(
            f"  eps0={epsilon_0:.4f}, V_SN={V_SN:.1f}, A_UV={A_UV_0:.2f} -> chi2={total_chi2:.2f}"
        )

    return total_chi2


def calibrate():
    """Run calibration."""
    print("=" * 60)
    print("Calibrating baseline model to z=4-8 observations")
    print("=" * 60)

    obs = get_all_observations()

    # Initial guess
    x0 = [0.05, 80.0, 0.5]

    print("\nStarting optimization...")
    print("Parameters: [epsilon_0, V_SN, A_UV_0]")
    print(f"Initial guess: {x0}")

    # Try multiple starting points
    best_result = None
    best_chi2 = np.inf

    starting_points = [
        [0.05, 80.0, 0.5],
        [0.10, 60.0, 0.3],
        [0.03, 100.0, 0.8],
        [0.15, 50.0, 0.2],
        [0.08, 70.0, 0.4],
    ]

    for i, x0 in enumerate(starting_points):
        print(f"\n--- Starting point {i + 1}: {x0} ---")
        try:
            result = minimize(
                objective,
                x0,
                args=(obs, True),
                method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-4, "fatol": 1.0},
            )

            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_result = result
                print(f"  -> New best: chi2={result.fun:.2f}, params={result.x}")
        except Exception as e:
            print(f"  -> Failed: {e}")

    if best_result is None:
        print("Calibration failed!")
        return

    print(f"\n{'=' * 60}")
    print(f"BEST FIT PARAMETERS:")
    print(f"  epsilon_0 = {best_result.x[0]:.4f}")
    print(f"  V_SN = {best_result.x[1]:.1f} km/s")
    print(f"  A_UV_0 = {best_result.x[2]:.2f} mag")
    print(f"  chi2 (z=4-8) = {best_result.fun:.2f}")
    print(f"{'=' * 60}")

    # Run full model with best-fit parameters
    config = SAMConfig(
        epsilon_0=best_result.x[0],
        V_SN=best_result.x[1],
        A_UV_0=best_result.x[2],
        use_zdep_sfe=False,
        use_feedback_weakening=False,
    )
    model = SemiAnalyticModel(config)

    redshifts = [4, 6, 8, 10, 12, 14]
    M_UV_bins = np.linspace(-24, -14, 50)

    results = {}
    for z in redshifts:
        M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)
        results[z] = (M_UV, phi)

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(
        results, config, os.path.join(results_dir, "baseline_calibrated_results.json")
    )

    # Full metrics
    metrics = compute_all_metrics(results, obs)
    output = {
        "model": "baseline_calibrated",
        "config": config.to_dict(),
        "metrics": metrics,
        "calibration_chi2": float(best_result.fun),
    }
    with open(os.path.join(results_dir, "baseline_calibrated_RESULTS.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n--- CALIBRATED BASELINE RESULTS ---")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    return config, results, metrics


if __name__ == "__main__":
    calibrate()
