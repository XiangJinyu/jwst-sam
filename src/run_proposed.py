"""
Run the PROPOSED SAM model with redshift-dependent SFE and feedback weakening.
Calibrate additional parameters (sfe_slope, z_pivot, fb_weak_factor) using
z=10-14 JWST data while keeping the baseline z=4-8 calibration as constraint.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig, save_results
from observational_data import get_all_observations
from metrics import compute_all_metrics, interpolate_model_to_obs, chi_squared


def objective_proposed(params, obs_data, baseline_params, verbose=False):
    """
    Objective function for proposed model.

    Optimized parameters:
    params[0] = sfe_slope: SFE boost per unit redshift above z_pivot
    params[1] = z_pivot: pivot redshift for SFE enhancement
    params[2] = fb_weak_factor: feedback weakening at low Z
    params[3] = Z_crit: critical metallicity for feedback transition

    Fixed from baseline calibration:
    epsilon_0, V_SN, A_UV_0
    """
    sfe_slope, z_pivot, fb_weak_factor, Z_crit = params
    epsilon_0, V_SN, A_UV_0 = baseline_params

    # Bounds check
    if sfe_slope < 0.01 or sfe_slope > 1.0:
        return 1e10
    if z_pivot < 4 or z_pivot > 12:
        return 1e10
    if fb_weak_factor < 0.05 or fb_weak_factor > 0.95:
        return 1e10
    if Z_crit < 0.01 or Z_crit > 0.5:
        return 1e10

    config = SAMConfig(
        epsilon_0=epsilon_0,
        V_SN=V_SN,
        A_UV_0=A_UV_0,
        use_zdep_sfe=True,
        sfe_slope=sfe_slope,
        sfe_max=0.5,  # allow higher cap
        z_pivot=z_pivot,
        use_feedback_weakening=True,
        fb_weak_factor=fb_weak_factor,
        Z_crit=Z_crit,
        n_mass=150,
    )
    model = SemiAnalyticModel(config)

    M_UV_bins = np.linspace(-24, -14, 40)

    # Compute chi2 for ALL redshifts (both low-z and high-z)
    chi2_total = 0

    for z in [4, 6, 8, 10, 12, 14]:
        if z not in obs_data:
            continue

        M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)
        obs_M_UV = obs_data[z]["M_UV"]
        obs_phi = obs_data[z]["phi"]
        obs_phi_err = obs_data[z]["phi_err"]

        model_phi_interp = interpolate_model_to_obs(M_UV, phi, obs_M_UV)
        chi2, ndof = chi_squared(model_phi_interp, obs_phi, obs_phi_err)

        if np.isinf(chi2):
            return 1e10

        # Weight high-z more to prioritize fixing the tension
        weight = 2.0 if z >= 10 else 1.0
        chi2_total += weight * chi2

    if verbose:
        print(
            f"  slope={sfe_slope:.3f}, z_piv={z_pivot:.1f}, fb_w={fb_weak_factor:.3f}, Z_c={Z_crit:.3f} -> chi2={chi2_total:.2f}"
        )

    return chi2_total


def run_proposed():
    """Run proposed model with calibration."""
    print("=" * 60)
    print("Running PROPOSED SAM model (z-dependent SFE + feedback weakening)")
    print("=" * 60)

    obs = get_all_observations()

    # Baseline calibrated parameters (from calibrate_baseline.py)
    baseline_params = [0.0508, 87.5, 0.06]
    print(
        f"Baseline params: eps0={baseline_params[0]:.4f}, V_SN={baseline_params[1]:.1f}, A_UV={baseline_params[2]:.2f}"
    )

    # Optimize proposed model parameters
    print("\nOptimizing proposed model parameters...")
    print("Parameters: [sfe_slope, z_pivot, fb_weak_factor, Z_crit]")

    # Use differential evolution for global optimization
    bounds = [
        (0.05, 0.8),  # sfe_slope
        (5.0, 10.0),  # z_pivot
        (0.1, 0.8),  # fb_weak_factor
        (0.02, 0.3),  # Z_crit
    ]

    print("\nRunning differential evolution (global optimization)...")
    result = differential_evolution(
        objective_proposed,
        bounds,
        args=(obs, baseline_params, True),
        seed=42,
        maxiter=50,
        popsize=10,
        tol=0.5,
        mutation=(0.5, 1.5),
        recombination=0.7,
    )

    print(f"\n{'=' * 60}")
    print(f"BEST FIT PROPOSED PARAMETERS:")
    print(f"  sfe_slope = {result.x[0]:.4f}")
    print(f"  z_pivot = {result.x[1]:.2f}")
    print(f"  fb_weak_factor = {result.x[2]:.4f}")
    print(f"  Z_crit = {result.x[3]:.4f}")
    print(f"  weighted chi2 = {result.fun:.2f}")
    print(f"{'=' * 60}")

    # Run full model with best-fit parameters
    config = SAMConfig(
        epsilon_0=baseline_params[0],
        V_SN=baseline_params[1],
        A_UV_0=baseline_params[2],
        use_zdep_sfe=True,
        sfe_slope=result.x[0],
        sfe_max=0.5,
        z_pivot=result.x[1],
        use_feedback_weakening=True,
        fb_weak_factor=result.x[2],
        Z_crit=result.x[3],
    )
    model = SemiAnalyticModel(config)

    redshifts = [4, 6, 8, 10, 12, 14]
    M_UV_bins = np.linspace(-24, -14, 50)

    results = {}
    for z in redshifts:
        M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)
        results[z] = (M_UV, phi)
        bright_mask = M_UV < -20
        if np.any(phi[bright_mask] > 0):
            phi_bright = np.max(phi[bright_mask])
            print(f"  z={z:2d}: phi(M_UV<-20) max = {phi_bright:.2e} Mpc^-3 mag^-1")

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_results(results, config, os.path.join(results_dir, "proposed_results.json"))

    # Full metrics
    metrics = compute_all_metrics(results, obs)
    output = {
        "model": "proposed",
        "config": config.to_dict(),
        "metrics": metrics,
        "optimization_chi2": float(result.fun),
    }
    with open(os.path.join(results_dir, "proposed_RESULTS.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n--- PROPOSED MODEL RESULTS ---")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    return config, results, metrics


if __name__ == "__main__":
    run_proposed()
