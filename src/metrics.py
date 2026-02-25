"""
Metrics module for comparing model UV LFs with observational data.
"""

import numpy as np
from scipy.interpolate import interp1d


def chi_squared(model_phi, obs_phi, obs_phi_err):
    """
    Compute chi-squared between model and observations.
    Uses log-space comparison for better dynamic range handling.

    Parameters
    ----------
    model_phi : array
        Model phi values (Mpc^-3 mag^-1)
    obs_phi : array
        Observed phi values
    obs_phi_err : array
        Observed phi uncertainties

    Returns
    -------
    chi2 : float
        Chi-squared value
    n_dof : int
        Number of degrees of freedom (data points)
    """
    # Work in log space
    valid = (model_phi > 0) & (obs_phi > 0) & (obs_phi_err > 0)
    if np.sum(valid) == 0:
        return np.inf, 0

    log_model = np.log10(model_phi[valid])
    log_obs = np.log10(obs_phi[valid])
    # Error propagation: sigma_log = sigma / (phi * ln(10))
    log_err = obs_phi_err[valid] / (obs_phi[valid] * np.log(10))
    log_err = np.maximum(log_err, 0.1)  # floor on log error

    chi2 = np.sum((log_model - log_obs) ** 2 / log_err**2)
    n_dof = np.sum(valid)

    return float(chi2), int(n_dof)


def interpolate_model_to_obs(model_M_UV, model_phi, obs_M_UV):
    """
    Interpolate model UV LF to observational magnitude bins.
    """
    # Only interpolate where model has nonzero values
    valid = model_phi > 0
    if np.sum(valid) < 2:
        return np.zeros_like(obs_M_UV)

    # Log-space interpolation
    log_phi_interp = interp1d(
        model_M_UV[valid],
        np.log10(model_phi[valid]),
        kind="linear",
        fill_value=-30,
        bounds_error=False,
    )

    model_phi_at_obs = 10 ** log_phi_interp(obs_M_UV)
    return model_phi_at_obs


def compute_all_metrics(model_results, obs_data):
    """
    Compute comprehensive metrics comparing model to observations.

    Parameters
    ----------
    model_results : dict
        {z: (M_UV, phi)} from model
    obs_data : dict
        {z: {'M_UV': array, 'phi': array, 'phi_err': array}}

    Returns
    -------
    metrics : dict
        Dictionary of all metrics
    """
    metrics = {}

    total_chi2 = 0
    total_ndof = 0

    # Chi-squared at each redshift
    for z in sorted(obs_data.keys()):
        if z not in model_results:
            # Find nearest model redshift
            model_zs = np.array(list(model_results.keys()))
            closest_z = model_zs[np.argmin(np.abs(model_zs - z))]
            model_M_UV, model_phi = model_results[closest_z]
        else:
            model_M_UV, model_phi = model_results[z]

        obs_M_UV = obs_data[z]["M_UV"]
        obs_phi = obs_data[z]["phi"]
        obs_phi_err = obs_data[z]["phi_err"]

        # Interpolate model to observational bins
        model_phi_interp = interpolate_model_to_obs(model_M_UV, model_phi, obs_M_UV)

        chi2, ndof = chi_squared(model_phi_interp, obs_phi, obs_phi_err)

        metrics[f"chi2_z{z}"] = round(chi2, 4)
        metrics[f"ndof_z{z}"] = ndof
        metrics[f"chi2_reduced_z{z}"] = round(chi2 / max(ndof, 1), 4)

        total_chi2 += chi2
        total_ndof += ndof

        # Bright-end excess diagnostic (M_UV < -20)
        bright = obs_M_UV < -20
        if np.any(bright) and np.any(model_phi_interp[bright] > 0):
            ratio_bright = np.mean(
                np.log10(obs_phi[bright] / np.maximum(model_phi_interp[bright], 1e-30))
            )
            metrics[f"log_excess_bright_z{z}"] = round(ratio_bright, 4)

    # Total metrics
    metrics["chi2_total"] = round(total_chi2, 4)
    metrics["ndof_total"] = total_ndof
    metrics["chi2_reduced_total"] = round(total_chi2 / max(total_ndof, 1), 4)

    # High-z specific metrics (z >= 10)
    chi2_highz = sum(metrics.get(f"chi2_z{z}", 0) for z in [10, 12, 14, 16])
    ndof_highz = sum(metrics.get(f"ndof_z{z}", 0) for z in [10, 12, 14, 16])
    metrics["chi2_highz_total"] = round(chi2_highz, 4)
    metrics["ndof_highz_total"] = ndof_highz
    metrics["chi2_reduced_highz"] = round(chi2_highz / max(ndof_highz, 1), 4)

    # Low-z metrics (z <= 8) for consistency check
    chi2_lowz = sum(metrics.get(f"chi2_z{z}", 0) for z in [4, 6, 8])
    ndof_lowz = sum(metrics.get(f"ndof_z{z}", 0) for z in [4, 6, 8])
    metrics["chi2_lowz_total"] = round(chi2_lowz, 4)
    metrics["ndof_lowz_total"] = ndof_lowz
    metrics["chi2_reduced_lowz"] = round(chi2_lowz / max(ndof_lowz, 1), 4)

    return metrics
