"""
Ablation study: test each component of the proposed model separately.

Variants:
1. Baseline (constant SFE, standard feedback)
2. Z-dep SFE only (no feedback weakening)
3. Feedback weakening only (no z-dep SFE)
4. Full proposed (both mechanisms)
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig, save_results
from observational_data import get_all_observations
from metrics import compute_all_metrics


def run_ablation():
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    obs = get_all_observations()

    # Baseline calibrated params
    base_eps0, base_VSN, base_AUV = 0.0508, 87.5, 0.06

    # Proposed best-fit params
    sfe_slope = 0.7854
    z_pivot = 7.74
    fb_weak_factor = 0.1368
    Z_crit = 0.1301

    variants = {
        "baseline": SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=False,
            use_feedback_weakening=False,
        ),
        "zdep_sfe_only": SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=True,
            sfe_slope=sfe_slope,
            sfe_max=0.5,
            z_pivot=z_pivot,
            use_feedback_weakening=False,
        ),
        "fb_weakening_only": SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=False,
            use_feedback_weakening=True,
            fb_weak_factor=fb_weak_factor,
            Z_crit=Z_crit,
        ),
        "full_proposed": SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=True,
            sfe_slope=sfe_slope,
            sfe_max=0.5,
            z_pivot=z_pivot,
            use_feedback_weakening=True,
            fb_weak_factor=fb_weak_factor,
            Z_crit=Z_crit,
        ),
    }

    redshifts = [4, 6, 8, 10, 12, 14]
    M_UV_bins = np.linspace(-24, -14, 50)

    all_results = {}
    all_metrics = {}

    for name, config in variants.items():
        print(f"\n--- Running variant: {name} ---")
        model = SemiAnalyticModel(config)

        results = {}
        for z in redshifts:
            M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)
            results[z] = (M_UV, phi)

        metrics = compute_all_metrics(results, obs)
        all_results[name] = results
        all_metrics[name] = metrics

        print(f"  chi2_reduced_lowz  = {metrics['chi2_reduced_lowz']:.4f}")
        print(f"  chi2_reduced_highz = {metrics['chi2_reduced_highz']:.4f}")
        print(f"  chi2_reduced_total = {metrics['chi2_reduced_total']:.4f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"ABLATION SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'Variant':<25} {'chi2_r(z<=8)':<15} {'chi2_r(z>=10)':<15} {'chi2_r(total)':<15}"
    )
    print(f"{'-' * 70}")
    for name in variants:
        m = all_metrics[name]
        print(
            f"{name:<25} {m['chi2_reduced_lowz']:<15.4f} {m['chi2_reduced_highz']:<15.4f} {m['chi2_reduced_total']:<15.4f}"
        )

    # Save ablation results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    ablation_output = {}
    for name in variants:
        ablation_output[name] = all_metrics[name]

    with open(os.path.join(results_dir, "ablation_RESULTS.json"), "w") as f:
        json.dump(ablation_output, f, indent=2, default=str)

    # Also save individual results for plotting
    for name, results in all_results.items():
        save_results(
            results,
            variants[name],
            os.path.join(results_dir, f"ablation_{name}_results.json"),
        )

    print(f"\nAblation results saved to {results_dir}/ablation_RESULTS.json")

    return all_results, all_metrics


# Sensitivity analysis
def run_sensitivity():
    """Test sensitivity to key parameters."""
    print(f"\n{'=' * 60}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'=' * 60}")

    obs = get_all_observations()
    base_eps0, base_VSN, base_AUV = 0.0508, 87.5, 0.06

    # Vary sfe_slope
    print("\n--- Varying sfe_slope ---")
    slopes = [0.3, 0.5, 0.7, 0.785, 0.9]
    for slope in slopes:
        config = SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=True,
            sfe_slope=slope,
            sfe_max=0.5,
            z_pivot=7.74,
            use_feedback_weakening=True,
            fb_weak_factor=0.1368,
            Z_crit=0.1301,
            n_mass=150,
        )
        model = SemiAnalyticModel(config)
        results = {}
        for z in [4, 6, 8, 10, 12, 14]:
            M_UV, phi = model.uv_luminosity_function(z)
            results[z] = (M_UV, phi)
        metrics = compute_all_metrics(results, obs)
        print(
            f"  sfe_slope={slope:.3f}: chi2_highz={metrics['chi2_reduced_highz']:.2f}, chi2_lowz={metrics['chi2_reduced_lowz']:.2f}"
        )

    # Vary z_pivot
    print("\n--- Varying z_pivot ---")
    pivots = [5.0, 6.0, 7.0, 7.74, 9.0, 10.0]
    for zpiv in pivots:
        config = SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=True,
            sfe_slope=0.7854,
            sfe_max=0.5,
            z_pivot=zpiv,
            use_feedback_weakening=True,
            fb_weak_factor=0.1368,
            Z_crit=0.1301,
            n_mass=150,
        )
        model = SemiAnalyticModel(config)
        results = {}
        for z in [4, 6, 8, 10, 12, 14]:
            M_UV, phi = model.uv_luminosity_function(z)
            results[z] = (M_UV, phi)
        metrics = compute_all_metrics(results, obs)
        print(
            f"  z_pivot={zpiv:.1f}: chi2_highz={metrics['chi2_reduced_highz']:.2f}, chi2_lowz={metrics['chi2_reduced_lowz']:.2f}"
        )

    # Vary fb_weak_factor
    print("\n--- Varying fb_weak_factor ---")
    fb_factors = [0.05, 0.1, 0.137, 0.2, 0.3, 0.5]
    for fbw in fb_factors:
        config = SAMConfig(
            epsilon_0=base_eps0,
            V_SN=base_VSN,
            A_UV_0=base_AUV,
            use_zdep_sfe=True,
            sfe_slope=0.7854,
            sfe_max=0.5,
            z_pivot=7.74,
            use_feedback_weakening=True,
            fb_weak_factor=fbw,
            Z_crit=0.1301,
            n_mass=150,
        )
        model = SemiAnalyticModel(config)
        results = {}
        for z in [4, 6, 8, 10, 12, 14]:
            M_UV, phi = model.uv_luminosity_function(z)
            results[z] = (M_UV, phi)
        metrics = compute_all_metrics(results, obs)
        print(
            f"  fb_weak={fbw:.3f}: chi2_highz={metrics['chi2_reduced_highz']:.2f}, chi2_lowz={metrics['chi2_reduced_lowz']:.2f}"
        )


if __name__ == "__main__":
    all_results, all_metrics = run_ablation()
    run_sensitivity()
