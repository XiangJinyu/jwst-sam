"""
Run the baseline SAM model (constant SFE, standard feedback).
Compute UV luminosity functions at z=4,6,8,10,12,14 and compare to observations.
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig, save_results
from observational_data import get_all_observations
from metrics import compute_all_metrics


def run_baseline():
    """Run baseline model and save results."""
    print("=" * 60)
    print("Running BASELINE SAM model (constant SFE)")
    print("=" * 60)

    # Configuration
    config = SAMConfig.baseline()
    model = SemiAnalyticModel(config)

    print(f"Configuration: epsilon_0={config.epsilon_0}, V_SN={config.V_SN}")
    print(f"SFE model: CONSTANT (no z-dependence)")
    print(f"Feedback: STANDARD (no metallicity modulation)")

    # Target redshifts
    redshifts = [4, 6, 8, 10, 12, 14]
    M_UV_bins = np.linspace(-24, -14, 50)

    # Compute UV LFs
    print("\nComputing UV luminosity functions...")
    results = {}
    for z in redshifts:
        M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)
        results[z] = (M_UV, phi)

        # Print summary
        bright_mask = M_UV < -20
        if np.any(phi[bright_mask] > 0):
            phi_bright = np.max(phi[bright_mask])
            print(f"  z={z:2d}: phi(M_UV<-20) max = {phi_bright:.2e} Mpc^-3 mag^-1")
        else:
            print(f"  z={z:2d}: No galaxies brighter than M_UV=-20")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_results(results, config, os.path.join(results_dir, "baseline_results.json"))

    # Compute metrics
    print("\nComputing comparison metrics...")
    obs = get_all_observations()
    metrics = compute_all_metrics(results, obs)

    # Save metrics
    output = {"model": "baseline", "config": config.to_dict(), "metrics": metrics}
    with open(os.path.join(results_dir, "baseline_RESULTS.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n--- BASELINE RESULTS ---")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    print(f"\nResults saved to {results_dir}/baseline_results.json")
    print(f"Metrics saved to {results_dir}/baseline_RESULTS.json")

    return results, metrics


if __name__ == "__main__":
    run_baseline()
