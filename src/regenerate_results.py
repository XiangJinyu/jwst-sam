"""Regenerate all model UV LF results for plotting."""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig, save_results

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
redshifts = [4, 6, 8, 10, 12, 14]
M_UV_bins = np.linspace(-24, -14, 50)

# Calibrated baseline
print("Regenerating calibrated baseline results...")
config_base = SAMConfig(
    epsilon_0=0.0508,
    V_SN=87.5,
    A_UV_0=0.06,
    use_zdep_sfe=False,
    use_feedback_weakening=False,
)
model_base = SemiAnalyticModel(config_base)
results_base = {}
for z in redshifts:
    M_UV, phi = model_base.uv_luminosity_function(z, M_UV_bins)
    results_base[z] = (M_UV, phi)
save_results(
    results_base,
    config_base,
    os.path.join(results_dir, "baseline_calibrated_results.json"),
)
print("  Done.")

# Proposed model
print("Regenerating proposed model results...")
config_prop = SAMConfig(
    epsilon_0=0.0508,
    V_SN=87.5,
    A_UV_0=0.06,
    use_zdep_sfe=True,
    sfe_slope=0.7854,
    sfe_max=0.5,
    z_pivot=7.74,
    use_feedback_weakening=True,
    fb_weak_factor=0.1368,
    Z_crit=0.1301,
)
model_prop = SemiAnalyticModel(config_prop)
results_prop = {}
for z in redshifts:
    M_UV, phi = model_prop.uv_luminosity_function(z, M_UV_bins)
    results_prop[z] = (M_UV, phi)
save_results(
    results_prop, config_prop, os.path.join(results_dir, "proposed_results.json")
)
print("  Done.")
print("All results regenerated.")
