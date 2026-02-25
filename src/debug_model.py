"""Debug script to examine intermediate model values."""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig
from cosmology_utils import (
    halo_mass_function,
    halo_accretion_rate,
    virial_temperature,
    fb,
    h,
)

config = SAMConfig.baseline()
model = SemiAnalyticModel(config)

z = 6
print(f"=== Debugging at z={z} ===")
print(f"Baryon fraction fb = {fb:.4f}")
print(f"h = {h:.4f}")

# Test with a few halo masses
M_test = np.array([1e9, 1e10, 1e11, 1e12, 1e13])
print(f"\nHalo Mass [Msun] | T_vir [K] | dM/dt [Msun/yr] | SFE | SFR [Msun/yr] | M_UV")
print("-" * 100)

for M in M_test:
    T = virial_temperature(M, z)
    dMdt = halo_accretion_rate(M, z)
    sfe = float(model.star_formation_efficiency(np.array([M]), z)[0])
    sfr_val = float(model.star_formation_rate(np.array([M]), z)[0])
    M_UV_val = float(model.uv_magnitude(np.array([sfr_val]), z)[0])
    print(
        f"{M:.1e} | {float(T):.1e} | {float(dMdt):.2e} | {sfe:.4f} | {sfr_val:.4e} | {M_UV_val:.2f}"
    )

# Check halo mass function
print(f"\n=== Halo Mass Function at z={z} ===")
log_M = np.linspace(8, 14, 30)
M_halo_h = 10**log_M * h
dndlog10M = halo_mass_function(M_halo_h, z)
print(f"log10(M/Msun_h) | dn/dlog10M [h^3/Mpc^3]")
for i in range(0, len(log_M), 5):
    print(f"  {log_M[i] + np.log10(h):.2f} | {dndlog10M[i]:.2e}")

# Check full UV LF
print(f"\n=== UV LF at z={z} ===")
M_UV_bins = np.linspace(-24, -14, 50)
M_UV, phi = model.uv_luminosity_function(z, M_UV_bins)
nonzero = phi > 0
if np.any(nonzero):
    print(
        f"M_UV range with phi>0: {M_UV[nonzero].min():.1f} to {M_UV[nonzero].max():.1f}"
    )
    print(f"Max phi = {np.max(phi):.2e} at M_UV = {M_UV[np.argmax(phi)]:.1f}")
else:
    print("NO nonzero phi values!")

# Check what M_UV values we're getting
print(f"\n=== M_UV distribution for all halos ===")
log_M_full = np.linspace(8, 14, 200)
M_halo_full = 10**log_M_full
sfr_full = model.star_formation_rate(M_halo_full, z)
M_UV_full = model.uv_magnitude(sfr_full, z)
print(f"SFR range: {sfr_full.min():.2e} to {sfr_full.max():.2e} Msun/yr")
print(f"M_UV range: {M_UV_full.min():.1f} to {M_UV_full.max():.1f}")
