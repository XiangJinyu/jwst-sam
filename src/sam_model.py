"""
Semi-Analytic Model (SAM) for Galaxy Formation.

This module implements a lightweight SAM that connects dark matter halos
to galaxy UV luminosities through:
1. Halo mass function (from colossus)
2. Baryon accretion and gas cooling
3. Star formation (with configurable SFE prescriptions)
4. Stellar/SN feedback
5. UV luminosity calculation

Two SFE prescriptions are implemented:
- BASELINE: Standard constant SFE (epsilon_* ~ 0.01-0.03)
- PROPOSED: Redshift-dependent SFE with metallicity-modulated feedback
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import json
import os

from cosmology_utils import (
    fb,
    halo_mass_function,
    halo_accretion_rate,
    virial_temperature,
    hubble_parameter,
    age_of_universe,
    cosmic_time_difference,
    halo_mass_array,
    h,
    Om0,
)


class SAMConfig:
    """Configuration for the SAM model."""

    def __init__(self, **kwargs):
        # Star formation parameters
        self.epsilon_0 = kwargs.get("epsilon_0", 0.02)  # base SFE
        self.t_star = kwargs.get("t_star", 1.0)  # SF timescale multiplier

        # Feedback parameters
        self.V_SN = kwargs.get("V_SN", 120.0)  # SN feedback velocity scale [km/s]
        self.alpha_SN = kwargs.get("alpha_SN", 2.0)  # SN feedback mass-loading slope
        self.epsilon_SN = kwargs.get("epsilon_SN", 3.0)  # SN feedback efficiency

        # AGN feedback (only relevant for massive halos)
        self.f_AGN = kwargs.get("f_AGN", 0.01)  # AGN feedback efficiency
        self.M_AGN_crit = kwargs.get(
            "M_AGN_crit", 1e12
        )  # critical halo mass for AGN [Msun]

        # UV conversion
        self.kappa_UV = kwargs.get(
            "kappa_UV", 1.15e28
        )  # L_UV per SFR [erg/s/Hz per Msun/yr]
        # Kennicutt & Evans (2012): SFR=1 Msun/yr -> L_UV ~ 1.15e28 erg/s/Hz at 1500A

        # Dust attenuation (simple IRX-beta relation)
        self.A_UV_0 = kwargs.get("A_UV_0", 1.0)  # dust attenuation at z=0 [mag]
        self.dust_z_evol = kwargs.get(
            "dust_z_evol", True
        )  # enable dust evolution with z

        # Halo mass range
        self.log_Mmin = kwargs.get("log_Mmin", 8.0)
        self.log_Mmax = kwargs.get("log_Mmax", 14.0)
        self.n_mass = kwargs.get("n_mass", 200)

        # Redshift-dependent SFE parameters (PROPOSED model only)
        self.use_zdep_sfe = kwargs.get("use_zdep_sfe", False)
        self.z_pivot = kwargs.get("z_pivot", 8.0)  # pivot redshift
        self.sfe_slope = kwargs.get(
            "sfe_slope", 0.15
        )  # SFE boost per unit z above z_pivot
        self.sfe_max = kwargs.get("sfe_max", 0.3)  # maximum SFE cap

        # Feedback weakening parameters (PROPOSED model)
        self.use_feedback_weakening = kwargs.get("use_feedback_weakening", False)
        self.Z_crit = kwargs.get(
            "Z_crit", 0.1
        )  # critical metallicity (Z_sun) for feedback transition
        self.fb_weak_factor = kwargs.get(
            "fb_weak_factor", 0.3
        )  # feedback reduction factor at low Z

        # Random seed
        self.seed = kwargs.get("seed", 42)

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def baseline(cls):
        """Standard baseline configuration."""
        return cls(epsilon_0=0.02, use_zdep_sfe=False, use_feedback_weakening=False)

    @classmethod
    def proposed(cls):
        """Proposed model with redshift-dependent SFE and feedback weakening."""
        return cls(
            epsilon_0=0.02,
            use_zdep_sfe=True,
            sfe_slope=0.15,
            sfe_max=0.3,
            z_pivot=8.0,
            use_feedback_weakening=True,
            Z_crit=0.1,
            fb_weak_factor=0.3,
        )


class SemiAnalyticModel:
    """
    Lightweight Semi-Analytic Galaxy Formation Model.

    Maps halo mass function -> galaxy UV luminosity function at each redshift.
    """

    def __init__(self, config=None):
        if config is None:
            config = SAMConfig.baseline()
        self.config = config
        np.random.seed(config.seed)

    def star_formation_efficiency(self, M_halo, z):
        """
        Compute effective star formation efficiency epsilon_*(M, z).

        BASELINE: epsilon_* = epsilon_0 * f_cool(M) * f_fb(M)
        PROPOSED: epsilon_* = epsilon_0 * g(z) * f_cool(M) * f_fb_mod(M, z)

        Parameters
        ----------
        M_halo : array-like
            Halo mass in Msun
        z : float
            Redshift

        Returns
        -------
        epsilon : array
            Effective SFE (dimensionless, 0 to 1)
        """
        M_halo = np.atleast_1d(M_halo).astype(float)

        # Cooling efficiency: suppressed below atomic cooling threshold
        T_vir = virial_temperature(M_halo, z)
        T_cool = 1e4  # atomic cooling threshold [K]
        f_cool = np.where(T_vir > T_cool, 1.0, (T_vir / T_cool) ** 3)

        # SN feedback: mass loading factor
        V_circ = np.sqrt(10 * 1.3807e-16 * T_vir / (0.59 * 1.6726e-24))  # cm/s
        V_circ_kms = V_circ / 1e5  # km/s

        eta_SN = self.config.epsilon_SN * (V_circ_kms / self.config.V_SN) ** (
            -self.config.alpha_SN
        )
        eta_SN = np.clip(eta_SN, 0, 100)  # cap mass loading

        # Feedback factor
        f_fb = 1.0 / (1.0 + eta_SN)

        # AGN feedback: suppress SF in massive halos
        f_AGN = np.where(
            M_halo > self.config.M_AGN_crit,
            np.exp(-self.config.f_AGN * (M_halo / self.config.M_AGN_crit - 1)),
            1.0,
        )

        # Base SFE
        epsilon = self.config.epsilon_0 * f_cool * f_fb * f_AGN

        # PROPOSED: Redshift-dependent enhancement
        if self.config.use_zdep_sfe:
            # SFE boost at high z
            z_boost = 1.0 + self.config.sfe_slope * np.maximum(
                z - self.config.z_pivot, 0
            )
            epsilon = epsilon * z_boost

        # PROPOSED: Feedback weakening at low metallicity (high z proxy)
        if self.config.use_feedback_weakening:
            # Approximate metallicity as function of z (higher z -> lower Z)
            # Z/Z_sun ~ 0.5 * (1 + z)^(-1.5) -- rough approximation
            Z_approx = 0.5 * (1 + z) ** (-1.5)

            # Weaken feedback below critical metallicity
            if Z_approx < self.config.Z_crit:
                fb_reduction = self.config.fb_weak_factor + (
                    1 - self.config.fb_weak_factor
                ) * (Z_approx / self.config.Z_crit)
                # Apply: less feedback -> more SF
                eta_SN_mod = eta_SN * fb_reduction
                f_fb_mod = 1.0 / (1.0 + eta_SN_mod)
                epsilon = self.config.epsilon_0 * f_cool * f_fb_mod * f_AGN
                if self.config.use_zdep_sfe:
                    z_boost = 1.0 + self.config.sfe_slope * np.maximum(
                        z - self.config.z_pivot, 0
                    )
                    epsilon = epsilon * z_boost

        # Cap SFE
        epsilon = np.clip(epsilon, 0, self.config.sfe_max)

        return epsilon

    def star_formation_rate(self, M_halo, z):
        """
        Compute star formation rate for a halo of mass M at redshift z.

        SFR = epsilon_* * fb * dM_halo/dt

        Parameters
        ----------
        M_halo : array-like
            Halo mass in Msun
        z : float
            Redshift

        Returns
        -------
        sfr : array
            Star formation rate in Msun/yr
        """
        epsilon = self.star_formation_efficiency(M_halo, z)
        dMdt = halo_accretion_rate(M_halo, z)
        sfr = epsilon * fb * dMdt
        return sfr

    def uv_magnitude(self, sfr, z):
        """
        Convert SFR to UV absolute magnitude M_UV.

        Using L_UV = kappa_UV * SFR and M_UV = -2.5 * log10(L_UV) + 51.6
        (AB magnitude system at 1500 Angstrom)

        Parameters
        ----------
        sfr : array-like
            Star formation rate in Msun/yr
        z : float
            Redshift

        Returns
        -------
        M_UV : array
            Absolute UV magnitude (AB)
        """
        sfr = np.atleast_1d(sfr).astype(float)

        # Luminosity at 1500 Angstrom
        L_UV = self.config.kappa_UV * sfr  # erg/s/Hz

        # Avoid log of zero
        L_UV = np.maximum(L_UV, 1e-50)

        # AB magnitude
        M_UV = -2.5 * np.log10(L_UV) + 51.6

        # Dust attenuation (simple model: decreases with z)
        if self.config.dust_z_evol:
            A_UV = self.config.A_UV_0 * np.exp(-0.15 * (z - 4))
            A_UV = np.maximum(A_UV, 0.0)
        else:
            A_UV = self.config.A_UV_0

        # Apply dust: make fainter (larger M_UV)
        # At high z, less dust -> less attenuation -> brighter galaxies
        M_UV = M_UV + A_UV

        return M_UV

    def uv_luminosity_function(self, z, M_UV_bins=None):
        """
        Compute the UV luminosity function phi(M_UV) at redshift z.

        Method: Abundance matching approach.
        phi(M_UV) dM_UV = n(M_halo) dM_halo
        -> phi(M_UV) = n(M_halo) * |dM_halo/dM_UV|

        Parameters
        ----------
        z : float
            Redshift
        M_UV_bins : array-like, optional
            UV magnitude bin centers. Default: -24 to -14

        Returns
        -------
        M_UV_centers : array
            UV magnitude bin centers
        phi : array
            Number density phi(M_UV) in Mpc^-3 mag^-1
        """
        if M_UV_bins is None:
            M_UV_bins = np.linspace(-24, -14, 40)

        # Halo mass array
        log_M = np.linspace(
            self.config.log_Mmin, self.config.log_Mmax, self.config.n_mass
        )
        M_halo = 10**log_M  # Msun
        M_halo_h = M_halo * h  # Msun/h for colossus

        # Halo mass function dn/dlog10M [h^3/Mpc^3]
        dndlog10M = halo_mass_function(M_halo_h, z)

        # Convert to dn/dM [h^3/Mpc^3 / (Msun/h)] then to [Mpc^-3 / Msun]
        # dn/dM = dn/dlog10M / (M * ln(10))
        # In physical units: multiply by h^3 to get Mpc^-3, divide by h for mass -> net h^2
        # Actually colossus output is in comoving h^3/Mpc^3
        dndlog10M_phys = dndlog10M * h**3  # Mpc^-3 per dlog10(M/h)
        # Since dlog10(M*h) = dlog10(M) for fixed h, this is fine

        # SFR for each halo
        sfr = self.star_formation_rate(M_halo, z)

        # UV magnitude for each halo
        M_UV = self.uv_magnitude(sfr, z)

        # Sort by M_UV (ascending = brightest first since more negative = brighter)
        sort_idx = np.argsort(M_UV)
        M_UV_sorted = M_UV[sort_idx]
        dndlog10M_sorted = dndlog10M_phys[sort_idx]
        log_M_sorted = log_M[sort_idx]

        # Bin the luminosity function
        dM_UV = M_UV_bins[1] - M_UV_bins[0]
        phi = np.zeros(len(M_UV_bins))

        for i, M_UV_center in enumerate(M_UV_bins):
            M_UV_lo = M_UV_center - dM_UV / 2
            M_UV_hi = M_UV_center + dM_UV / 2

            mask = (M_UV_sorted >= M_UV_lo) & (M_UV_sorted < M_UV_hi)
            if np.sum(mask) > 0:
                # Integrate the halo mass function contribution
                # phi(M_UV) dM_UV = sum of n(M_i) dlog10(M_i)
                dlog10M = np.median(np.diff(log_M))
                phi[i] = np.sum(dndlog10M_sorted[mask]) * dlog10M / dM_UV

        return M_UV_bins, phi

    def compute_uv_lf_all_redshifts(self, redshifts, M_UV_bins=None):
        """
        Compute UV LF at multiple redshifts.

        Returns
        -------
        results : dict
            {z: (M_UV_bins, phi)} for each redshift
        """
        results = {}
        for z in redshifts:
            M_UV_centers, phi = self.uv_luminosity_function(z, M_UV_bins)
            results[z] = (M_UV_centers, phi)
        return results


def save_results(results, config, filename):
    """Save results to JSON file."""
    output = {"config": config.to_dict(), "results": {}}
    for z, (M_UV, phi) in results.items():
        output["results"][str(z)] = {"M_UV": M_UV.tolist(), "phi": phi.tolist()}

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)


def load_results(filename):
    """Load results from JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)

    results = {}
    for z_str, vals in data["results"].items():
        z = float(z_str)
        results[z] = (np.array(vals["M_UV"]), np.array(vals["phi"]))

    return data["config"], results
