"""
Cosmology utilities for the SAM project.
Provides halo mass function, accretion rates, and cosmological calculations.
Uses colossus for consistency with standard cosmological calculations.
"""

import numpy as np
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d
from scipy.integrate import quad

# Set Planck 2018 cosmology as default
cosmo = colossus_cosmology.setCosmology("planck18")

# Cosmological parameters
H0 = cosmo.H0  # km/s/Mpc
Om0 = cosmo.Om0
Ob0 = cosmo.Ob0
h = cosmo.H0 / 100.0
fb = Ob0 / Om0  # baryon fraction ~ 0.157


def hubble_parameter(z):
    """Hubble parameter H(z) in km/s/Mpc."""
    return cosmo.Hz(z)


def age_of_universe(z):
    """Age of the universe at redshift z in Gyr."""
    return cosmo.age(z)


def cosmic_time_difference(z1, z2):
    """Time difference between redshifts z1 and z2 in Gyr. z1 > z2."""
    return cosmo.age(z2) - cosmo.age(z1)


def critical_density(z):
    """Critical density at redshift z in Msun/Mpc^3."""
    # rho_crit = 3 H^2 / (8 pi G)
    return cosmo.rho_c(z) * 1e9  # colossus returns in Msun h^2 / kpc^3, convert


def halo_mass_function(M_halo, z, model="tinker08"):
    """
    Halo mass function dn/dlog10(M) in units of h^3 Mpc^-3.

    Parameters
    ----------
    M_halo : array-like
        Halo masses in Msun/h
    z : float
        Redshift
    model : str
        Mass function model (default: Tinker et al. 2008)

    Returns
    -------
    dndlog10M : array
        Number density per dlog10(M) in h^3/Mpc^3
    """
    # colossus mass_function.massFunction returns dn/dlnM in h^3/Mpc^3
    # we want dn/dlog10M = dn/dlnM * ln(10)
    mfunc = mass_function.massFunction(
        M_halo, z, mdef="200m", model=model, q_out="dndlnM"
    )
    return mfunc * np.log(10)


def halo_accretion_rate(M_halo, z):
    """
    Mean halo mass accretion rate dM/dt in Msun/yr.

    Uses the Fakhouri, Ma & Boylan-Kolchin (2010) fitting formula:
    <dM/dt> = 46.1 (M/1e12)^1.1 (1+1.11*z) * sqrt(Om0*(1+z)^3 + OmL) Msun/yr

    Parameters
    ----------
    M_halo : array-like
        Halo mass in Msun (not Msun/h)
    z : float
        Redshift

    Returns
    -------
    dMdt : array
        Accretion rate in Msun/yr
    """
    OmL = 1.0 - Om0
    Ez = np.sqrt(Om0 * (1 + z) ** 3 + OmL)
    dMdt = 46.1 * (M_halo / 1e12) ** 1.1 * (1 + 1.11 * z) * Ez  # Msun/yr
    return dMdt


def virial_temperature(M_halo, z):
    """
    Virial temperature of a halo in Kelvin.

    T_vir ~ 1e4 * (M_halo / 1e8 Msun)^(2/3) * ((1+z)/10)

    More precisely using the virial relation.
    """
    mu = 0.59  # mean molecular weight for ionized primordial gas
    mp = 1.6726e-24  # proton mass in g
    kB = 1.3807e-16  # Boltzmann in erg/K
    G = 6.674e-8  # gravitational constant in cgs
    Msun = 1.989e33  # solar mass in g
    Mpc = 3.086e24  # Mpc in cm

    # Virial radius and velocity
    # Delta_c ~ 200 for simplicity
    Delta_c = 200.0
    rho_crit_cgs = 3 * (hubble_parameter(z) * 1e5 / Mpc) ** 2 / (8 * np.pi * G)

    R_vir = (3 * M_halo * Msun / (4 * np.pi * Delta_c * rho_crit_cgs)) ** (1.0 / 3.0)
    V_vir = np.sqrt(G * M_halo * Msun / R_vir)

    T_vir = 0.5 * mu * mp * V_vir**2 / kB
    return T_vir


def cooling_function_primordial(T):
    """
    Simplified cooling function for primordial gas (H, He) in erg cm^3 s^-1.
    Approximation from Sutherland & Dopita (1993) for zero metallicity.
    """
    logT = np.log10(np.clip(T, 1e3, 1e9))
    # Piecewise power-law approximation
    Lambda = np.where(
        logT < 4.0,
        1e-30,  # negligible below 1e4 K
        np.where(
            logT < 5.5,
            10 ** (-21.0 - 0.5 * (logT - 4.5) ** 2),  # H cooling peak
            np.where(
                logT < 7.0,
                10 ** (-22.5 + 0.5 * (logT - 5.5)),  # bremsstrahlung rise
                10 ** (-23.0 + 0.4 * (logT - 7.0)),  # bremsstrahlung
            ),
        ),
    )
    return Lambda


def redshift_array(z_min=4.0, z_max=16.0, n_steps=50):
    """Generate an array of redshifts for computation."""
    return np.linspace(z_min, z_max, n_steps)


def halo_mass_array(log_Mmin=8.0, log_Mmax=14.0, n_mass=100):
    """Generate an array of halo masses in Msun/h."""
    return 10 ** np.linspace(log_Mmin, log_Mmax, n_mass)
