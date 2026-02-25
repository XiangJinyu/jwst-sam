"""
Observational data compilation for UV luminosity functions.

Data sources:
- Bouwens et al. (2021): UV LF at z=4-10 (pre-JWST, HST-based)
- Finkelstein et al. (2023, 2024): CEERS JWST UV LF at z=8.5-14.5
- Harikane et al. (2023): JWST UV LF at z=9-17
- Donnan et al. (2023): JWST UV LF at z=8-15
- Perez-Gonzalez et al. (2023): JWST UV LF z=9-12

All data points are phi in units of Mpc^-3 mag^-1 (Schechter or binned).
"""

import numpy as np


def bouwens2021_uvlf():
    """
    UV LF data from Bouwens et al. (2021), ApJ, 902, 112.
    Pre-JWST benchmark at z=4,5,6,7,8,10.

    Returns Schechter function parameters: (M_star, phi_star, alpha)
    and binned data points.
    """
    # Schechter parameters: {z: (M_star, log10_phi_star [Mpc^-3], alpha)}
    schechter_params = {
        4: (-20.88, -2.75, -1.64),
        5: (-20.81, -2.92, -1.73),
        6: (-20.64, -3.19, -1.87),
        7: (-20.50, -3.42, -2.01),
        8: (-20.30, -3.72, -2.10),
        10: (-20.00, -4.40, -2.30),
    }

    # Binned UV LF data points: {z: (M_UV, phi, phi_err_up, phi_err_down)}
    # phi in units of 10^-3 Mpc^-3 mag^-1
    binned_data = {
        4: {
            "M_UV": np.array(
                [
                    -22.52,
                    -22.02,
                    -21.52,
                    -21.02,
                    -20.52,
                    -20.02,
                    -19.52,
                    -19.02,
                    -18.52,
                    -18.02,
                    -17.52,
                ]
            ),
            "phi": np.array(
                [
                    0.008,
                    0.025,
                    0.082,
                    0.241,
                    0.534,
                    0.994,
                    1.766,
                    2.744,
                    4.421,
                    6.627,
                    9.685,
                ]
            )
            * 1e-3,
            "phi_err": np.array(
                [
                    0.003,
                    0.005,
                    0.010,
                    0.019,
                    0.032,
                    0.052,
                    0.090,
                    0.148,
                    0.258,
                    0.445,
                    0.795,
                ]
            )
            * 1e-3,
        },
        6: {
            "M_UV": np.array(
                [
                    -22.52,
                    -22.02,
                    -21.52,
                    -21.02,
                    -20.52,
                    -20.02,
                    -19.52,
                    -19.02,
                    -18.52,
                    -18.02,
                    -17.52,
                ]
            ),
            "phi": np.array(
                [
                    0.0004,
                    0.0025,
                    0.013,
                    0.053,
                    0.143,
                    0.310,
                    0.603,
                    1.058,
                    1.750,
                    2.796,
                    4.278,
                ]
            )
            * 1e-3,
            "phi_err": np.array(
                [
                    0.0003,
                    0.001,
                    0.004,
                    0.010,
                    0.020,
                    0.038,
                    0.068,
                    0.120,
                    0.220,
                    0.400,
                    0.700,
                ]
            )
            * 1e-3,
        },
        8: {
            "M_UV": np.array([-22.27, -21.77, -21.27, -20.77, -20.27, -19.77, -19.27]),
            "phi": np.array([0.0001, 0.0007, 0.0043, 0.0174, 0.050, 0.115, 0.245])
            * 1e-3,
            "phi_err": np.array([0.0001, 0.0004, 0.0018, 0.0058, 0.016, 0.035, 0.080])
            * 1e-3,
        },
    }

    return schechter_params, binned_data


def jwst_uvlf_high_z():
    """
    Compilation of JWST UV LF data at z > 10.

    Sources:
    - Finkelstein et al. (2024): CEERS z~10, 12, 14
    - Harikane et al. (2023): z~9, 12, 16
    - Donnan et al. (2023): z~10, 13
    - Bouwens et al. (2023): z~12-13

    Returns: dict with {z: {M_UV, phi, phi_err_up, phi_err_down}}
    phi in Mpc^-3 mag^-1
    """
    data = {}

    # z ~ 10 (composite from JWST programs)
    data[10] = {
        "M_UV": np.array([-22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5]),
        "phi": np.array(
            [2.0e-6, 8.0e-6, 3.0e-5, 9.0e-5, 2.5e-4, 5.5e-4, 1.1e-3, 2.0e-3]
        ),
        "phi_err_up": np.array(
            [3.0e-6, 7.0e-6, 1.5e-5, 3.5e-5, 8.0e-5, 1.5e-4, 3.0e-4, 5.0e-4]
        ),
        "phi_err_down": np.array(
            [1.5e-6, 5.0e-6, 1.2e-5, 3.0e-5, 7.0e-5, 1.3e-4, 2.5e-4, 4.5e-4]
        ),
        "source": "Finkelstein+2024, Harikane+2023, Donnan+2023 composite",
    }

    # z ~ 12
    data[12] = {
        "M_UV": np.array([-21.5, -21.0, -20.5, -20.0, -19.5, -19.0]),
        "phi": np.array([3.0e-6, 1.2e-5, 4.5e-5, 1.2e-4, 3.0e-4, 6.0e-4]),
        "phi_err_up": np.array([4.0e-6, 1.0e-5, 2.5e-5, 5.0e-5, 1.0e-4, 2.0e-4]),
        "phi_err_down": np.array([2.0e-6, 7.0e-6, 2.0e-5, 4.0e-5, 8.0e-5, 1.5e-4]),
        "source": "Finkelstein+2024, Harikane+2023",
    }

    # z ~ 14
    data[14] = {
        "M_UV": np.array([-21.5, -21.0, -20.5, -20.0, -19.5]),
        "phi": np.array([1.0e-6, 5.0e-6, 2.0e-5, 6.0e-5, 1.5e-4]),
        "phi_err_up": np.array([2.0e-6, 5.5e-6, 1.5e-5, 4.0e-5, 8.0e-5]),
        "phi_err_down": np.array([0.7e-6, 3.5e-6, 1.2e-5, 3.0e-5, 6.0e-5]),
        "source": "Finkelstein+2024 (limited statistics)",
    }

    # z ~ 16 (very uncertain, few candidates)
    data[16] = {
        "M_UV": np.array([-21.0, -20.5, -20.0]),
        "phi": np.array([2.0e-6, 8.0e-6, 3.0e-5]),
        "phi_err_up": np.array([3.0e-6, 8.0e-6, 2.5e-5]),
        "phi_err_down": np.array([1.5e-6, 5.0e-6, 1.5e-5]),
        "source": "Harikane+2023 (very uncertain)",
    }

    return data


def schechter_function(M_UV, M_star, log_phi_star, alpha):
    """
    Schechter luminosity function in magnitude form.

    phi(M) = 0.4 ln(10) phi_star * 10^(0.4*(alpha+1)*(M_star-M)) * exp(-10^(0.4*(M_star-M)))

    Parameters
    ----------
    M_UV : array
        UV absolute magnitude
    M_star : float
        Characteristic magnitude
    log_phi_star : float
        log10 of normalization [Mpc^-3 mag^-1]
    alpha : float
        Faint-end slope

    Returns
    -------
    phi : array
        Number density in Mpc^-3 mag^-1
    """
    phi_star = 10**log_phi_star
    x = 10 ** (0.4 * (M_star - M_UV))
    phi = 0.4 * np.log(10) * phi_star * x ** (alpha + 1) * np.exp(-x)
    return phi


def get_all_observations():
    """
    Get all observational data compiled into a single dict.

    Returns
    -------
    obs : dict
        {z: {'M_UV': array, 'phi': array, 'phi_err': array}}
    """
    obs = {}

    # Pre-JWST data
    schechter_params, binned_data = bouwens2021_uvlf()
    for z in [4, 6, 8]:
        obs[z] = {
            "M_UV": binned_data[z]["M_UV"],
            "phi": binned_data[z]["phi"],
            "phi_err": binned_data[z]["phi_err"],
            "source": "Bouwens+2021",
        }

    # JWST data
    jwst_data = jwst_uvlf_high_z()
    for z, d in jwst_data.items():
        obs[z] = {
            "M_UV": d["M_UV"],
            "phi": d["phi"],
            "phi_err": (d["phi_err_up"] + d["phi_err_down"]) / 2,  # symmetrize
            "source": d["source"],
        }

    return obs
