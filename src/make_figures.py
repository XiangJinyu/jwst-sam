"""
Generate publication-quality figures for the paper.

Figure 1: UV LF comparison (baseline vs proposed vs observations) at multiple z
Figure 2: SFE as a function of halo mass at different redshifts
Figure 3: Ablation study summary
Figure 4: Sensitivity analysis
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from sam_model import SemiAnalyticModel, SAMConfig, load_results
from observational_data import get_all_observations

# Style settings
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "baseline": "#2196F3",
    "proposed": "#E91E63",
    "obs": "#333333",
    "zdep_only": "#FF9800",
    "fb_only": "#4CAF50",
}

fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(fig_dir, exist_ok=True)


def figure1_uvlf_comparison():
    """Figure 1: UV LF at z=4,6,8,10,12,14 comparing baseline, proposed, and data."""
    print("Generating Figure 1: UV LF comparison...")

    # Load results
    _, baseline = load_results(
        os.path.join(results_dir, "baseline_calibrated_results.json")
    )
    _, proposed = load_results(os.path.join(results_dir, "proposed_results.json"))
    obs = get_all_observations()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True, sharex=True)
    axes = axes.flatten()

    redshifts = [4, 6, 8, 10, 12, 14]

    for i, z in enumerate(redshifts):
        ax = axes[i]

        # Baseline model
        if z in baseline:
            M_UV_b, phi_b = baseline[z]
            valid_b = phi_b > 0
            if np.any(valid_b):
                ax.plot(
                    M_UV_b[valid_b],
                    phi_b[valid_b],
                    "-",
                    color=COLORS["baseline"],
                    linewidth=2,
                    label="Baseline (const. SFE)",
                    zorder=2,
                )

        # Proposed model
        if z in proposed:
            M_UV_p, phi_p = proposed[z]
            valid_p = phi_p > 0
            if np.any(valid_p):
                ax.plot(
                    M_UV_p[valid_p],
                    phi_p[valid_p],
                    "-",
                    color=COLORS["proposed"],
                    linewidth=2.5,
                    label="Proposed (z-dep. SFE)",
                    zorder=3,
                )

        # Observations
        if z in obs:
            ax.errorbar(
                obs[z]["M_UV"],
                obs[z]["phi"],
                yerr=obs[z]["phi_err"],
                fmt="o",
                color=COLORS["obs"],
                markersize=5,
                capsize=3,
                label=f"Obs ({obs[z].get('source', '')})",
                zorder=4,
            )

        ax.set_yscale("log")
        ax.set_xlim(-24, -17)
        ax.set_ylim(1e-7, 1e-1)
        ax.set_title(f"$z = {z}$", fontweight="bold")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        if i >= 3:
            ax.set_xlabel(r"$M_{\rm UV}$ [mag]")
        if i % 3 == 0:
            ax.set_ylabel(r"$\phi$ [Mpc$^{-3}$ mag$^{-1}$]")

        if i == 0:
            ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        "UV Luminosity Functions: Baseline vs. Proposed SAM",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_uvlf_comparison.png"))
    plt.savefig(os.path.join(fig_dir, "fig1_uvlf_comparison.pdf"))
    plt.close()
    print("  Saved fig1_uvlf_comparison.png/pdf")


def figure2_sfe_evolution():
    """Figure 2: Star formation efficiency vs halo mass at different z."""
    print("Generating Figure 2: SFE evolution...")

    # Baseline params
    base_config = SAMConfig(
        epsilon_0=0.0508,
        V_SN=87.5,
        A_UV_0=0.06,
        use_zdep_sfe=False,
        use_feedback_weakening=False,
    )
    baseline_model = SemiAnalyticModel(base_config)

    # Proposed params
    prop_config = SAMConfig(
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
    proposed_model = SemiAnalyticModel(prop_config)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    M_halo = np.logspace(8, 14, 200)
    redshifts = [4, 6, 8, 10, 12, 14]
    cmap = plt.cm.plasma
    colors_z = [cmap(i / (len(redshifts) - 1)) for i in range(len(redshifts))]

    # Panel A: SFE vs M_halo for both models
    ax = axes[0]
    for iz, z in enumerate(redshifts):
        sfe_base = baseline_model.star_formation_efficiency(M_halo, z)
        sfe_prop = proposed_model.star_formation_efficiency(M_halo, z)

        ax.plot(M_halo, sfe_base, "--", color=colors_z[iz], alpha=0.5, linewidth=1)
        ax.plot(M_halo, sfe_prop, "-", color=colors_z[iz], linewidth=2, label=f"z={z}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm halo}$ [$M_\odot$]")
    ax.set_ylabel(r"$\epsilon_*$ (Star Formation Efficiency)")
    ax.set_title("SFE: Dashed=Baseline, Solid=Proposed")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(1e-5, 1)
    ax.grid(True, alpha=0.3)

    # Panel B: SFE enhancement ratio vs z for a fixed halo mass
    ax = axes[1]
    z_arr = np.linspace(4, 16, 50)
    M_test = [1e10, 1e11, 1e12]
    linestyles = ["-", "--", ":"]

    for im, M in enumerate(M_test):
        ratio = []
        for z in z_arr:
            sfe_base = float(
                baseline_model.star_formation_efficiency(np.array([M]), z)[0]
            )
            sfe_prop = float(
                proposed_model.star_formation_efficiency(np.array([M]), z)[0]
            )
            ratio.append(sfe_prop / max(sfe_base, 1e-10))
        ax.plot(
            z_arr,
            ratio,
            linestyles[im],
            color=COLORS["proposed"],
            linewidth=2,
            label=f"$M_h = 10^{{{int(np.log10(M))}}} M_\\odot$",
        )

    ax.axhline(y=1, color="gray", linestyle="-", alpha=0.5)
    ax.axvline(x=7.74, color="gray", linestyle=":", alpha=0.5, label="$z_{\\rm pivot}$")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"SFE Enhancement Ratio ($\epsilon_{\rm prop}/\epsilon_{\rm base}$)")
    ax.set_title("SFE Enhancement vs. Redshift")
    ax.legend(fontsize=9)
    ax.set_xlim(4, 16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_sfe_evolution.png"))
    plt.savefig(os.path.join(fig_dir, "fig2_sfe_evolution.pdf"))
    plt.close()
    print("  Saved fig2_sfe_evolution.png/pdf")


def figure3_ablation():
    """Figure 3: Ablation study bar chart."""
    print("Generating Figure 3: Ablation study...")

    with open(os.path.join(results_dir, "ablation_RESULTS.json"), "r") as f:
        ablation = json.load(f)

    variants = ["baseline", "zdep_sfe_only", "fb_weakening_only", "full_proposed"]
    labels = [
        "Baseline\n(const. SFE)",
        "z-dep SFE\nonly",
        "Feedback\nweakening only",
        "Full\nproposed",
    ]
    colors_bar = [
        COLORS["baseline"],
        COLORS["zdep_only"],
        COLORS["fb_only"],
        COLORS["proposed"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics_to_plot = [
        ("chi2_reduced_lowz", "$\\chi^2_{\\rm red}$ (z=4-8)", "Low-z Fit Quality"),
        ("chi2_reduced_highz", "$\\chi^2_{\\rm red}$ (z=10-16)", "High-z Fit Quality"),
        ("chi2_reduced_total", "$\\chi^2_{\\rm red}$ (all z)", "Overall Fit Quality"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics_to_plot):
        values = [ablation[v][metric] for v in variants]
        # Cap for display
        values_display = [min(v, 100) for v in values]

        bars = ax.bar(
            labels, values_display, color=colors_bar, edgecolor="black", linewidth=0.5
        )

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 100:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    fontsize=8,
                )

        ax.axhline(
            y=1,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="$\\chi^2_{\\rm red}=1$",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Ablation Study: Contribution of Each Model Component",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_ablation.png"))
    plt.savefig(os.path.join(fig_dir, "fig3_ablation.pdf"))
    plt.close()
    print("  Saved fig3_ablation.png/pdf")


def figure4_sensitivity():
    """Figure 4: Sensitivity analysis plots."""
    print("Generating Figure 4: Sensitivity analysis...")

    obs = get_all_observations()
    base_eps0, base_VSN, base_AUV = 0.0508, 87.5, 0.06

    from metrics import compute_all_metrics

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: sfe_slope sensitivity
    ax = axes[0]
    slopes = np.linspace(0.1, 1.0, 20)
    chi2_highz_list = []
    chi2_lowz_list = []
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
            n_mass=100,
        )
        model = SemiAnalyticModel(config)
        results = {}
        for z in [4, 6, 8, 10, 12, 14]:
            M_UV, phi = model.uv_luminosity_function(z)
            results[z] = (M_UV, phi)
        metrics = compute_all_metrics(results, obs)
        chi2_highz_list.append(metrics["chi2_reduced_highz"])
        chi2_lowz_list.append(metrics["chi2_reduced_lowz"])

    ax.plot(
        slopes,
        chi2_highz_list,
        "-",
        color=COLORS["proposed"],
        linewidth=2,
        label="$z\\geq10$",
    )
    ax.plot(
        slopes,
        chi2_lowz_list,
        "--",
        color=COLORS["baseline"],
        linewidth=2,
        label="$z\\leq8$",
    )
    ax.axhline(y=1, color="green", linestyle=":", alpha=0.5)
    ax.axvline(x=0.785, color="gray", linestyle=":", alpha=0.5, label="Best fit")
    ax.set_xlabel(r"$\beta_{\rm SFE}$ (SFE slope)")
    ax.set_ylabel(r"$\chi^2_{\rm red}$")
    ax.set_title(r"Sensitivity to $\beta_{\rm SFE}$")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3)

    # Panel B: z_pivot sensitivity
    ax = axes[1]
    pivots = np.linspace(5.0, 11.0, 20)
    chi2_highz_list = []
    chi2_lowz_list = []
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
            n_mass=100,
        )
        model = SemiAnalyticModel(config)
        results = {}
        for z in [4, 6, 8, 10, 12, 14]:
            M_UV, phi = model.uv_luminosity_function(z)
            results[z] = (M_UV, phi)
        metrics = compute_all_metrics(results, obs)
        chi2_highz_list.append(metrics["chi2_reduced_highz"])
        chi2_lowz_list.append(metrics["chi2_reduced_lowz"])

    ax.plot(
        pivots,
        chi2_highz_list,
        "-",
        color=COLORS["proposed"],
        linewidth=2,
        label="$z\\geq10$",
    )
    ax.plot(
        pivots,
        chi2_lowz_list,
        "--",
        color=COLORS["baseline"],
        linewidth=2,
        label="$z\\leq8$",
    )
    ax.axhline(y=1, color="green", linestyle=":", alpha=0.5)
    ax.axvline(x=7.74, color="gray", linestyle=":", alpha=0.5, label="Best fit")
    ax.set_xlabel(r"$z_{\rm pivot}$")
    ax.set_ylabel(r"$\chi^2_{\rm red}$")
    ax.set_title(r"Sensitivity to $z_{\rm pivot}$")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 30)
    ax.grid(True, alpha=0.3)

    # Panel C: fb_weak_factor sensitivity
    ax = axes[2]
    fb_factors = np.linspace(0.05, 0.8, 20)
    chi2_highz_list = []
    chi2_lowz_list = []
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
            n_mass=100,
        )
        model = SemiAnalyticModel(config)
        results = {}
        for z in [4, 6, 8, 10, 12, 14]:
            M_UV, phi = model.uv_luminosity_function(z)
            results[z] = (M_UV, phi)
        metrics = compute_all_metrics(results, obs)
        chi2_highz_list.append(metrics["chi2_reduced_highz"])
        chi2_lowz_list.append(metrics["chi2_reduced_lowz"])

    ax.plot(
        fb_factors,
        chi2_highz_list,
        "-",
        color=COLORS["proposed"],
        linewidth=2,
        label="$z\\geq10$",
    )
    ax.plot(
        fb_factors,
        chi2_lowz_list,
        "--",
        color=COLORS["baseline"],
        linewidth=2,
        label="$z\\leq8$",
    )
    ax.axhline(y=1, color="green", linestyle=":", alpha=0.5)
    ax.axvline(x=0.137, color="gray", linestyle=":", alpha=0.5, label="Best fit")
    ax.set_xlabel(r"$f_{\rm fb,weak}$")
    ax.set_ylabel(r"$\chi^2_{\rm red}$")
    ax.set_title(r"Sensitivity to $f_{\rm fb,weak}$")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_sensitivity.png"))
    plt.savefig(os.path.join(fig_dir, "fig4_sensitivity.pdf"))
    plt.close()
    print("  Saved fig4_sensitivity.png/pdf")


if __name__ == "__main__":
    figure1_uvlf_comparison()
    figure2_sfe_evolution()
    figure3_ablation()
    figure4_sensitivity()
    print("\nAll figures generated successfully!")
