#!/usr/bin/env python3
from __future__ import annotations

"""Reproduce all numerical figures for the anisotropic-collapse manuscript.

The script is self-contained and uses only NumPy, SciPy, and Matplotlib.
It separates five numerical roles:
  1. frozen scalar OU consistency checks;
  2. controlled localized nonlinear and adiabatic consistency tests;
  3. an exact coupled nonnormal rotating-mode Lyapunov test;
  4. a finite-history critical-state estimation experiment;
  5. a fixed-noise nonlinear stress test preserving the original mechanism.

Usage
-----
python simulation_code.py [output_directory]

When no output directory is supplied, figures are written to the current
directory. All random seeds and numerical parameters are fixed in this file.
"""

import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

DPI = 300
TITLE_FS = 8.6
LEGEND_FS = 7.6
TICK_FS = 7.7

C_GREEN = "#3a7d44"
C_PURP = "#6a4c93"
C_CTRL = "#777777"
C_TEAL = "#2a7a6f"
C_GREY = "#999999"
C_DARK = "#1a1a2e"
C_ORANGE = "#c44b25"
C_BLUE = "#33658a"
C_GRID = "#d7d7d7"
C_RED = "#a23b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.4,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "lines.linewidth": 1.3,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


def style_axes(ax, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color=C_GRID, lw=0.55, alpha=0.8, ls="--")
    ax.tick_params(labelsize=TICK_FS)


def panel_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=TITLE_FS, fontweight="bold", pad=4)


def rolling_var(a: np.ndarray, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not 1 <= w < len(a):
        raise ValueError("rolling_var expects a 1D array and 1 <= w < len(a)")
    cs = np.cumsum(np.insert(a, 0, 0.0))
    cs2 = np.cumsum(np.insert(a * a, 0, 0.0))
    sw = cs[w:] - cs[:-w]
    sw2 = cs2[w:] - cs2[:-w]
    m = sw / w
    return sw2 / w - m * m


# -----------------------------------------------------------------------------
# Shared scalar definitions
# -----------------------------------------------------------------------------

SIGMA_FIXED = 0.3
SE = 0.05


def fisher_scalar(x: np.ndarray, alpha: float, se: float = SE) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return alpha ** 2 * np.abs(x) ** (2 * alpha - 2) / se ** 2


# -----------------------------------------------------------------------------
# Figure 1: phase diagram and frozen OU check
# -----------------------------------------------------------------------------


def generate_fig1(out: Path) -> None:
    fig = plt.figure(figsize=(6.69, 4.72), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.0))
    axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

    ax = axes[0]
    delta = np.linspace(0, 1.45, 300)
    ax.fill_between(delta, 0, delta, color="#d9eef2", alpha=0.95)
    ax.fill_between(delta, delta, 2.15, color="#eadcf2", alpha=0.95)
    ax.axhline(0, color=C_CTRL, lw=2.0)
    ax.plot(delta, delta, color=C_DARK, lw=1.8, ls="--")
    ax.text(0.73, -0.088, "preserved channel: $\\kappa=0$", ha="center", va="center",
            fontsize=8.8, color=C_CTRL)
    for d, k, col, marker in [
        (1.0, 0.0, C_CTRL, "o"), (1.0, 0.5, C_TEAL, "o"), (1.0, 2.0, C_PURP, "o"),
        (0.5, 0.0, C_CTRL, "s"), (0.5, 0.25, C_TEAL, "s"), (0.5, 1.0, C_PURP, "s"),
    ]:
        ax.plot(d, k, marker=marker, ms=7.2, color=col, mec="white", mew=0.8)
    ax.text(1.04, 1.90, r"PF $\alpha=3$", color=C_PURP, fontsize=7.9)
    ax.text(0.56, 1.06, r"SN $\alpha=2$", color=C_PURP, fontsize=7.9)
    ax.text(1.04, 0.56, r"PF $\alpha=1.5$", color=C_TEAL, fontsize=7.9)
    ax.text(0.54, 0.30, r"SN $\alpha=1.25$", color=C_TEAL, fontsize=7.9)
    ax.text(1.04, 0.10, r"PF $\alpha=1$", color=C_CTRL, fontsize=7.9)
    ax.text(0.42, 0.10, r"SN $\alpha=1$", color=C_CTRL, fontsize=7.9, ha="right")
    ax.set_xlim(0, 1.45)
    ax.set_ylim(-0.14, 2.15)
    ax.set_xlabel(r"Restoring-rate exponent $\delta$")
    ax.set_ylabel(r"Fisher-decay exponent $\kappa$")
    panel_title(ax, "(a) Fixed-excitation / normalized phase diagram")
    style_axes(ax)

    def ou_variance_ensemble(a_values: np.ndarray, seed: int) -> np.ndarray:
        n_rep, ds, burn_steps, sample_steps = 24, 0.01, 3000, 12000
        d0 = SIGMA_FIXED ** 2
        rng = np.random.default_rng(seed)
        estimates = np.empty((n_rep, len(a_values)))
        for j, a_val in enumerate(a_values):
            y = np.zeros(n_rep)
            noise_scale = np.sqrt((d0 / a_val) * ds)
            for _ in range(burn_steps):
                y += -y * ds + noise_scale * rng.normal(size=n_rep)
            s1 = np.zeros(n_rep)
            s2 = np.zeros(n_rep)
            for _ in range(sample_steps):
                y += -y * ds + noise_scale * rng.normal(size=n_rep)
                s1 += y
                s2 += y * y
            mean = s1 / sample_steps
            estimates[:, j] = s2 / sample_steps - mean * mean
        return estimates

    def ou_panel(ax, kind: str, alphas: list[float], title: str) -> None:
        mu = np.logspace(-2, 0, 18)
        d0 = SIGMA_FIXED ** 2
        colors = [C_CTRL, C_TEAL, C_PURP]
        a = 2.0 * mu if kind == "pitchfork" else 2.0 * np.sqrt(mu)
        runs = ou_variance_ensemble(a, 9100 if kind == "pitchfork" else 9200)
        for alpha, col in zip(alphas, colors):
            fisher = alpha ** 2 * mu ** (alpha - 1) / SE ** 2
            power = d0 * fisher / (2.0 * a)
            fn = fisher / fisher[-1]
            pn = power / power[-1]
            est = fisher[None, :] * runs / power[-1]
            med = np.median(est, axis=0)
            lo, hi = np.percentile(est, [10, 90], axis=0)
            ax.plot(mu, fn, color=col, lw=1.7, alpha=0.72)
            ax.plot(mu, pn, color=col, lw=2.0, ls="--", label=rf"$\alpha={alpha:g}$")
            ax.fill_between(mu, lo, hi, color=col, alpha=0.12, linewidth=0)
            ax.plot(mu, med, "o", color=col, ms=2.8, alpha=0.78)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(mu.max(), mu.min())
        ax.set_xlabel(r"Distance to bifurcation $\mu$")
        ax.set_ylabel("Normalized strength or power")
        panel_title(ax, title)
        ax.legend(fontsize=7.4, framealpha=0.95, loc="best")
        style_axes(ax)

    ou_panel(axes[1], "pitchfork", [1.0, 1.5, 3.0], "(b) Pitchfork frozen OU")
    ou_panel(axes[2], "saddlenode", [1.0, 1.25, 2.0], "(c) Saddle-node frozen OU")
    fig.savefig(out / "fig1_critical_mode_phase_diagram.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: scalar geometry
# -----------------------------------------------------------------------------


def generate_fig2(out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.37, 5.05), constrained_layout=True)
    x = np.linspace(-2, 2, 400)
    V = -x ** 2 / 2 + x ** 4 / 4
    V = (V - V.min()) / (V.max() - V.min())
    g = np.sign(x) * np.abs(x) ** 3
    g /= np.max(np.abs(g))

    ax = axes[0]
    ax.axvspan(-0.5, 0.5, alpha=0.10, color=C_PURP)
    ax.axvline(0, color=C_PURP, lw=1, ls="--", alpha=0.4)
    ax.plot(x, V, color=C_DARK, lw=2.0, label=r"$V(x;\mu=1)$")
    ax.plot(x, g, color=C_PURP, lw=2.0, ls="--", label=r"$g_3(x)=x^3$")
    ax.plot([-1, 1], [0, 0], "o", color=C_GREEN, ms=6)
    ax.text(0, -0.75, "low-sensitivity\nregion", fontsize=8.2, color=C_PURP,
            ha="center", va="center", fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_PURP, alpha=0.85))
    ax.set_xlabel("$x$")
    ax.set_ylabel("Normalized value")
    panel_title(ax, "(a) Potential and observation map")
    ax.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    ax.set_ylim(-1.1, 1.1)
    style_axes(ax)

    ax = axes[1]
    for alpha, col, ls in [(3, C_PURP, "-"), (1.5, C_TEAL, "--"), (1, C_CTRL, ":")]:
        ax.plot(x, np.clip(fisher_scalar(x, alpha), 0, 500), color=col, lw=2, ls=ls,
                label=rf"$\alpha={alpha:g}$")
    ax.axvspan(-0.35, 0.35, alpha=0.06, color=C_PURP)
    ax.axvline(0, color=C_PURP, lw=0.8, ls="--", alpha=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\mathcal{I}_\alpha(x)$")
    panel_title(ax, "(b) Instantaneous state Fisher field")
    ax.legend(fontsize=7.5, framealpha=0.95, loc="center right")
    ax.set_ylim(-20, 520)
    style_axes(ax)
    fig.savefig(out / "fig2_scalar_geometry.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Controlled localized nonlinear simulations and adiabatic covariance
# -----------------------------------------------------------------------------


def localized_pitchfork_statistics(
    mu_values: np.ndarray,
    n_rep: int = 32,
    sigma0: float = 0.20,
    s_exp: float = 1.25,
    dtau: float = 0.01,
    burn: int = 5000,
    keep: int = 18000,
    stride: int = 4,
    seed: int = 1441,
) -> dict[str, np.ndarray]:
    """Frozen nonlinear pitchfork in scaled coordinates z=x/sqrt(mu).

    Under tau=mu*t, dz=(z-z^3)d tau + sigma0*mu^(s-1)dW_tau.
    This avoids a numerical slowdown while preserving the exact frozen SDE law.
    """
    rng = np.random.default_rng(seed)
    alphas = (1.0, 2.0, 3.0)
    n_mu = len(mu_values)
    var_x = np.empty((n_rep, n_mu))
    var_g = {a: np.empty((n_rep, n_mu)) for a in alphas}
    means = np.empty((n_rep, n_mu))

    for j, mu in enumerate(mu_values):
        z = np.ones(n_rep) + 0.01 * rng.normal(size=n_rep)
        noise = sigma0 * mu ** (s_exp - 1.0) * np.sqrt(dtau)
        for _ in range(burn):
            z += (z - z ** 3) * dtau + noise * rng.normal(size=n_rep)
        samples = []
        for k in range(keep):
            z += (z - z ** 3) * dtau + noise * rng.normal(size=n_rep)
            if k % stride == 0:
                samples.append(np.sqrt(mu) * z.copy())
        x = np.asarray(samples)
        means[:, j] = np.mean(x, axis=0)
        var_x[:, j] = np.var(x, axis=0)
        for a in alphas:
            gx = np.sign(x) * np.abs(x) ** a
            var_g[a][:, j] = np.var(gx, axis=0)

    return {"var_x": var_x, "var_g1": var_g[1.0], "var_g2": var_g[2.0], "var_g3": var_g[3.0], "mean_x": means}


def nonstationary_ou_covariance(
    eps: float,
    mu_eval_desc: np.ndarray,
    sigma0: float = 0.20,
    s_exp: float = 1.25,
    zeta: float = 2.2,
) -> np.ndarray:
    mu0 = float(mu_eval_desc[0])
    vf = lambda m: sigma0 ** 2 * m ** (2 * s_exp - 1) / 4.0

    def rhs(mu: float, y: np.ndarray) -> list[float]:
        # dV/dt=-4 mu V+sigma(mu)^2 and dmu/dt=-eps mu^zeta.
        return [(4 * mu * y[0] - sigma0 ** 2 * mu ** (2 * s_exp)) / (eps * mu ** zeta)]

    sol = solve_ivp(rhs, (mu0, float(mu_eval_desc[-1])), [vf(mu0)],
                    t_eval=mu_eval_desc, method="Radau", rtol=2e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0]


def generate_fig3(out: Path) -> None:
    """Generate the four-panel controlled-limit validation figure."""
    mu = np.geomspace(1.0, 1.0e-3, 14)
    sigma0, s_exp = 0.20, 1.25
    stats = localized_pitchfork_statistics(mu, sigma0=sigma0, s_exp=s_exp)
    var_ou = sigma0 ** 2 * mu ** (2 * s_exp - 1) / 4.0

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.72), constrained_layout=True)

    def bands(ax, data, color, label=None, marker="o", line: bool = True,
              alpha_fill: float = 0.15, lw: float = 1.3):
        med = np.median(data, axis=0)
        lo, hi = np.percentile(data, [10, 90], axis=0)
        ax.fill_between(mu, lo, hi, color=color, alpha=alpha_fill, linewidth=0, zorder=1)
        if line:
            ax.plot(mu, med, marker=marker, ms=3.2, color=color, label=label, lw=lw, zorder=2)
        else:
            ax.plot(mu, med, color=color, label=label, lw=lw, zorder=2)
        return med

    # (a) Localization relative to the branch scale.
    ax = axes[0, 0]
    loc = np.sqrt(stats["var_x"]) / np.sqrt(mu)[None, :]
    bands(ax, loc, C_PURP, "nonlinear SDE")
    ax.plot(mu, np.sqrt(var_ou) / np.sqrt(mu), color=C_DARK, ls="--", lw=1.5,
            label="local OU prediction", zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel("Relative fluctuation scale")
    panel_title(ax, "(a) Localization improves")
    ax.legend(fontsize=7.0, framealpha=0.95, loc="upper right")
    style_axes(ax)

    # (b) Nonlinear latent variance versus the OU prediction.
    ax = axes[0, 1]
    bands(ax, stats["var_x"] / var_ou[None, :], C_TEAL, "nonlinear / OU")
    ax.axhline(1, color=C_DARK, ls="--", lw=1.2)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0.75, 1.30)
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel("Variance ratio")
    panel_title(ax, "(b) Variance converges")
    ax.legend(fontsize=7.0, framealpha=0.95, loc="best")
    style_axes(ax)

    # (c) Observation-order sweep: amplification, balance, and collapse.
    ax = axes[1, 0]
    colors = {1.0: C_BLUE, 2.0: C_TEAL, 3.0: C_PURP}
    var_map = {1.0: stats["var_g1"], 2.0: stats["var_g2"], 3.0: stats["var_g3"]}
    sigma_sq = (sigma0 * mu ** s_exp) ** 2
    for alpha in (1.0, 2.0, 3.0):
        transmission = var_map[alpha] / sigma_sq[None, :]
        bands(ax, transmission, colors[alpha], label=rf"$\alpha={int(alpha)}$",
              marker="o", line=False, alpha_fill=0.10, lw=1.8)
        theory = (alpha ** 2 / 4.0) * mu ** (alpha - 2.0)
        ax.plot(mu, theory, color="black", ls=(0, (4, 2)), lw=1.5, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel("Noise-normalized transmission")
    panel_title(ax, "(c) Observation order controls transmission")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=colors[1.0], lw=1.8, label=r"$\alpha=1$"),
        Line2D([0], [0], color=colors[2.0], lw=1.8, label=r"$\alpha=2$"),
        Line2D([0], [0], color=colors[3.0], lw=1.8, label=r"$\alpha=3$"),
        Line2D([0], [0], color="black", ls=(0, (4, 2)), lw=1.5,
               label="dashed = local prediction"),
    ]
    ax.legend(handles=handles, fontsize=6.5, framealpha=0.95, loc="lower left",
              handlelength=2.6, borderpad=0.4, labelspacing=0.3)
    style_axes(ax)

    # (d) Adiabatic convergence under decreasing drift prefactor.
    ax = axes[1, 1]
    mu_dense = np.geomspace(1.0, 1.0e-3, 240)
    vf_dense = sigma0 ** 2 * mu_dense ** (2 * s_exp - 1) / 4.0
    for eps, col in [(1.0, C_RED), (0.3, C_ORANGE), (0.05, C_TEAL)]:
        V = nonstationary_ou_covariance(eps, mu_dense, sigma0=sigma0, s_exp=s_exp)
        ax.plot(mu_dense, V / vf_dense, color=col, label=rf"$\epsilon={eps:g}$")
    ax.axhline(1, color=C_DARK, ls="--", lw=1.1)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$V(t)/V_{\rm frozen}(\mu)$")
    panel_title(ax, "(d) Drift-rate convergence")
    ax.legend(fontsize=7.0, framealpha=0.95, loc="best")
    style_axes(ax)

    fig.savefig(out / "fig3_controlled_limits.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Coupled nonnormal rotating critical mode
# -----------------------------------------------------------------------------


def rotating_model(
    mu: float,
    alpha: float = 3.0,
    eta: float = 2.5,
    chi: float = 2.0,
    mu_max: float = 0.8,
    theta_max: float = np.pi / 3,
    theta_power: float = 0.65,
    sigma: float = 0.08,
    se: float = 0.20,
    all_dark: bool = False,
) -> dict[str, np.ndarray | float]:
    theta = theta_max * (mu / mu_max) ** theta_power
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s], [s, c]])
    A0 = np.array([[-2.0 * mu, chi], [0.0, -eta]])
    A = Q @ A0 @ Q.T
    D = sigma ** 2 * np.eye(2)
    Sigma = solve_continuous_lyapunov(A, -D)
    q = Q @ np.array([1.0, 0.0])
    p0 = np.array([1.0, chi / (eta - 2.0 * mu)])
    p = Q @ p0
    dc = float(p @ D @ p)
    xstar = np.sqrt(mu) * q
    if all_dark:
        J = np.diag([alpha * abs(xstar[0]) ** (alpha - 1),
                     alpha * abs(xstar[1]) ** (alpha - 1)])
    else:
        J = np.diag([alpha * abs(xstar[0]) ** (alpha - 1), 1.0])
    H = J / se
    I = J.T @ J / se ** 2
    F = float(q @ I @ q)
    sigma_c = dc / (4.0 * mu) * np.outer(q, q)
    E = Sigma - sigma_c
    Cfull = H @ Sigma @ H.T
    Ccrit = H @ sigma_c @ H.T
    Crem = H @ E @ H.T
    return {
        "theta": theta, "Q": Q, "A": A, "D": D, "Sigma": Sigma, "q": q, "p": p,
        "dc": dc, "xstar": xstar, "J": J, "H": H, "F": F, "sigma_c": sigma_c,
        "E": E, "Cfull": Cfull, "Ccrit": Ccrit, "Crem": Crem,
    }


def generate_fig4(out: Path) -> None:
    mu = np.geomspace(0.8, 0.005, 180)
    vals = [rotating_model(float(m)) for m in mu]
    dark_vals = [rotating_model(float(m), all_dark=True) for m in mu]

    dc = np.array([v["dc"] for v in vals], dtype=float)
    qDq = np.array([
        float(np.asarray(v["q"]) @ np.asarray(v["D"]) @ np.asarray(v["q"]))
        for v in vals
    ])
    nonnormal_gain = dc / qDq

    state_crit = np.array([np.linalg.norm(v["sigma_c"], 2) for v in vals])
    state_rem = np.array([np.linalg.norm(v["E"], 2) for v in vals])
    F = np.array([v["F"] for v in vals], dtype=float)

    mixed_full = np.array([np.trace(v["Cfull"]) for v in vals], dtype=float)
    mixed_crit = np.array([np.trace(v["Ccrit"]) for v in vals], dtype=float)
    dark_full = np.array([np.trace(v["Cfull"]) for v in dark_vals], dtype=float)
    dark_crit = np.array([np.trace(v["Ccrit"]) for v in dark_vals], dtype=float)

    # Exact decomposition of directional Fisher strength into fixed-sensor channels.
    f1, f2 = [], []
    for v in vals:
        q = np.asarray(v["q"])
        J = np.asarray(v["J"])
        se = 0.20
        f1.append((J[0, 0] * q[0] / se) ** 2)
        f2.append((J[1, 1] * q[1] / se) ** 2)
    f1, f2 = np.asarray(f1), np.asarray(f2)

    fig, axes = plt.subplots(2, 3, figsize=(7.35, 5.05), constrained_layout=True)

    # (a) Geometry only: arrows are directions, dots are equilibria.
    ax = axes[0, 0]
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.axvline(0, color="#bbbbbb", lw=0.8)
    for m, col in [(0.8, C_BLUE), (0.1, C_TEAL), (0.01, C_PURP)]:
        v = rotating_model(m)
        q, xs = np.asarray(v["q"]), np.asarray(v["xstar"])
        ax.arrow(
            0, 0, q[0], q[1], width=0.010, head_width=0.08,
            length_includes_head=True, color=col, alpha=0.9,
        )
        ax.plot(xs[0], xs[1], "o", color=col, ms=4.5)
        ax.text(q[0] * 1.05, q[1] * 1.05, rf"$\mu={m:g}$", color=col, fontsize=7.1)
    ax.text(0.72, -0.16, "flattening $x_1$ sensor", color=C_PURP, fontsize=7.1, ha="center")
    ax.text(-0.16, 0.72, "preserved $x_2$ sensor", color=C_CTRL, fontsize=7.1, rotation=90, va="center")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xlabel("physical $x_1$")
    ax.set_ylabel("physical $x_2$")
    panel_title(ax, "(a) Rotating mode and\nfixed sensors")
    style_axes(ax)

    # (b) Single-purpose nonnormality diagnostic; no repeated angle/alignment curve.
    ax = axes[0, 1]
    ax.plot(mu, nonnormal_gain, color=C_ORANGE, lw=2.0, label=r"nonnormal gain $d_c/(q^\top Dq)$")
    ax.axhline(1.0, color=C_DARK, ls="--", lw=1.2, label="normal baseline")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"Excitation gain $d_c/(q^\top Dq)$")
    panel_title(ax, "(b) Nonnormal excitation gain")
    ax.legend(fontsize=6.7, framealpha=0.95, loc="upper left")
    style_axes(ax)

    # (c) Exact covariance decomposition; ratio is stated in the caption, not replotted.
    ax = axes[0, 2]
    # Figure-wide style grammar: purple dashed = critical term; gray dotted = remainder.
    ax.plot(mu, state_crit, color=C_PURP, ls="--", lw=1.7, label=r"$\|\Sigma_c\|_2$")
    ax.plot(mu, state_rem, color=C_CTRL, ls=":", lw=1.8, label=r"$\|E\|_2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Covariance norm")
    panel_title(ax, "(c) Exact Lyapunov\ndecomposition")
    ax.legend(fontsize=6.7, framealpha=0.95, loc="upper left", bbox_to_anchor=(0.03, 0.97))
    style_axes(ax)

    # (d) Fisher decomposition only; P_c is shown later and is not duplicated here.
    ax = axes[1, 0]
    ax.plot(mu, F, color=C_DARK, lw=2.0, label=r"total $F_c$")
    ax.plot(mu, f1, color=C_PURP, ls="--", lw=1.6, label=r"$x_1$ flattening")
    ax.plot(mu, f2, color=C_CTRL, ls=":", lw=1.8, label=r"$x_2$ misalignment")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Directional Fisher strength")
    panel_title(ax, "(d) Two routes to\nFisher collapse")
    ax.legend(fontsize=6.4, framealpha=0.95, loc="lower left")
    style_axes(ax)

    # (e,f) Symmetric comparison: full signal trace versus critical contribution only.
    ax = axes[1, 1]
    ax.plot(mu, mixed_full, color=C_DARK, lw=2.0, label="full signal trace")
    ax.plot(mu, mixed_crit, color=C_PURP, ls="--", lw=1.7, label=r"critical term $P_c$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Whitened signal covariance")
    panel_title(ax, "(e) Stable signal masks\ncritical collapse")
    ax.legend(fontsize=6.5, framealpha=0.95, loc="upper right")
    style_axes(ax)

    ax = axes[1, 2]
    ax.plot(mu, dark_full, color=C_DARK, lw=2.0, label="full signal trace")
    ax.plot(mu, dark_crit, color=C_PURP, ls="--", lw=1.7, label=r"critical term $P_c$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Whitened signal covariance")
    panel_title(ax, "(f) All signal channels flatten")
    ax.legend(fontsize=6.5, framealpha=0.95, loc="lower left")
    style_axes(ax)

    fig.savefig(out / "fig4_nonnormal_rotating_validation.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Finite-history state-estimation experiment
# -----------------------------------------------------------------------------


def _gls_mse_from_synthetic_records(
    a_value: float,
    fisher_value: float,
    horizon: float,
    rng: np.random.Generator,
    n_batches: int = 20,
    trials_per_batch: int = 400,
    n_time: int = 160,
    xi_true: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Simulate whitened observation increments and apply the GLS estimator.

    The measurement-noise-limited scalar record is
        dZ_t = sqrt(F_c) exp(-a t) xi_true dt + dV_t.
    Midpoint discretization gives a discrete Fisher information that converges
    to F_c(1-exp(-2aH))/(2a).  The estimator is constructed from each synthetic
    record rather than sampling estimator errors directly from their target law.
    """
    dt = horizon / n_time
    t = (np.arange(n_time) + 0.5) * dt
    design = np.sqrt(fisher_value) * np.exp(-a_value * t)
    information_discrete = float(np.sum(design ** 2) * dt)
    mse = np.empty(n_batches)
    signal_increment = xi_true * design * dt
    for b in range(n_batches):
        noise_increment = np.sqrt(dt) * rng.normal(size=(trials_per_batch, n_time))
        observation_increment = signal_increment[None, :] + noise_increment
        xi_hat = observation_increment @ design / information_discrete
        mse[b] = np.mean((xi_hat - xi_true) ** 2)
    return mse, information_discrete


def generate_fig5(out: Path) -> None:
    mu = np.geomspace(0.8, 0.02, 18)
    alpha, se = 1.5, 1.0
    a = 2.0 * mu
    F = alpha ** 2 * mu ** (alpha - 1) / se ** 2
    H_fixed = np.ones_like(mu)
    H_matched = 5.0 / a
    I_fixed = F * (1 - np.exp(-2 * a * H_fixed)) / (2 * a)
    I_matched = F * (1 - np.exp(-2 * a * H_matched)) / (2 * a)

    # Genuine synthetic-observation GLS experiment.  Each trial generates a
    # discretized Gaussian observation record and applies the estimator.
    rng = np.random.default_rng(7007)
    n_batches, trials_per_batch, n_time = 20, 400, 160
    mse_fixed = np.empty((n_batches, len(mu)))
    mse_matched = np.empty((n_batches, len(mu)))
    I_fixed_discrete = np.empty_like(mu)
    I_matched_discrete = np.empty_like(mu)
    for j, (aj, Fj, Hf, Hm) in enumerate(zip(a, F, H_fixed, H_matched)):
        mse_fixed[:, j], I_fixed_discrete[j] = _gls_mse_from_synthetic_records(
            float(aj), float(Fj), float(Hf), rng,
            n_batches=n_batches, trials_per_batch=trials_per_batch, n_time=n_time,
        )
        mse_matched[:, j], I_matched_discrete[j] = _gls_mse_from_synthetic_records(
            float(aj), float(Fj), float(Hm), rng,
            n_batches=n_batches, trials_per_batch=trials_per_batch, n_time=n_time,
        )

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.85), constrained_layout=True)

    ax = axes[0]
    ax.plot(mu, F / F[0], color=C_PURP, ls="--", lw=1.7, label=r"instantaneous $F_c$")
    ax.plot(mu, I_fixed / I_fixed[0], color=C_ORANGE, lw=1.8, label=r"fixed $H=1$")
    ax.plot(mu, I_matched / I_matched[0], color=C_TEAL, ls=":", lw=2.0,
            label=r"matched $H=5/a$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"Information, normalized at $\mu=0.8$")
    panel_title(ax, "(a) Finite history changes\ninformation scaling")
    ax.legend(fontsize=6.7, framealpha=0.95, loc="best")
    style_axes(ax)

    ax = axes[1]
    med_f = np.median(mse_fixed, axis=0)
    lo_f, hi_f = np.percentile(mse_fixed, [10, 90], axis=0)
    med_m = np.median(mse_matched, axis=0)
    lo_m, hi_m = np.percentile(mse_matched, [10, 90], axis=0)
    ax.fill_between(mu, lo_f, hi_f, color=C_ORANGE, alpha=0.13, linewidth=0)
    ax.fill_between(mu, lo_m, hi_m, color=C_TEAL, alpha=0.13, linewidth=0)
    ax.plot(mu, 1 / I_fixed, color=C_ORANGE, lw=1.7, label=r"fixed CR bound")
    ax.plot(mu, med_f, "o", mfc="white", mec=C_ORANGE, mew=0.9, ms=3.2,
            label=r"fixed GLS MSE")
    ax.plot(mu, 1 / I_matched, color=C_TEAL, lw=1.7, label=r"matched CR bound")
    ax.plot(mu, med_m, "s", mfc="white", mec=C_TEAL, mew=0.9, ms=3.0,
            label=r"matched GLS MSE")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\mathbb{E}(\widehat\xi-\xi)^2$")
    panel_title(ax, "(b) Fixed and matched\nprecision diverge")
    ax.legend(fontsize=5.9, framealpha=0.95, loc="best", labelspacing=0.25)
    style_axes(ax)

    ax = axes[2]
    rate_fixed = I_fixed / H_fixed
    rate_matched = I_matched / H_matched
    ax.plot(mu, rate_fixed, color=C_ORANGE, lw=1.8, label=r"fixed $I_H/H$")
    ax.plot(mu, rate_matched, color=C_TEAL, ls=":", lw=2.0,
            label=r"matched $I_H/H$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel("Information per unit time")
    panel_title(ax, "(c) Information rate\nstill collapses")
    ax.legend(fontsize=6.7, framealpha=0.95, loc="best")
    style_axes(ax)

    fig.savefig(out / "fig5_finite_history_estimation.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Fixed-noise finite-drift stress test preserving the original result
# -----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def simulate_fixed_noise(seed: int, n_steps: int = 30000) -> tuple[np.ndarray, np.ndarray]:
    dt, mu0, muend = 0.01, 2.0, 0.05
    rng = np.random.default_rng(seed)
    mu = np.linspace(mu0, muend, n_steps)
    x = np.empty(n_steps)
    x[0] = np.sqrt(mu0) + rng.normal(0, 0.05)
    for i in range(1, n_steps):
        x[i] = x[i - 1] + (mu[i - 1] * x[i - 1] - x[i - 1] ** 3) * dt \
            + SIGMA_FIXED * np.sqrt(dt) * rng.normal()
    return x, mu


def generate_fig6(out: Path) -> None:
    n_rep, w, dt = 24, 1500, 0.01
    n_density = 96
    t = np.arange(30000) * dt
    mu = np.linspace(2.0, 0.05, 30000)
    tg = t[w - 1:]
    channels = {
        "linear $x$": (lambda x: x, C_CTRL, ":"),
        r"saturating $\tanh(2x)$": (lambda x: np.tanh(2 * x), C_TEAL, "--"),
        "flattening $x^3$": (lambda x: x ** 3, C_PURP, "-"),
    }
    var_runs = {k: [] for k in channels}
    fisher_runs = {"linear": [], "saturating": [], "cubic": []}
    density_paths = []

    for seed in range(n_density):
        x, _ = simulate_fixed_noise(3000 + seed)
        if seed < n_rep:
            # Each channel receives an independent measurement-noise realization
            # with the same variance while sharing the same latent trajectory.
            eps_rng = np.random.default_rng(9000 + seed)
            for name, (fn, _, _) in channels.items():
                eps = SE * eps_rng.normal(size=len(x))
                var_runs[name].append(rolling_var(fn(x) + eps, w))
            fisher_runs["linear"].append(np.ones_like(x) / SE ** 2)
            fisher_runs["saturating"].append((2 / np.cosh(2 * x) ** 2) ** 2 / SE ** 2)
            fisher_runs["cubic"].append(9 * x ** 4 / SE ** 2)
        density_paths.append(x)

    fig = plt.figure(figsize=(7.15, 4.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # (a) Time-resolved ensemble state density across many trajectories.
    ax = ax_a
    density = np.asarray(density_paths)
    stride = 15
    t_s = t[::stride]
    density_s = density[:, ::stride]
    x_abs = max(2.2, float(np.percentile(np.abs(density_s), 99.5)))
    x_edges = np.linspace(-x_abs, x_abs, 121)
    time_edges = np.linspace(t_s[0], t_s[-1], 101)
    H, _, _ = np.histogram2d(
        np.tile(t_s, density_s.shape[0]),
        density_s.reshape(-1),
        bins=[time_edges, x_edges],
    )
    # Normalize each time column to show the conditional state density p(x|t).
    H = H.T
    colsum = H.sum(axis=0, keepdims=True)
    H = np.divide(H, colsum, out=np.zeros_like(H), where=colsum > 0)
    ax.imshow(
        H,
        origin='lower', aspect='auto', interpolation='nearest',
        extent=[time_edges[0], time_edges[-1], x_edges[0], x_edges[-1]],
        cmap='viridis', vmin=0.0, vmax=np.quantile(H[H > 0], 0.985) if np.any(H > 0) else 1.0,
    )
    ax.plot(t, np.sqrt(mu), color=C_GREEN, lw=1.4, ls='--', label=r"stable branches $\pm\sqrt{\mu}$")
    ax.plot(t, -np.sqrt(mu), color=C_GREEN, lw=1.0, ls='--', alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    panel_title(ax, "(a) Time-state density\nunder fixed noise")
    ax.legend(fontsize=6.4, framealpha=0.95, loc='upper right')
    style_axes(ax)

    # (b) Realized local Fisher strengths on the same latent ensembles.
    ax = ax_b
    for key, col, ls, label in [
        ("cubic", C_PURP, "-", r"flattening $x^3$"),
        ("saturating", C_TEAL, "--", r"saturating $\tanh(2x)$"),
        ("linear", C_CTRL, ":", r"linear $x$")]:
        data = np.asarray(fisher_runs[key])
        med = np.median(data, axis=0)
        lo, hi = np.percentile(data, [10, 90], axis=0)
        ax.fill_between(t, lo, hi, color=col, alpha=0.10, linewidth=0)
        ax.plot(t, med, color=col, ls=ls, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Realized local Fisher strength $\mathcal{I}(x_t)$")
    panel_title(ax, "(b) State-dependent channel strengths")
    ax.legend(fontsize=6.7, framealpha=0.95, loc="best")
    style_axes(ax)

    # (c) Main outcome panel enlarged across the full bottom row.
    ax = ax_c
    for name, (_, col, ls) in channels.items():
        data = np.asarray(var_runs[name])
        med = np.median(data, axis=0)
        lo, hi = np.percentile(data, [10, 90], axis=0)
        ax.fill_between(tg, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(tg, med, color=col, ls=ls, label=name)
    ax.axhline(SE ** 2, color=C_DARK, lw=1.0, ls="--", label="measurement-noise floor")
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute rolling variance")
    panel_title(ax, "(c) Absolute variance reveals channel-specific suppression")
    ax.legend(fontsize=6.7, framealpha=0.95, ncol=2, loc="upper left")
    style_axes(ax)

    fig.savefig(out / "fig6_fixed_noise_stress_test.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    generate_fig1(out)
    generate_fig2(out)
    generate_fig3(out)
    generate_fig4(out)
    generate_fig5(out)
    generate_fig6(out)
    print(f"Generated 6 figures in {out}")


if __name__ == "__main__":
    main()
