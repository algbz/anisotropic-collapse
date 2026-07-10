#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
from __future__ import annotations

"""Reproduce the figures for the manuscript
"Fisher Collapse and Directional Concealment Near Bifurcation."

The script is self-contained and generates all 13 manuscript figures from
analytic calculations and seeded stochastic simulations. No external data or
project-specific helper modules are required.

Author
------
Aldo Alberto Aguilar Bermúdez <aldgbz@pm.me>

Repository
----------
https://github.com/algbz/anisotropic-collapse

Requirements
------------
Python 3.10 or newer, NumPy, SciPy, and Matplotlib.

Examples
--------
Generate every figure in the default ``figures/`` directory::

    python generate_figures.py

Generate every figure in a custom directory::

    python generate_figures.py --output-dir results

Generate one figure only::

    python generate_figures.py --figure 8

List the available figures::

    python generate_figures.py --list

Outputs
-------
fig1_engineering_schematic.png
fig2_engineering_policy.png
fig3_fisher_field.png
fig4_hierarchy.png
fig5_channel.png
fig6_scalar_occupation_v2.png
fig7_ablation.png
fig8_channel_geometry.png
fig9_directional_collapse.png
fig10_alignment.png
fig11_anisotropic_occupation_v2.png
fig12_masking.png
fig13_diagnostics.png
"""

import argparse
import gc
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------

TITLE_FS = 11
LEGEND_FS = 11
TICK_FS = 10
DPI = 300

C_GREEN = "#3a7d44"
C_PURP = "#6a4c93"
C_CTRL = "#777777"
C_TEAL = "#2a7a6f"
C_GREY = "#999999"
C_DARK = "#1a1a2e"
C_ORANGE = "#c44b25"
C_BG = "#f7f7fb"
C_GRID = "#d7d7d7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------

SIGMA = 0.3
DT = 0.01
N_STEPS = 30000
MU0 = 2.0
MU_END = 0.05
NU = 1.5
SE = 0.05
W = 1500
ALPHA = 3.0

TIME = np.arange(N_STEPS) * DT
T_TRANS = (MU0 - 0.1) / ((MU0 - MU_END) / (N_STEPS * DT))

ATT_INIT = [
    (np.sqrt(MU0), np.sqrt(NU)),
    (np.sqrt(MU0), -np.sqrt(NU)),
    (-np.sqrt(MU0), np.sqrt(NU)),
    (-np.sqrt(MU0), -np.sqrt(NU)),
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def smooth(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(a)
    return np.convolve(np.asarray(a), np.ones(k) / k, mode="same")


def smooth_edge(a: np.ndarray, k: int) -> np.ndarray:
    """Moving average with edge padding, evaluated in O(n) time."""
    a = np.asarray(a, dtype=float)
    if k <= 1:
        return a.copy()
    left = k // 2
    right = k - 1 - left
    padded = np.pad(a, (left, right), mode="edge")
    cs = np.cumsum(np.insert(padded, 0, 0.0))
    return (cs[k:] - cs[:-k]) / float(k)


def rolling_var(a: np.ndarray, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    cs = np.cumsum(np.insert(a, 0, 0.0))
    cs2 = np.cumsum(np.insert(a * a, 0, 0.0))
    sw = cs[w:] - cs[:-w]
    sw2 = cs2[w:] - cs2[:-w]
    mean = sw / w
    return (sw2 / w - mean * mean)[1:]


def add_transition_line(ax, x=T_TRANS, text=r"$\mu \approx 0.1$", alpha=0.4, label=True, y=None):
    ax.axvline(x, color=C_GREY, lw=0.9, ls="--", alpha=alpha)
    if label:
        if y is None:
            ymin, ymax = ax.get_ylim()
            y = ymax * 0.90 if np.isfinite(ymax) else 0.9
        ax.text(x + 4, y, text, fontsize=11, color=C_GREY, alpha=0.75, fontstyle="italic")


def style_axes(ax, grid=True):
    if grid:
        ax.grid(True, color=C_GRID, lw=0.6, alpha=0.8, ls="--")
    ax.tick_params(labelsize=TICK_FS)


def fig_2plus1_canvas():
    fig = plt.figure(figsize=(12, 8.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.24, wspace=0.28)
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])


def fig_3h_canvas():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    return fig, axes


def rounded_box(ax, x, y, w, h, ec, fc="white", lw=1.0, r=0.03):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=1
    )
    ax.add_patch(box)
    return box


def arrow(ax, p1, p2, color=C_DARK, lw=1.3, connectionstyle="arc3"):
    """Draw an arrow whose tip and tail terminate at the supplied coordinates.

    For the schematic panels, coordinates are chosen on the borders of
    rounded boxes. Explicit zero shrink avoids the default Matplotlib
    offset that can make arrows appear to stop short of the box border.
    """
    arr = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16, lw=lw,
        color=color, connectionstyle=connectionstyle,
        shrinkA=0.0, shrinkB=0.0, zorder=5
    )
    ax.add_patch(arr)
    return arr


def panel_label(ax, label: str, title: str, color: str):
    ax.text(
        0.04, 0.96, label, transform=ax.transAxes, va="top", ha="center",
        fontsize=12.5, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.18", fc=color, ec=color)
    )
    ax.text(
        0.13, 0.965, title, transform=ax.transAxes, va="top", ha="left",
        fontsize=13.5, fontweight="bold", color=color
    )


# ---------------------------------------------------------------------
# Simulation layer
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def simulate_scalar(seed: int = 42):
    rng = np.random.RandomState(seed)
    mu = np.linspace(MU0, MU_END, N_STEPS)
    x = np.zeros(N_STEPS)
    x[0] = np.sqrt(MU0) + rng.normal(0, 0.05)
    for i in range(1, N_STEPS):
        drift = mu[i] * x[i - 1] - x[i - 1] ** 3
        x[i] = x[i - 1] + drift * DT + SIGMA * np.sqrt(DT) * rng.normal()
    return x, mu


def observe_scalar(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(x) * np.abs(x) ** alpha


def fisher_scalar(x: np.ndarray, alpha: float, se: float = SE) -> np.ndarray:
    x = np.asarray(x)
    return alpha ** 2 * np.abs(x) ** (2 * alpha - 2) / se ** 2


@lru_cache(maxsize=None)
def simulate_2d(seed: int = 42):
    rng = np.random.RandomState(seed)
    mu = np.linspace(MU0, MU_END, N_STEPS)
    nu = NU * np.ones(N_STEPS)
    x1 = np.zeros(N_STEPS)
    x2 = np.zeros(N_STEPS)
    x1[0] = np.sqrt(MU0) + rng.normal(0, 0.05)
    x2[0] = np.sqrt(NU) + rng.normal(0, 0.05)

    for i in range(1, N_STEPS):
        d1 = mu[i] * x1[i - 1] - x1[i - 1] ** 3
        d2 = nu[i] * x2[i - 1] - x2[i - 1] ** 3
        x1[i] = x1[i - 1] + d1 * DT + SIGMA * np.sqrt(DT) * rng.normal()
        x2[i] = x2[i - 1] + d2 * DT + SIGMA * np.sqrt(DT) * rng.normal()
    return x1, x2, mu, nu


def G_collapse(x1: np.ndarray, x2: np.ndarray, alpha: float = ALPHA):
    return x2.copy(), np.sign(x1) * np.abs(x1) ** alpha


def G_linear(x1: np.ndarray, x2: np.ndarray):
    return x2.copy(), x1.copy()


def fisher_eigs(x1: np.ndarray, x2: np.ndarray, alpha: float = ALPHA, se: float = SE):
    lam_x1 = alpha ** 2 * np.abs(x1) ** (2 * (alpha - 1)) / se ** 2
    lam_x2 = np.ones_like(x1) / se ** 2
    return np.minimum(lam_x1, lam_x2), np.maximum(lam_x1, lam_x2)


def fisher_eigs_rotated(x1: np.ndarray, x2: np.ndarray, alpha: float, phi: float, se: float = SE):
    c, s = np.cos(phi), np.sin(phi)
    xr = c * x1 + s * x2
    lam_r = alpha ** 2 * np.abs(xr) ** (2 * (alpha - 1)) / se ** 2
    lam_p = np.ones_like(x1) / se ** 2
    return np.minimum(lam_r, lam_p), np.maximum(lam_r, lam_p)


# ---------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------

def generate_fig1_engineering_schematic(out: Path):
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.04, wspace=0.03)
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    panel_colors = [C_PURP, C_PURP, "#0d8065", "#f25c05"]
    titles = [
        "Monitored structure near instability",
        "Candidate observation channels",
        "Channel evaluation",
        "Policy action",
    ]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, col, ttl, lab in zip(axs, panel_colors, titles, labels):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        rounded_box(ax, 0.015, 0.02, 0.97, 0.95, ec=col, fc=C_BG, lw=0.9, r=0.025)
        panel_label(ax, lab, ttl, col)

    def ctext(ax, x, y, w, h, s, **kw):
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", **kw)

    def arrow_label(ax, p0, p1, s, color, rotation=0, dy=0.0, dx=0.0, fs=10.8):
        xm = 0.5 * (p0[0] + p1[0]) + dx
        ym = 0.5 * (p0[1] + p1[1]) + dy
        ax.text(
            xm, ym, s, ha="center", va="center", rotation=rotation,
            fontsize=fs, color=color,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
            zorder=10,
        )

    # Panel a
    ax = axs[0]
    top_box = (0.13, 0.43, 0.69, 0.38)
    bot_box = (0.13, 0.07, 0.69, 0.22)
    rounded_box(ax, *top_box, ec="#8c7dd0", fc="white", lw=1.0)
    ax.text(0.48, 0.75, "Latent system (continuum)", ha="center", va="center", fontsize=13.0, color=C_PURP)
    ax.text(0.48, 0.675, "state $q$", ha="center", va="center", fontsize=11.5, color=C_PURP)
    ax.plot([0.15, 0.77], [0.61, 0.61], color=C_PURP, lw=1.6)
    for xpos in [0.19, 0.47, 0.70]:
        ax.plot([xpos, xpos], [0.595, 0.625], color=C_PURP, lw=1.3)
    rounded_box(ax, 0.35, 0.54, 0.23, 0.09, ec="none", fc="#ded9ec", lw=0.0, r=0.015)
    ax.text(0.19, 0.56, r"$-q_c$", ha="center", va="center", fontsize=13.0, color=C_PURP)
    ax.text(0.47, 0.56, r"$0$", ha="center", va="center", fontsize=13.0, color=C_PURP)
    ax.text(0.705, 0.56, r"$q_c$", ha="center", va="center", fontsize=13.0, color=C_PURP)
    ax.text(0.79, 0.61, r"$q$", ha="center", va="center", fontsize=13.0, color=C_PURP)
    ax.text(0.47, 0.47, "Blind region", ha="center", va="center", fontsize=13.5, color=C_PURP)
    p0 = (0.48, 0.43)
    p1 = (0.48, 0.29)
    arrow(ax, p0, p1, color=C_PURP)
    arrow_label(ax, p0, p1, "measurement", color=C_PURP, rotation=0, fs=10.8)
    rounded_box(ax, *bot_box, ec="#5f3ab4", fc="white", lw=1.0)
    ax.text(0.48, 0.235, r"Current (degraded) channel   $g_A(q)$", ha="center", va="center", fontsize=13.0, color=C_PURP)
    ax.text(0.48, 0.165, r"observed feature   $z = g_A(q)$", ha="center", va="center", fontsize=12.5, color=C_PURP)
    ax.text(0.48, 0.095, r"Low slope and low local information near $q \approx 0$", ha="center", va="bottom", fontsize=11.5, color=C_PURP)

    # Panel b
    ax = axs[1]
    state_box = (0.07, 0.38, 0.20, 0.23)
    rounded_box(ax, *state_box, ec="#5f3ab4", fc="white", lw=1.0)
    ax.text(state_box[0] + state_box[2] / 2, state_box[1] + 0.145, "State\nestimate", ha="center", va="center", fontsize=14.0, color=C_PURP)
    ax.text(state_box[0] + state_box[2] / 2, state_box[1] + 0.055, r"$\tilde{x}_t=(\hat q_t,\hat z_t)$", ha="center", va="center", fontsize=12.8, color=C_PURP)
    yslots = [0.66, 0.41, 0.16]
    texts = [
        ("1) Current channel (degraded)", r"$G_A(q,z)=(z,\operatorname{sign}(q)|q|^\alpha)$"),
        ("2) Auxiliary channel (aligned)", r"$G_B(q,z)=(z,q)$"),
        ("3) High-resolution channel", r"$G_C(q,z)=(z,\beta q)$"),
    ]
    chan_boxes = []
    for y0, (tt, eq) in zip(yslots, texts):
        box = (0.37, y0, 0.51, 0.17)
        chan_boxes.append(box)
        rounded_box(ax, *box, ec="#5f3ab4", fc="white", lw=1.0)
        ax.text(box[0] + box[2] / 2, box[1] + 0.125, tt, ha="center", va="center", fontsize=13.0, color=C_PURP)
        ax.text(box[0] + box[2] / 2, box[1] + 0.055, eq, ha="center", va="center", fontsize=13.1, color=C_PURP)
    sx = state_box[0] + state_box[2]
    sy_mid = state_box[1] + state_box[3] / 2
    top_y = chan_boxes[0][1] + chan_boxes[0][3] / 2
    mid_y = chan_boxes[1][1] + chan_boxes[1][3] / 2
    bot_y = chan_boxes[2][1] + chan_boxes[2][3] / 2
    arrow(ax, (sx, sy_mid + 0.08), (chan_boxes[0][0], top_y), color=C_PURP, connectionstyle="angle3,angleA=0,angleB=90")
    arrow(ax, (sx, sy_mid), (chan_boxes[1][0], mid_y), color=C_PURP)
    arrow(ax, (sx, sy_mid - 0.08), (chan_boxes[2][0], bot_y), color=C_PURP, connectionstyle="angle3,angleA=0,angleB=-90")
    ax.text(
        0.50, 0.018,
        "Same latent system; different local Jacobians\n"
        r"$J_a = \partial G_a / \partial(q,z)\,|_{\hat x_t}$",
        ha="center", va="bottom", fontsize=10.6, color=C_PURP
    )

    # Panel c
    ax = axs[2]
    left_box = (0.08, 0.48, 0.38, 0.24)
    right_box = (0.58, 0.48, 0.34, 0.24)
    rounded_box(ax, *left_box, ec="#0d8065", fc="white", lw=1.0)
    rounded_box(ax, *right_box, ec="#0d8065", fc="white", lw=1.0)
    ax.text(left_box[0] + left_box[2] / 2, left_box[1] + 0.16, "Local channel geometry", ha="center", fontsize=13.5, color="#0d8065")
    ax.text(left_box[0] + left_box[2] / 2, left_box[1] + 0.07, r"$I_a(\hat x_t)=J_a^\top R_a^{-1}J_a$", ha="center", fontsize=14.5, color="#0d8065")
    p0 = (left_box[0] + left_box[2], left_box[1] + left_box[3] / 2)
    p1 = (right_box[0], right_box[1] + right_box[3] / 2)
    arrow(ax, p0, p1, color="#0d8065")
    ax.text(right_box[0] + right_box[2] / 2, right_box[1] + 0.16, "Critical-direction score", ha="center", fontsize=13.5, color="#0d8065")
    ax.text(right_box[0] + right_box[2] / 2, right_box[1] + 0.07, r"$e_q^\top I_a e_q$", ha="center", fontsize=15.5, color="#0d8065")
    ax.text(0.25, 0.22, r"Low score: blind to $q$", ha="center", fontsize=13.5, color="#0d8065")
    ax.text(0.76, 0.22, r"High score: $q$ is visible", ha="center", fontsize=13.5, color="#0d8065")

    # Panel d
    ax = axs[3]
    d1 = (0.08, 0.64, 0.40, 0.16)
    d2 = (0.60, 0.64, 0.28, 0.16)
    d3 = (0.08, 0.34, 0.40, 0.16)
    d4 = (0.60, 0.34, 0.28, 0.16)
    d5 = (0.20, 0.04, 0.60, 0.16)
    for box in [d1, d2, d3, d4, d5]:
        rounded_box(ax, *box, ec="#f25c05", fc="white", lw=1.0)
    ctext(ax, *d1, "Diagnostic\n" + r"$e_q^\top I_a e_q < \tau_t\,?$", fontsize=11.6, color="#f25c05")
    ctext(ax, *d2, "Variance may\nbe unreliable", fontsize=11.6, color="#f25c05")
    ctext(ax, *d3, "Sensing step\n" + r"$a_t^*=\arg\max_a\ e_q^\top I_a e_q-c(a)$", fontsize=10.8, color="#f25c05")
    ctext(ax, *d4, "Add new sensor\n(jump in channel)", fontsize=11.6, color="#f25c05")
    ctext(ax, *d5, "Optional control\nSmall action to avoid blind region", fontsize=11.6, color="#f25c05")
    p_yes0 = (d1[0] + d1[2], d1[1] + d1[3] / 2)
    p_yes1 = (d2[0], d2[1] + d2[3] / 2)
    arrow(ax, p_yes0, p_yes1, color="#f25c05")
    arrow_label(ax, p_yes0, p_yes1, "Yes", color="#f25c05", dy=0.045, fs=10.5)
    p_no0 = (d1[0] + d1[2] / 2, d1[1])
    p_no1 = (d3[0] + d3[2] / 2, d3[1] + d3[3])
    arrow(ax, p_no0, p_no1, color="#f25c05")
    arrow_label(ax, p_no0, p_no1, "No", color="#f25c05", dx=0.03, fs=10.5)
    arrow(ax, (d3[0] + d3[2], d3[1] + d3[3] / 2), (d4[0], d4[1] + d4[3] / 2), color="#f25c05")
    arrow(ax, (d3[0] + d3[2] / 2, d3[1]), (d3[0] + d3[2] / 2, d5[1] + d5[3]), color="#f25c05")
    arrow(ax, (d4[0] + d4[2] / 2, d4[1]), (d4[0] + d4[2] / 2, d5[1] + d5[3]), color="#f25c05")

    fig.savefig(out / "fig1_engineering_schematic.png")

def generate_fig2_engineering_policy(out: Path):
    fig, ax1, ax2, ax3 = fig_2plus1_canvas()

    # Figure 02: simulation-driven, ensemble-aware, and readable.
    # No switch time is hard-coded. Switching is computed from simulated
    # local perturbation experiments across an ensemble of latent trajectories.
    #
    # Median first-switch time means: across ensemble runs, the first time each
    # run selects sensor B; the median is the time by which half of the runs
    # have made that first switch.

    alpha_A = 3.0
    sigma_A = 0.045
    sigma_B = 0.11
    delta_q = 0.035
    n_rep = 40
    smooth_k = 900
    hysteresis = 0.015
    n_ensemble = 24
    representative_seed = 42

    def smooth_edge(a: np.ndarray, k: int) -> np.ndarray:
        """Moving average without artificial edge decay, evaluated in O(n)."""
        a = np.asarray(a, dtype=float)
        if k <= 1:
            return a.copy()
        left = k // 2
        right = k - 1 - left
        padded = np.pad(a, (left, right), mode="edge")
        cs = np.cumsum(np.insert(padded, 0, 0.0))
        return (cs[k:] - cs[:-k]) / float(k)

    def simulate_sensor(q, alpha, sigma, rng, n_obs):
        q = np.asarray(q)
        base = observe_scalar(q, alpha)
        noise = sigma * rng.normal(size=(q.size, n_obs))
        return base[:, None] + noise

    def empirical_separation(q_path, alpha, sigma, rng, delta=delta_q, n_obs=n_rep):
        q_plus = q_path + delta
        q_minus = q_path - delta
        y_plus = simulate_sensor(q_plus, alpha, sigma, rng, n_obs)
        y_minus = simulate_sensor(q_minus, alpha, sigma, rng, n_obs)
        mean_gap = y_plus.mean(axis=1) - y_minus.mean(axis=1)
        pooled_var = 0.5 * (y_plus.var(axis=1) + y_minus.var(axis=1)) + 1e-12
        return (mean_gap ** 2) / pooled_var

    def select_channel(score_A, score_B, hysteresis_margin=hysteresis):
        selected = np.zeros(len(score_A), dtype=int)
        selected[0] = 0 if score_A[0] >= score_B[0] else 1
        for i in range(1, len(score_A)):
            prev = selected[i - 1]
            selected[i] = prev
            if prev == 0 and (score_B[i] - score_A[i]) > hysteresis_margin:
                selected[i] = 1
            elif prev == 1 and (score_A[i] - score_B[i]) > hysteresis_margin:
                selected[i] = 0
        return selected

    # ------------------------------------------------------------------
    # Panel a: simulated noisy channel response experiments.
    # ------------------------------------------------------------------
    rng_resp = np.random.RandomState(123)
    q_grid = np.linspace(-1.8, 1.8, 85)
    resp_A = simulate_sensor(q_grid, alpha_A, sigma_A, rng_resp, 80)
    resp_B = simulate_sensor(q_grid, 1.0, sigma_B, rng_resp, 80)
    mean_A = resp_A.mean(axis=1)
    mean_B = resp_B.mean(axis=1)
    lo_A, hi_A = np.percentile(resp_A, [10, 90], axis=1)
    lo_B, hi_B = np.percentile(resp_B, [10, 90], axis=1)

    ax1.axvspan(-0.45, 0.45, color=C_PURP, alpha=0.10)
    ax1.axhline(0, color=C_GREY, lw=0.8)
    ax1.fill_between(q_grid, lo_A, hi_A, color=C_PURP, alpha=0.16)
    ax1.fill_between(q_grid, lo_B, hi_B, color=C_TEAL, alpha=0.14)
    ax1.plot(q_grid, mean_A, color=C_PURP, lw=2.2, label=r"sensor $A$: nonlinear")
    ax1.plot(q_grid, mean_B, color=C_TEAL, lw=2.0, ls="--", label=r"sensor $B$: linear")
    ax1.text(0.0, -2.0, "blind region", fontsize=10.5, color=C_PURP, ha="center")
    ax1.set_title("(a) Simulated sensor responses", fontsize=TITLE_FS, fontweight="bold")
    ax1.set_xlabel("latent state $q$")
    ax1.set_ylabel("observed signal")
    ax1.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    ax1.set_xlim(-1.9, 1.9)
    ax1.set_ylim(-2.8, 2.8)
    style_axes(ax1)

    # ------------------------------------------------------------------
    # Ensemble simulations for panels b-c.
    # ------------------------------------------------------------------
    all_info_A = []
    all_info_B = []
    for ens_idx in range(n_ensemble):
        state_seed = representative_seed + ens_idx
        x_path, _ = simulate_scalar(seed=state_seed)
        rng_info_A = np.random.RandomState(1000 + ens_idx)
        rng_info_B = np.random.RandomState(2000 + ens_idx)
        info_A_emp = empirical_separation(x_path, alpha_A, sigma_A, rng_info_A)
        info_B_emp = empirical_separation(x_path, 1.0, sigma_B, rng_info_B)

        # Edge-aware smoothing removes the false start/end decay of the linear sensor.
        all_info_A.append(smooth_edge(info_A_emp, smooth_k))
        all_info_B.append(smooth_edge(info_B_emp, smooth_k))

    all_info_A = np.asarray(all_info_A)
    all_info_B = np.asarray(all_info_B)

    # Use one common scale so A and B are comparable.
    ref_scale = np.nanpercentile(np.r_[all_info_A.ravel(), all_info_B.ravel()], 99.5)
    ref_scale = max(float(ref_scale), 1e-12)
    all_score_A = np.log10(1.0 + all_info_A) / np.log10(1.0 + ref_scale)
    all_score_B = np.log10(1.0 + all_info_B) / np.log10(1.0 + ref_scale)
    all_score_A = np.clip(all_score_A, 0.0, 1.0)
    all_score_B = np.clip(all_score_B, 0.0, 1.0)

    ensemble_selected = np.asarray([select_channel(a, b) for a, b in zip(all_score_A, all_score_B)])

    # First-switch summary: for each ensemble trajectory, record only the first
    # time the policy leaves sensor A and switches to sensor B. Panel c then
    # shows the cumulative number of distinct trajectories that have switched
    # by each time, so no trajectory is ever double counted.
    first_switch_idx = []
    first_switch_times = []
    for sel in ensemble_selected:
        sp = np.where((sel[:-1] == 0) & (sel[1:] == 1))[0] + 1
        if len(sp):
            first_switch_idx.append(int(sp[0]))
            first_switch_times.append(float(TIME[sp[0]]))
    first_switch_idx = np.asarray(first_switch_idx, dtype=int)
    first_switch_times = np.asarray(first_switch_times, dtype=float)

    cumulative_switched = np.zeros_like(TIME, dtype=float)
    for idx in first_switch_idx:
        cumulative_switched[idx:] += 1.0
    cumulative_fraction = cumulative_switched / float(n_ensemble)

    # ------------------------------------------------------------------
    # Panel b: ensemble information comparison.
    # ------------------------------------------------------------------
    mean_A = all_score_A.mean(axis=0)
    mean_B = all_score_B.mean(axis=0)
    lo_A, hi_A = np.percentile(all_score_A, [10, 90], axis=0)
    lo_B, hi_B = np.percentile(all_score_B, [10, 90], axis=0)

    ax2.fill_between(TIME, lo_A, hi_A, color=C_PURP, alpha=0.16)
    ax2.fill_between(TIME, lo_B, hi_B, color=C_TEAL, alpha=0.14)
    ax2.plot(TIME, mean_A, color=C_PURP, lw=2.4, label=r"sensor $A$ ensemble mean")
    ax2.plot(TIME, mean_B, color=C_TEAL, lw=2.2, ls="--", label=r"sensor $B$ ensemble mean")
    ax2.set_title("(b) Ensemble empirical information", fontsize=TITLE_FS, fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("normalized empirical discriminability")
    ax2.set_xlim(0, TIME[-1])
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="center left", bbox_to_anchor=(0.03, 0.30))
    style_axes(ax2)

    # ------------------------------------------------------------------
    # Panel c: cumulative first-switch count and cumulative fraction.
    # Count is with respect to the full ensemble: each trajectory can be
    # counted at most once, and the fraction is cumulative_switched / n_ensemble.
    # ------------------------------------------------------------------
    count_line = ax3.step(
        TIME, cumulative_switched, where="post", color="#222222", lw=2.4,
        label="cumulative switched count"
    )
    ax3.fill_between(TIME, 0, cumulative_switched, step="post", color=C_TEAL, alpha=0.12)
    ax3.set_title("(c) Cumulative first-switch count and fraction", fontsize=TITLE_FS, fontweight="bold")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("cumulative switched count")
    ax3.set_xlim(0, TIME[-1])
    ax3.set_ylim(0, max(n_ensemble, cumulative_switched.max()) + 0.5)
    style_axes(ax3)

    ax3r = ax3.twinx()
    frac_line = ax3r.plot(
        TIME, cumulative_fraction, color=C_ORANGE, lw=2.2, ls='--',
        label='cumulative switched fraction'
    )
    ax3r.set_ylabel("cumulative switched fraction")
    ax3r.set_ylim(0, 1.02)
    ax3r.tick_params(labelsize=TICK_FS)
    ax3r.spines['top'].set_visible(False)
    ax3r.spines['right'].set_visible(True)
    ax3r.spines['right'].set_linewidth(0.7)
    ax3r.spines['right'].set_color('#333333')

    handles = count_line + frac_line
    labels = [h.get_label() for h in handles]
    ax3.legend(handles, labels, fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")


    fig.savefig(out / "fig2_engineering_policy.png")
    plt.close(fig)

def generate_fig3_fisher_field(out: Path):
    fig, ax1, ax2, ax3 = fig_2plus1_canvas()

    xr = np.linspace(-2, 2, 400)
    V = -xr ** 2 / 2 + xr ** 4 / 4
    V = (V - V.min()) / (V.max() - V.min())
    g = np.sign(xr) * np.abs(xr) ** 3
    g = g / np.max(np.abs(g))

    ax1.axvspan(-0.5, 0.5, alpha=0.10, color=C_PURP, zorder=0)
    ax1.axvline(0, color=C_PURP, lw=1, ls="--", alpha=0.4)
    ax1.plot(xr, V, color=C_DARK, lw=2.0, label="$V(x)$")
    ax1.plot(xr, g, color=C_PURP, lw=2.0, ls="--", label=r"$g_3(x)=\operatorname{sign}(x)|x|^3$")
    ax1.plot([-1, 1], [0, 0], "o", color=C_GREEN, ms=6, zorder=5)
    ax1.text(
        0.0, 0.86, "information\nblind spot",
        fontsize=11, color=C_PURP, ha="center", va="center",
        fontstyle="italic", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_PURP, alpha=0.85)
    )
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("Normalised")
    ax1.set_title("(a) Potential and observation", fontsize=TITLE_FS, fontweight="bold")
    ax1.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    ax1.set_ylim(-1.1, 1.1)
    style_axes(ax1)

    for alpha, c, lbl, ls in [(3, C_PURP, r"$\alpha=3$", "-"), (1.5, C_TEAL, r"$\alpha=1.5$", "--"), (1, C_CTRL, r"$\alpha=1$", ":")]:
        ax2.plot(xr, np.clip(fisher_scalar(xr, alpha), 0, 500), color=c, lw=2, ls=ls, label=lbl)
    ax2.axvspan(-0.35, 0.35, alpha=0.06, color=C_PURP)
    ax2.axvline(0, color=C_PURP, lw=0.8, ls="--", alpha=0.3)
    ax2.text(0, 460, "blind spot", ha="center", fontsize=11, color=C_PURP, fontstyle="italic", alpha=0.75)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel(r"$\mathcal{I}(x)$")
    ax2.set_title("(b) Fisher information field", fontsize=TITLE_FS, fontweight="bold")
    ax2.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper right")
    ax2.set_ylim(-20, 520)
    style_axes(ax2)

    mu_r = np.linspace(0.01, 2.0, 300)
    for alpha, c, lbl, ls in [(3, C_PURP, r"$\alpha=3$", "-"), (2, C_DARK, r"$\alpha=2$", "-."), (1.5, C_TEAL, r"$\alpha=1.5$", "--"), (1, C_CTRL, r"$\alpha=1$", ":")]:
        fi_att = alpha ** 2 * mu_r ** (alpha - 1) / SE ** 2
        ax3.plot(mu_r, fi_att, color=c, lw=2.0, ls=ls, label=lbl)
    ax3.axhline(1, color=C_GREY, lw=0.8, ls="--", alpha=0.5)
    ax3.set_xlabel(r"$\mu$ (smaller values approach bifurcation)")
    ax3.set_ylabel(r"$\mathcal{I}(x^*(\mu))$")
    ax3.set_title("(c) Fisher at attractor", fontsize=TITLE_FS, fontweight="bold")
    ax3.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="lower right")
    ax3.set_yscale("log")
    ax3.set_ylim(1e-1, 1e5)
    style_axes(ax3)

    fig.savefig(out / "fig3_fisher_field.png")
    plt.close(fig)


def generate_fig4_hierarchy(out: Path):
    """Scalar hierarchy as an ensemble summary rather than one illustrative seed."""
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    rt = TIME[W:]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.2), sharex=True)
    titles = [("(a) Linear channel"), ("(b) Silent degradation"), ("(c) Variance collapse")]

    # Simulate the latent scalar pitchfork ensemble once, then reuse it across
    # all observation channels. Reusing the same latent paths isolates the
    # effect of the observation map from trajectory-selection artifacts.
    xs_ensemble = []
    mu_reference = None
    for seed in seeds:
        x_seed, mu_seed = simulate_scalar(seed=int(seed))
        xs_ensemble.append(x_seed)
        if mu_reference is None:
            mu_reference = mu_seed
    xs_ensemble = np.asarray(xs_ensemble)

    latent_rv = np.asarray([rolling_var(x_seed, W) for x_seed in xs_ensemble])
    latent_rv_s = np.asarray([smooth_edge(rv, 500) for rv in latent_rv])
    latent_med = np.median(latent_rv_s, axis=0)
    latent_lo, latent_hi = np.percentile(latent_rv_s, [10, 90], axis=0)

    for col, alpha in [(0, 1), (1, 1.5), (2, 3)]:
        col_color = C_PURP if alpha > 1 else C_CTRL
        xs_star = np.sqrt(np.clip(mu_reference, 0.001, None))
        fi_att = fisher_scalar(xs_star, alpha)

        # One observation-noise path per latent seed; the same noise convention
        # is used across channels to keep the comparison fair.
        obs_rv = []
        for seed, x_seed in zip(seeds, xs_ensemble):
            rng = np.random.RandomState(1000 + int(seed))
            eps = SE * rng.normal(size=N_STEPS)
            S = observe_scalar(x_seed, alpha) + eps
            obs_rv.append(rolling_var(S, W))
        obs_rv = np.asarray(obs_rv)
        obs_rv_s = np.asarray([smooth_edge(rv, 500) for rv in obs_rv])
        obs_med = np.median(obs_rv_s, axis=0)
        obs_lo, obs_hi = np.percentile(obs_rv_s, [10, 90], axis=0)

        ax = axes[0, col]
        ax.plot(TIME, smooth_edge(fi_att, 500), color=col_color, lw=2)
        ax.axhline(1, color=C_GREY, lw=0.6, ls="--", alpha=0.5)
        ax.set_title(titles[col], fontsize=TITLE_FS, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e5)
        if col == 0:
            ax.set_ylabel("Fisher at attractor\n$\\mathcal{I}(x^*)$")
        add_transition_line(ax, label=(col == 2))
        if col == 1:
            ax.text(
                0.50, 0.08, "model-implied reference",
                transform=ax.transAxes, fontsize=9.5, color="#666666", ha="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", alpha=0.90)
            )
        style_axes(ax)

        ax = axes[1, col]
        ax.fill_between(rt, latent_lo, latent_hi, color=C_GREEN, alpha=0.16, lw=0)
        ax.plot(rt, latent_med, color=C_GREEN, lw=2.2, label="ensemble median")
        if col == 0:
            ax.set_ylabel("Latent variance\nVar$[x]$")
            ax.legend(fontsize=10, framealpha=0.95, loc="upper left")
        add_transition_line(ax, label=False)
        style_axes(ax)

        ax = axes[2, col]
        ax.fill_between(rt, obs_lo, obs_hi, color=col_color, alpha=0.16, lw=0)
        ax.plot(rt, obs_med, color=col_color, lw=2.2, label="median; 10–90% band")
        if col == 0:
            ax.set_ylabel("Observed variance\nVar$[S]$")
        ax.set_xlabel("Time")
        add_transition_line(ax, label=False)

        n = len(obs_med)
        mid, end = n // 3, 2 * n // 3
        sl = np.polyfit(np.arange(mid, end), obs_med[mid:end], 1)[0]
        direction = "↓" if sl < 0 else "↑"
        ax.text(
            0.95, 0.90, direction, transform=ax.transAxes, fontsize=14, ha="right",
            color=C_PURP if sl < 0 else C_GREEN, fontweight="bold"
        )
        style_axes(ax)

    fig.savefig(out / "fig4_hierarchy.png")
    plt.close(fig)

def generate_fig5_channel(out: Path):
    """Figure 5: scalar channel dynamics as a 24-seed ensemble.

    Row 1 shows the realised channel strength along simulated latent paths,
    summarized by the ensemble median and 10--90% band, with the analytic
    attractor-level reference overlaid as a dashed curve. Row 2 shows the
    corresponding observed rolling variance across the same latent ensemble.
    """
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    mu_reference = np.linspace(MU0, MU_END, N_STEPS)
    xs_star = np.sqrt(np.clip(mu_reference, 0.001, None))
    rt = TIME[W:]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)

    for col, alpha, ttl in [(0, 3, "(a) Strong-collapse channel"), (1, 1.5, "(b) Moderate-collapse channel")]:
        c_col = C_PURP if alpha == 3 else C_TEAL
        gamma_paths = []
        rv_paths = []

        for seed in seeds:
            x_seed, _ = simulate_scalar(seed=int(seed))
            gamma_paths.append(smooth_edge(fisher_scalar(x_seed, alpha), 500))

            # Use a seed-specific observation-noise path so each latent run is
            # an independent replicate. This avoids reusing one lucky/noisy path.
            rng = np.random.RandomState(1200 + int(seed))
            S = observe_scalar(x_seed, alpha) + SE * rng.normal(size=N_STEPS)
            rv_paths.append(smooth_edge(rolling_var(S, W), 500))

        gamma_paths = np.asarray(gamma_paths)
        rv_paths = np.asarray(rv_paths)

        gamma_med = np.median(gamma_paths, axis=0)
        gamma_lo, gamma_hi = np.percentile(gamma_paths, [10, 90], axis=0)
        rv_med = np.median(rv_paths, axis=0)
        rv_lo, rv_hi = np.percentile(rv_paths, [10, 90], axis=0)

        Gamma_star = smooth_edge(fisher_scalar(xs_star, alpha), 500)

        ax = axes[0, col]
        ax.fill_between(TIME, gamma_lo, gamma_hi, color=c_col, alpha=0.14, lw=0)
        ax.plot(TIME, gamma_med, color=c_col, lw=2.2, label=r"$\Gamma_t$ ensemble median")
        ax.plot(TIME, Gamma_star, color=C_DARK, lw=2, ls="--", label=r"$\Gamma_t^*$ attractor ref.")
        ax.axhline(1, color=C_GREY, lw=0.6, ls="--", alpha=0.4)
        ax.set_ylabel("Channel strength")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e5)
        ax.set_title(ttl, fontsize=TITLE_FS, fontweight="bold")
        ax.legend(fontsize=LEGEND_FS, framealpha=0.95)
        add_transition_line(ax, label=(col == 0))
        style_axes(ax)

        ax = axes[1, col]
        ax.fill_between(rt, rv_lo, rv_hi, color=c_col, alpha=0.16, lw=0)
        ax.plot(rt, rv_med, color=c_col, lw=2.2, label="ensemble median")
        ax.set_ylabel("Rolling Var[$S$]")
        ax.set_xlabel("Time")
        add_transition_line(ax, label=False)
        ax.set_title("(c) Observed variance" if col == 0 else "(d) Observed variance", fontsize=TITLE_FS, fontweight="bold")
        if col == 0:
            ax.legend(fontsize=10, framealpha=0.95, loc="upper left")
        style_axes(ax)

    fig.savefig(out / "fig5_channel.png")
    plt.close(fig)

def generate_fig6_scalar_occupation_v2(out: Path):
    # Ensemble version: the stochastic occupation estimates are summarized
    # across 24 independently seeded latent trajectories. Panel c remains an
    # analytic/model-implied attractor-level reference.
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    rt = TIME[W:]
    fig, ax1, ax2, ax3 = fig_2plus1_canvas()

    # Simulate once and reuse paths for every threshold/exponent. This keeps
    # the comparison matched across panels and avoids unnecessary resimulation.
    x_ensemble = []
    for seed in seeds:
        x, _ = simulate_scalar(seed=int(seed))
        x_ensemble.append(x)
    x_ensemble = np.asarray(x_ensemble)

    def rolling_occupation(below: np.ndarray) -> np.ndarray:
        below = np.asarray(below, dtype=float)
        cs = np.cumsum(np.insert(below, 0, 0.0))
        frac = (cs[W:] - cs[:-W]) / W
        return smooth_edge(frac[1:], 200)

    def plot_occ_ensemble(ax, values, color, label, ls="-", lw=2.1, alpha_band=0.16):
        values = np.asarray(values, dtype=float)
        med = np.median(values, axis=0)
        lo, hi = np.percentile(values, [10, 90], axis=0)
        ax.fill_between(rt, lo, hi, color=color, alpha=alpha_band, linewidth=0)
        ax.plot(rt, med, color=color, lw=lw, ls=ls, label=label)
        return med, lo, hi

    # Panel a: for alpha=3, how often does the realised trajectory occupy
    # regions below two absolute Fisher-information thresholds?
    fi3_ens = fisher_scalar(x_ensemble, 3)
    for c_val, c_col, lbl, ls in [(1, C_PURP, r"$\mathcal{I}<1$", "-"), (10, C_ORANGE, r"$\mathcal{I}<10$", "--")]:
        occ = [rolling_occupation(fi3_ens[i] < c_val) for i in range(n_ensemble)]
        plot_occ_ensemble(ax1, occ, c_col, lbl, ls=ls)
    add_transition_line(ax1, y=0.70)
    ax1.set_title("(a) Blind-region occupation", fontsize=TITLE_FS, fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Fraction of window")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    style_axes(ax1)

    # Panel b: compare occupation across observation exponents using the same
    # ensemble seeds and the common threshold I < 1.
    for alpha, c_col, lbl, ls in [(3, C_PURP, r"$\alpha=3$", "-"), (1.5, C_TEAL, r"$\alpha=1.5$", "--"), (1, C_CTRL, r"$\alpha=1$", ":")]:
        fi = fisher_scalar(x_ensemble, alpha)
        occ = [rolling_occupation(fi[i] < 1) for i in range(n_ensemble)]
        plot_occ_ensemble(ax2, occ, c_col, lbl, ls=ls)
    add_transition_line(ax2, y=0.70)
    ax2.set_title("(b) Occupation across exponents", fontsize=TITLE_FS, fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"Fraction with $\mathcal{I}<1$")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    style_axes(ax2)

    # Panel c: deterministic reference: Fisher information at the attracting
    # branch as the branch migrates toward the low-information zone.
    mu = np.linspace(MU0, MU_END, N_STEPS)
    xs = np.sqrt(np.clip(mu, 1e-6, None))
    for alpha, c_col, lbl, ls in [(3, C_PURP, r"$\alpha=3$", "-"), (1.5, C_TEAL, r"$\alpha=1.5$", "--"), (1, C_CTRL, r"$\alpha=1$", ":")]:
        fi_att = fisher_scalar(xs, alpha)
        ax3.plot(TIME, fi_att / fi_att[0], color=c_col, lw=2.2, ls=ls, label=lbl)
    ax3.axhline(1e-2, color=C_PURP, lw=1.1, ls="--", alpha=0.45, label=r"dashed ref.: $10^{-2}$")
    ax3.axhline(5e-2, color=C_ORANGE, lw=1.1, ls="--", alpha=0.45, label=r"dashed ref.: $5\times 10^{-2}$")
    add_transition_line(ax3)
    ax3.set_yscale("log")
    ax3.set_ylim(1e-4, 1.5)
    ax3.set_title("(c) Attractor enters low-information zone", fontsize=TITLE_FS, fontweight="bold")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Relative attractor information")
    ax3.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="lower left")
    style_axes(ax3)

    fig.savefig(out / "fig6_scalar_occupation_v2.png")
    plt.close(fig)

def generate_fig7_ablation(out: Path):
    """Figure 7: scalar ablation as a matched 24-seed ensemble.

    The same latent scalar trajectories and the same observation-noise paths
    are reused across observation maps for each seed. This makes the ablation
    isolate the observation geometry rather than seed/noise idiosyncrasies.
    """
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    rt = TIME[W:]
    obs_maps = [
        (r"$x^3$ (collapse)", lambda x_: observe_scalar(x_, 3), lambda x_: fisher_scalar(x_, 3), C_PURP, "-"),
        (r"$\tanh(2x)$ (nonlinear control)", lambda x_: np.tanh(2 * x_), lambda x_: 4 / np.cosh(2 * x_) ** 4 / SE ** 2, C_TEAL, "--"),
        (r"$x$ (linear control)", lambda x_: x_, lambda x_: np.ones_like(x_) / SE ** 2, C_CTRL, ":"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)

    # Precompute matched latent trajectories and observation-noise paths. Each
    # seed is an independent replicate, but the same latent/noise realization is
    # used across channels within that replicate.
    latent_paths = []
    noise_paths = []
    for seed in seeds:
        x_seed, _ = simulate_scalar(seed=int(seed))
        rng = np.random.RandomState(1800 + int(seed))
        eps_seed = SE * rng.normal(size=N_STEPS)
        latent_paths.append(x_seed)
        noise_paths.append(eps_seed)
    latent_paths = np.asarray(latent_paths)
    noise_paths = np.asarray(noise_paths)

    for name, obs_fn, fi_fn, col, ls in obs_maps:
        gamma_paths = []
        rv_paths = []
        for x_seed, eps_seed in zip(latent_paths, noise_paths):
            gamma_paths.append(smooth_edge(fi_fn(x_seed), 500))
            S = obs_fn(x_seed) + eps_seed
            rv = rolling_var(S, W)
            # Normalize per replicate using a robust peak over the non-edge
            # region. This preserves the original peak-scaled ablation idea
            # while avoiding edge-dominated maxima.
            core = rv[500:-500] if len(rv) > 1000 else rv
            norm = np.nanpercentile(core, 99.0)
            if not np.isfinite(norm) or norm <= 0:
                norm = np.nanmax(core) if np.nanmax(core) > 0 else 1.0
            rv_paths.append(smooth_edge(rv / norm, 500))

        gamma_paths = np.asarray(gamma_paths)
        rv_paths = np.asarray(rv_paths)

        gamma_med = np.median(gamma_paths, axis=0)
        gamma_lo, gamma_hi = np.percentile(gamma_paths, [10, 90], axis=0)
        rv_med = np.median(rv_paths, axis=0)
        rv_lo, rv_hi = np.percentile(rv_paths, [10, 90], axis=0)

        axes[0].fill_between(TIME, gamma_lo, gamma_hi, color=col, alpha=0.14, linewidth=0)
        axes[0].plot(TIME, gamma_med, color=col, lw=2.2, ls=ls, label=name, alpha=0.90)
        axes[1].fill_between(rt, rv_lo, rv_hi, color=col, alpha=0.14, linewidth=0)
        axes[1].plot(rt, rv_med, color=col, lw=2.2, ls=ls, label=name, alpha=0.90)

    axes[0].axhline(1, color=C_GREY, lw=0.6, ls="--", alpha=0.4)
    axes[0].set_ylabel("Channel strength $\\Gamma_t$")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-2, 1e5)
    axes[0].set_title("(a) Matched ensemble channel trajectories", fontsize=TITLE_FS, fontweight="bold")
    axes[0].legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper right", bbox_to_anchor=(0.995, 0.50))
    add_transition_line(axes[0])
    style_axes(axes[0])

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Rolling Var[$S$] (peak-scaled)")
    axes[1].set_title("(b) Matched ensemble observed variance", fontsize=TITLE_FS, fontweight="bold")
    axes[1].legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    add_transition_line(axes[1], label=False)
    style_axes(axes[1])

    fig.savefig(out / "fig7_ablation.png")
    plt.close(fig)

def generate_fig8_channel_geometry(out: Path):
    """Anisotropic channel geometry with ensemble-supported occupation.

    Panel a is the only stochastic panel in this figure. It now shows early and
    late occupation across the same N=24 ensemble convention used elsewhere in
    the paper. Panels b-c are deterministic/model-implied channel-geometry
    diagnostics: the Fisher field and representative information ellipses.
    """
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    # Match Figure 9 exactly: same canvas size, constrained layout, spacing,
    # and panel proportions across the three horizontal panels.
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.45), constrained_layout=True)

    xr = np.linspace(-2, 2, 200)
    yr = np.linspace(-2, 2, 200)
    X1, X2 = np.meshgrid(xr, yr)
    V = -MU0 * X1 ** 2 / 2 + X1 ** 4 / 4 - NU * X2 ** 2 / 2 + X2 ** 4 / 4

    ax = axes[0]
    ax.contourf(X1, X2, V, levels=26, cmap="bone_r", alpha=0.48)
    ax.axvspan(-0.4, 0.4, alpha=0.12, color=C_PURP)
    ax.axvline(0, color=C_PURP, lw=1, ls="--", alpha=0.5)

    # Ensemble occupation evidence: early and late trajectory clouds.
    # Sparse, transparent samples replace the former single representative path.
    for seed in seeds:
        x1e, x2e, _, _ = simulate_2d(seed=int(seed))
        early = slice(0, N_STEPS // 3, 90)
        late = slice(2 * N_STEPS // 3, N_STEPS, 90)
        ax.plot(
            x1e[early], x2e[early], ".", color=C_TEAL, ms=1.35, alpha=0.13,
            label="early occupation" if seed == 0 else None,
            rasterized=True,
        )
        ax.plot(
            x1e[late], x2e[late], "x", color=C_ORANGE, ms=1.35, alpha=0.15,
            label="late occupation" if seed == 0 else None,
            rasterized=True,
        )

    for a in ATT_INIT:
        ax.plot(*a, "o", color=C_GREEN, ms=7)
    ax.annotate("", xy=(0, 1.5), xytext=(0, 0.55), arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2))
    ax.text(0.22, 1.04, "$x_2$ (bright)", color=C_GREEN, fontsize=11, fontweight="bold")
    ax.annotate("", xy=(1.18, 0), xytext=(0.45, 0), arrowprops=dict(arrowstyle="->", color=C_PURP, lw=2))
    ax.text(0.77, 0.18, "$x_1$ (dark)", color=C_PURP, fontsize=11, fontweight="bold")
    ax.text(
        0, -1.62, "dark zone\n($x_1\\approx 0$)", ha="center", fontsize=10.5,
        color=C_PURP, fontstyle="italic", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_PURP, alpha=0.86)
    )
    ax.legend(fontsize=9.3, framealpha=0.92, loc="upper left", markerscale=3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a) Ensemble occupation", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    style_axes(ax)

    ax = axes[1]
    lmn, _ = fisher_eigs(X1, X2)
    im = ax.pcolormesh(X1, X2, np.log10(np.clip(lmn, 1e-2, None)), cmap="magma", shading="auto", vmin=-2, vmax=5)
    ax.contour(
        X1, X2, np.log10(np.clip(lmn, 1e-2, None)),
        levels=[0, 1], colors=["white", "#cccccc"], linewidths=[1.5, 0.8], linestyles=["--", ":"]
    )
    for a in ATT_INIT:
        ax.plot(*a, "o", color=C_GREEN, ms=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.025)
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(r"(b) Model-implied $\log_{10}\lambda_{\min}$", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    style_axes(ax, grid=False)

    ax = axes[2]
    ax.axvspan(-0.4, 0.4, alpha=0.08, color=C_PURP)
    # Representative deterministic lattice of local information ellipses.
    # This avoids selecting points from any single stochastic run.
    pts = [
        (px, py)
        for px in [-1.35, -0.55, -0.15, 0.15, 0.55, 1.35]
        for py in [-1.15, 0.0, 1.15]
    ]
    for px, py in pts:
        lmn_p, lmx_p = fisher_eigs(np.array([px]), np.array([py]))
        lv = float(lmn_p[0])
        hv = float(lmx_p[0])
        sx = min(np.sqrt(lv), 80) / 80 * 0.24
        sy = min(np.sqrt(hv), 80) / 80 * 0.24
        col = C_PURP if lv < 0.5 * hv else C_GREEN
        ax.add_patch(Ellipse((px, py), width=2 * sx, height=2 * sy, angle=0, fc="none", ec=col, lw=1.45, alpha=0.82))
        if lv < 0.5 * hv:
            ax.annotate("", xy=(px + 0.12, py), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.65))
    for a in ATT_INIT:
        ax.plot(*a, "o", color=C_GREEN, ms=6)
    ax.text(
        0.03, 0.04, "deterministic\nellipse lattice",
        transform=ax.transAxes, fontsize=9.2, color="#555555", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#cccccc", alpha=0.92)
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(c) Information ellipses", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    style_axes(ax)

    fig.savefig(out / "fig8_channel_geometry.png")
    plt.close(fig)

def generate_fig9_directional_collapse(out: Path):
    """Figure 9: directional eigenvalue collapse with ensemble support.

    Panel a and panel c are deterministic/model-implied Fisher-geometry
    references. Panel b summarizes the realised collapsing eigenvalue across
    24 independently seeded two-dimensional trajectories, so the temporal
    degradation is not tied to a single seed.
    """
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    # Wider than the generic three-panel canvas: the log-axis labels and
    # legends in this figure otherwise crowd the panel titles.
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.45), constrained_layout=True)

    ax = axes[0]
    mu_r = np.linspace(0.05, 2.0, 200)
    lam_x1_att = ALPHA ** 2 * mu_r ** (ALPHA - 1) / SE ** 2
    lam_x2_att = np.ones_like(mu_r) / SE ** 2
    ax.plot(mu_r, lam_x2_att, color=C_GREEN, lw=2.3, ls="--", label=r"$\lambda_{x_2}^*$ — preserved")
    ax.plot(mu_r, lam_x1_att, color=C_PURP, lw=2.3, label=r"$\lambda_{x_1}^*$ — collapsing")
    ax.axhline(1, color=C_GREY, lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel(r"$\mu$ (smaller $\to$ bifurcation)")
    ax.set_ylabel("Channel quality at attractor")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e6)
    ax.invert_xaxis()
    ax.set_title("(a) Coordinate-specific information at attractor", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=10.2, framealpha=0.95, loc="upper right")
    style_axes(ax)

    ax = axes[1]
    lam_paths = []
    for seed in seeds:
        x1, x2, mu, nu = simulate_2d(seed=int(seed))
        lam_x1 = ALPHA ** 2 * np.abs(x1) ** (2 * (ALPHA - 1)) / SE ** 2
        lam_paths.append(smooth_edge(lam_x1, 800))
    lam_paths = np.asarray(lam_paths)
    lam_med = np.median(lam_paths, axis=0)
    lam_lo, lam_hi = np.percentile(lam_paths, [10, 90], axis=0)

    mu_ref = np.linspace(MU0, MU_END, N_STEPS)
    xs1 = np.sqrt(np.clip(mu_ref, 0.001, None))
    lam_att = ALPHA ** 2 * xs1 ** (2 * (ALPHA - 1)) / SE ** 2

    ax.fill_between(TIME, lam_lo, lam_hi, color=C_PURP, alpha=0.15, lw=0)
    ax.plot(TIME, lam_med, color=C_PURP, lw=2.3, label=r"realised $\lambda_{x_1}$ median")
    ax.plot(TIME, smooth_edge(lam_att, 800), color=C_DARK, lw=1.5, ls="--", label=r"attractor ref. $\lambda_{x_1}^*$")
    ax.axhline(1 / SE ** 2, color=C_GREEN, lw=1, ls=":", alpha=0.6, label=r"$\lambda_{x_2}=400$ (preserved)")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\lambda_{x_1}$ (collapsing direction)")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e5)
    ax.set_title("(b) Realised degradation across trajectories", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=9.0, framealpha=0.95, loc="upper right")
    style_axes(ax)

    ax = axes[2]
    xr = np.linspace(-2, 2, 100)
    yr = np.linspace(-2, 2, 100)
    X1g, X2g = np.meshgrid(xr, yr)
    lam_x1_g = ALPHA ** 2 * np.abs(X1g) ** (2 * (ALPHA - 1)) / SE ** 2
    im = ax.pcolormesh(X1g, X2g, np.log10(np.clip(lam_x1_g, 1e-2, None)), cmap="magma", shading="auto", vmin=-2, vmax=5)
    ax.contour(X1g, X2g, np.log10(np.clip(lam_x1_g, 1e-2, None)), levels=[np.log10(400)], colors=["white"], linewidths=1.5, linestyles="--")
    for a in ATT_INIT:
        ax.plot(*a, "o", color=C_GREEN, ms=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$\log_{10}\,\lambda_{x_1}$", fontsize=11)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(c) Collapsing-eigenvalue landscape", fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    style_axes(ax, grid=False)

    fig.savefig(out / "fig9_directional_collapse.png")
    plt.close(fig)

def generate_fig10_alignment(out: Path):
    """Figure 10: alignment comparison as a matched 24-seed ensemble.

    Panel a remains a deterministic geometry diagram. Panels b--c summarize
    the same ensemble of two-dimensional latent trajectories across the three
    channel orientations. Within each seed, the same observation-noise path is
    reused across orientations to isolate alignment rather than noise.
    """
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    rt = TIME[W:]
    angles = [
        (0.0, "Aligned ($\\phi=0$, on $x_1$)", C_PURP, "-"),
        (np.pi / 4, "Partial ($\\phi=\\pi/4$)", C_TEAL, "--"),
        (np.pi / 2, "Misaligned ($\\phi=\\pi/2$, on $x_2$)", C_CTRL, ":"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.45), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel a: deterministic orientation diagram.
    # ------------------------------------------------------------------
    ax = axes[0]
    xr = np.linspace(-2, 2, 80)
    yr = np.linspace(-2, 2, 80)
    X1g, X2g = np.meshgrid(xr, yr)
    V = -MU0 * X1g ** 2 / 2 + X1g ** 4 / 4 - NU * X2g ** 2 / 2 + X2g ** 4 / 4
    ax.contourf(X1g, X2g, V, levels=12, cmap="bone_r", alpha=0.18)
    for phi_val, _, col, _ in angles:
        c, s = np.cos(phi_val), np.sin(phi_val)
        ax.annotate("", xy=(1.2 * c, 1.2 * s), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
    ax.annotate("", xy=(1.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=2, ls="--"))
    ax.text(1.18, -0.42, "bifurcating\ndirection", color=C_ORANGE, fontsize=10.0, ha="center")
    for a in ATT_INIT:
        ax.plot(*a, "o", color=C_GREEN, ms=6)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_title("(a) Dark-direction alignment", fontsize=TITLE_FS, fontweight="bold")
    style_axes(ax)

    # Precompute the latent ensemble and matched noise paths.
    latent_ensemble = []
    noise_ensemble = []
    for seed in seeds:
        x1_seed, x2_seed, _, _ = simulate_2d(seed=int(seed))
        rng = np.random.RandomState(2400 + int(seed))
        eps_seed = SE * rng.normal(size=N_STEPS)
        latent_ensemble.append((x1_seed, x2_seed))
        noise_ensemble.append(eps_seed)

    # ------------------------------------------------------------------
    # Panel b: channel quality versus alignment, ensemble summary.
    # ------------------------------------------------------------------
    ax = axes[1]
    for phi_val, label, col, ls in angles:
        lmn_paths = []
        for x1_seed, x2_seed in latent_ensemble:
            lmn, _ = fisher_eigs_rotated(x1_seed, x2_seed, ALPHA, phi_val)
            lmn_paths.append(smooth_edge(lmn, 800))
        lmn_paths = np.asarray(lmn_paths)
        med = np.median(lmn_paths, axis=0)
        lo, hi = np.percentile(lmn_paths, [10, 90], axis=0)
        ax.fill_between(TIME, lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.plot(TIME, med, color=col, lw=2.2, ls=ls, label=label)
    ax.axhline(1, color=C_GREY, lw=0.6, ls="--", alpha=0.4)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e6)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\lambda_{\min}$ (channel quality)")
    ax.set_title("(b) Ensemble channel quality", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=9.2, framealpha=0.95, loc="lower left")
    style_axes(ax)

    # ------------------------------------------------------------------
    # Panel c: observable consequence, matched-noise ensemble summary.
    # ------------------------------------------------------------------
    ax = axes[2]
    for phi_val, label, col, ls in angles:
        c, s = np.cos(phi_val), np.sin(phi_val)
        rv_paths = []
        for (x1_seed, x2_seed), eps_seed in zip(latent_ensemble, noise_ensemble):
            xr_coord = c * x1_seed + s * x2_seed
            signal = np.sign(xr_coord) * np.abs(xr_coord) ** ALPHA
            rv = rolling_var(signal + eps_seed, W)
            rv_paths.append(smooth_edge(rv, 500))
        rv_paths = np.asarray(rv_paths)
        med = np.median(rv_paths, axis=0)
        lo, hi = np.percentile(rv_paths, [10, 90], axis=0)
        ax.fill_between(rt, lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.plot(rt, med, color=col, lw=2.2, ls=ls, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling Var[$S_2$]")
    ax.set_title("(c) Ensemble observable consequence", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=9.2, framealpha=0.95, loc="upper right")
    style_axes(ax)

    fig.savefig(out / "fig10_alignment.png")
    plt.close(fig)


def generate_fig11_anisotropic_occupation_v2(out: Path):
    rt = TIME[W:]
    fig, ax1, ax2, ax3 = fig_2plus1_canvas()
    bins = np.linspace(-2, 2, 50)
    n_third = N_STEPS // 3
    n_ensemble = 24
    seeds = np.arange(n_ensemble)

    # Simulate the ensemble once and reuse it across all panels.
    ensemble_paths = [simulate_2d(seed=int(seed))[:2] for seed in seeds]

    # Panels a-b: ensemble occupation densities rather than a single trajectory.
    for ax, sl, ttl in [
        (ax1, slice(0, n_third), "(a) Early-phase ensemble occupation"),
        (ax2, slice(2 * n_third, N_STEPS), "(b) Late-phase ensemble occupation"),
    ]:
        H_total = None
        for x1e, x2e in ensemble_paths:
            H, xe, ye = np.histogram2d(x1e[sl], x2e[sl], bins=[bins, bins])
            H_total = H if H_total is None else H_total + H
        H_total = H_total / max(H_total.sum(), 1.0)
        H_total = gaussian_filter(H_total, sigma=1)
        Xc = (xe[:-1] + xe[1:]) / 2
        Yc = (ye[:-1] + ye[1:]) / 2
        Xg, Yg = np.meshgrid(Xc, Yc)
        ax.pcolormesh(xe, ye, H_total.T, cmap="viridis", shading="auto")
        lmn_g, _ = fisher_eigs(Xg, Yg)
        ax.contour(
            Xg, Yg, np.log10(np.clip(lmn_g, 1e-2, None)),
            levels=[0], colors=[C_PURP], linewidths=1.5, linestyles="--"
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(ttl, fontsize=TITLE_FS, fontweight="bold")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        style_axes(ax, grid=False)

    # Panel c: median occupation fraction with 10--90% ensemble bands.
    for c_val, c_col, lbl, ls in [(1, C_PURP, r"$\lambda_{\min}<1$", "-"), (10, C_ORANGE, r"$\lambda_{\min}<10$", "--")]:
        frac_paths = []
        for x1e, x2e in ensemble_paths:
            lmne, _ = fisher_eigs(x1e, x2e)
            below = (lmne < c_val).astype(float)
            cs = np.cumsum(np.insert(below, 0, 0.0))
            frac = ((cs[W:] - cs[:-W]) / W)[1:]
            frac_paths.append(smooth_edge(frac, 500))
        frac_paths = np.asarray(frac_paths)
        med = np.median(frac_paths, axis=0)
        lo, hi = np.percentile(frac_paths, [10, 90], axis=0)
        ax3.fill_between(rt, lo, hi, color=c_col, alpha=0.14, linewidth=0)
        ax3.plot(rt, med, color=c_col, lw=2.2, ls=ls, label=lbl)

    add_transition_line(ax3)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("occupation fraction")
    ax3.set_title("(c) Ensemble occupation of dark regions", fontsize=TITLE_FS, fontweight="bold")
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    style_axes(ax3)

    fig.savefig(out / "fig11_anisotropic_occupation_v2.png")
    plt.close(fig)

def generate_fig12_masking(out: Path):
    """Directional masking, summarized over a matched 24-seed ensemble.

    The comparison is paired by seed: each latent trajectory is used for both
    the collapsing channel and the linear control, and the same observation
    noise paths are added to corresponding channels. This isolates the effect
    of the observation map rather than seed-to-seed or noise-draw variation.
    """
    rt = TIME[W:]
    n_ensemble = 24
    seeds = np.arange(n_ensemble)

    ratio_runs = []
    share_runs = []
    preserved_runs = []

    for seed in seeds:
        x1, x2, mu, nu = simulate_2d(seed=int(seed))

        # Matched observation noise: same noise paths for collapse and control.
        rng = np.random.RandomState(7000 + int(seed))
        eps_bright = SE * rng.normal(size=N_STEPS)
        eps_dark = SE * rng.normal(size=N_STEPS)

        s1_c, s2_c = G_collapse(x1, x2)
        s1_l, s2_l = G_linear(x1, x2)

        s1_c = s1_c + eps_bright
        s2_c = s2_c + eps_dark
        s1_l = s1_l + eps_bright
        s2_l = s2_l + eps_dark

        rv_s1_c = rolling_var(s1_c, W)
        rv_s2_c = rolling_var(s2_c, W)
        rv_s2_l = rolling_var(s2_l, W)

        ratio = rv_s2_c / np.clip(rv_s2_l, 1e-15, None)
        total = rv_s1_c + rv_s2_c
        share1 = rv_s1_c / np.clip(total, 1e-15, None)

        ratio_runs.append(smooth_edge(ratio, 800))
        share_runs.append(smooth_edge(share1, 500))
        preserved_runs.append(smooth_edge(rv_s1_c, 500))

    ratio_runs = np.asarray(ratio_runs)
    share_runs = np.asarray(share_runs)
    preserved_runs = np.asarray(preserved_runs)

    ratio_med = np.median(ratio_runs, axis=0)
    ratio_lo, ratio_hi = np.percentile(ratio_runs, [10, 90], axis=0)

    share_med = np.median(share_runs, axis=0)
    share_lo, share_hi = np.percentile(share_runs, [10, 90], axis=0)

    preserved_med = np.median(preserved_runs, axis=0)
    preserved_lo, preserved_hi = np.percentile(preserved_runs, [10, 90], axis=0)

    fig, ax1, ax2, ax3 = fig_2plus1_canvas()

    # ------------------------------------------------------------------
    # Panel a: suppression ratio relative to a matched linear control.
    # ------------------------------------------------------------------
    ax1.fill_between(
        rt, np.clip(ratio_lo, 1e-12, None), np.clip(ratio_hi, 1e-12, None),
        color=C_PURP, alpha=0.16, label="10--90% band"
    )
    ax1.plot(rt, np.clip(ratio_med, 1e-12, None), color=C_PURP, lw=2.4,
             label="ensemble median")
    ax1.axhline(1, color=C_GREY, lw=1, ls="--", alpha=0.5, label="No suppression")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"Var[$S_2$] / Var[$S_2^{\mathrm{linear}}$]")
    ax1.set_title("(a) Suppression relative to matched control", fontsize=TITLE_FS, fontweight="bold")
    ax1.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper right")
    ax1.set_yscale("log")
    style_axes(ax1)

    # ------------------------------------------------------------------
    # Panel b: aggregate masking by variance-share dominance.
    # ------------------------------------------------------------------
    ax2.fill_between(rt, share_lo, share_hi, color=C_GREEN, alpha=0.18,
                     label="10--90% band")
    ax2.plot(rt, share_med, color=C_GREEN, lw=2.5,
             label=r"preserved-channel share")
    ax2.axhline(0.5, color=C_GREY, lw=0.6, ls="--", alpha=0.45)
    ax2.text(
        0.04, 0.88, "preserved-direction share rises\nas the dark channel weakens",
        transform=ax2.transAxes, fontsize=10.2, color="#555555", va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.92)
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Preserved-channel variance share")
    ax2.set_ylim(0, 1)
    ax2.set_title("(b) Preserved-channel share", fontsize=TITLE_FS, fontweight="bold")
    ax2.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="lower right")
    style_axes(ax2)

    # ------------------------------------------------------------------
    # Panel c: preserved channel alone can remain active or increase.
    # ------------------------------------------------------------------
    ax3.fill_between(rt, preserved_lo, preserved_hi, color=C_GREEN, alpha=0.16,
                     label="10--90% band")
    ax3.plot(rt, preserved_med, color=C_GREEN, lw=2.5,
             label=r"Var[$S_1$] ensemble median")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Preserved-channel variance")
    ax3.set_title("(c) Preserved channel only", fontsize=TITLE_FS, fontweight="bold")
    ax3.legend(fontsize=LEGEND_FS, framealpha=0.95, loc="upper left")
    style_axes(ax3)

    fig.savefig(out / "fig12_masking.png")
    plt.close(fig)

def generate_fig13_diagnostics(out: Path):
    """Figure 13: diagnostic scores as a 24-seed ensemble.

    The directional and determinant diagnostics are computed for each latent
    trajectory relative to its own initial baseline. The aggregate-variance
    diagnostic uses matched noisy collapsed observations for the same seed.
    Curves are ensemble medians with 10--90% bands.
    """
    n_ensemble = 24
    seeds = np.arange(n_ensemble)
    rt = TIME[W:]

    lmin_scores = []
    det_scores = []
    agg_scores = []

    for seed in seeds:
        x1, x2, _, _ = simulate_2d(seed=int(seed))
        lam_min, lam_max = fisher_eigs(x1, x2)
        det_I = lam_min * lam_max

        # Baseline-relative Fisher diagnostics for this replicate.
        lmin_0 = np.mean(lam_min[:500])
        det_0 = np.mean(det_I[:500])
        c_lmin = np.log10(lmin_0 / np.clip(lam_min, 1e-10, None))
        c_det = np.log10(det_0 / np.clip(det_I, 1e-10, None))
        lmin_scores.append(smooth_edge(c_lmin, 800))
        det_scores.append(smooth_edge(c_det, 800))

        # Matched noisy observations for the aggregate-variance score.
        rng = np.random.RandomState(4100 + int(seed))
        s1, s2 = G_collapse(x1, x2)
        eps1 = SE * rng.normal(size=N_STEPS)
        eps2 = SE * rng.normal(size=N_STEPS)
        rv_agg = rolling_var(s1 + eps1, W) + rolling_var(s2 + eps2, W)
        rv_0 = np.nanmean(rv_agg[:500])
        rv_c = np.log10(rv_0 / np.clip(rv_agg, 1e-15, None))
        agg_scores.append(smooth_edge(rv_c, 500))

    lmin_scores = np.asarray(lmin_scores)
    det_scores = np.asarray(det_scores)
    agg_scores = np.asarray(agg_scores)

    lmin_med = np.median(lmin_scores, axis=0)
    lmin_lo, lmin_hi = np.percentile(lmin_scores, [10, 90], axis=0)
    det_med = np.median(det_scores, axis=0)
    det_lo, det_hi = np.percentile(det_scores, [10, 90], axis=0)
    agg_med = np.median(agg_scores, axis=0)
    agg_lo, agg_hi = np.percentile(agg_scores, [10, 90], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(12.2, 4.15))

    # Ensemble-level onset marker: first time the median directional score
    # exceeds the same 0.5 log10-decade threshold used in the original figure.
    t_dir = None
    hit = np.where(lmin_med > 0.5)[0]
    if len(hit):
        t_dir = TIME[int(hit[0])]
        ax.axvspan(t_dir, TIME[-1], alpha=0.10, color=C_PURP, zorder=0)
        # Place the annotation high and slightly left of the shaded-region midpoint
        # so it is centered visually without overlapping the median time series.
        label_x = t_dir + 0.34 * (TIME[-1] - t_dir)
        label_y = max(np.nanmax(lmin_hi) * 0.82, 4.20)
        ax.text(
            label_x, label_y,
            "directional score becomes prominent\nwhile aggregate variance remains mixed",
            ha="center", va="center", fontsize=10.7, color=C_PURP,
            fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_PURP, alpha=0.84),
            zorder=8
        )

    ax.fill_between(TIME, lmin_lo, lmin_hi, color=C_PURP, alpha=0.15, linewidth=0)
    ax.plot(TIME, lmin_med, color=C_PURP, lw=2.4, label="Directional")

    ax.fill_between(TIME, det_lo, det_hi, color=C_GREY, alpha=0.14, linewidth=0)
    ax.plot(TIME, det_med, color=C_GREY, lw=1.9, ls="--", label="Overall")

    ax.fill_between(rt, agg_lo, agg_hi, color=C_TEAL, alpha=0.14, linewidth=0)
    ax.plot(rt, agg_med, color=C_TEAL, lw=2.1, ls="-.", label="Aggregate")

    ax.axhline(0, color=C_GREY, lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Baseline-relative diagnostic score")
    ax.set_title("Ensemble directional diagnostics", fontsize=TITLE_FS, fontweight="bold")
    ax.legend(fontsize=10.0, framealpha=0.95, loc="upper left")
    style_axes(ax)

    fig.savefig(out / "fig13_diagnostics.png")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------
# Output names follow manuscript order: Figure 01 -> fig1_*, ..., Figure 13 -> fig13_*.

FIGURE_SPECS = [
    ("Figure 01", "Section 8 conceptual schematic", "fig1_engineering_schematic.png", generate_fig1_engineering_schematic),
    ("Figure 02", "Section 8 engineering policy", "fig2_engineering_policy.png", generate_fig2_engineering_policy),
    ("Figure 03", "Scalar Fisher field", "fig3_fisher_field.png", generate_fig3_fisher_field),
    ("Figure 04", "Scalar threshold hierarchy", "fig4_hierarchy.png", generate_fig4_hierarchy),
    ("Figure 05", "Scalar channel dynamics", "fig5_channel.png", generate_fig5_channel),
    ("Figure 06", "Scalar low-information occupation", "fig6_scalar_occupation_v2.png", generate_fig6_scalar_occupation_v2),
    ("Figure 07", "Scalar ablation", "fig7_ablation.png", generate_fig7_ablation),
    ("Figure 08", "Anisotropic channel geometry", "fig8_channel_geometry.png", generate_fig8_channel_geometry),
    ("Figure 09", "Directional eigenvalue collapse", "fig9_directional_collapse.png", generate_fig9_directional_collapse),
    ("Figure 10", "Alignment experiment", "fig10_alignment.png", generate_fig10_alignment),
    ("Figure 11", "Anisotropic occupation", "fig11_anisotropic_occupation_v2.png", generate_fig11_anisotropic_occupation_v2),
    ("Figure 12", "Directional masking", "fig12_masking.png", generate_fig12_masking),
    ("Figure 13", "Diagnostics", "fig13_diagnostics.png", generate_fig13_diagnostics),
]


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 13 figures for the anisotropic-collapse manuscript."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated PNG files (default: figures).",
    )
    parser.add_argument(
        "--figure",
        type=int,
        choices=range(1, 14),
        metavar="N",
        help="Generate only figure N, where N is between 1 and 13.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List figure numbers, descriptions, and output filenames.",
    )
    return parser.parse_args(argv)


def list_figures() -> None:
    for fig_no, description, filename, _ in FIGURE_SPECS:
        print(f"{fig_no}: {description} -> {filename}")


def generate_one(figure_number: int, output_dir: Path) -> None:
    _, description, filename, generator = FIGURE_SPECS[figure_number - 1]
    print(f"Generating Figure {figure_number:02d}: {description} [{filename}]", flush=True)
    generator(output_dir)
    plt.close("all")
    gc.collect()


def validate_outputs(output_dir: Path) -> None:
    missing = [filename for _, _, filename, _ in FIGURE_SPECS if not (output_dir / filename).exists()]
    if missing:
        raise RuntimeError("Missing generated figures: " + ", ".join(missing))

    generated = sorted(
        path.name for path in output_dir.glob("*.png")
        if path.name in {spec[2] for spec in FIGURE_SPECS}
    )
    if len(generated) != len(FIGURE_SPECS):
        raise RuntimeError(
            f"Expected {len(FIGURE_SPECS)} manuscript figures, found {len(generated)}."
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        list_figures()
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.figure is not None:
        generate_one(args.figure, output_dir)
        return

    if len(FIGURE_SPECS) != 13:
        raise RuntimeError(f"Expected 13 figure specifications, found {len(FIGURE_SPECS)}.")

    print(f"Generating all 13 figures in: {output_dir}\n")

    # Each figure is generated in a fresh process to bound peak memory use.
    script = Path(__file__).resolve()
    for figure_number in range(1, len(FIGURE_SPECS) + 1):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--output-dir",
                str(output_dir),
                "--figure",
                str(figure_number),
            ],
            check=True,
        )

    validate_outputs(output_dir)
    print("\nAll 13 manuscript figures were generated successfully.")


if __name__ == "__main__":
    main()
