#!/usr/bin/env python3
from __future__ import annotations

"""Independent numerical audit for the Fisher-collapse figures.

Run from the manuscript directory:
    python validate_numerics.py

The script reproduces the convergence and ensemble-stability numbers reported
in the Numerical details appendix. It does not regenerate the manuscript plots.
"""

import numpy as np

import simulation_code as g


def coupled_timestep_audit() -> tuple[float, float]:
    mu_values = np.geomspace(1.0, 1.0e-3, 14)
    n_rep = 32
    sigma0, s_exp = 0.20, 1.25
    dt_coarse, dt_fine = 0.01, 0.005
    burn, keep, stride = 5000, 18000, 4
    rng = np.random.default_rng(4242)
    vc, vf, gc, gf = [], [], [], []

    for mu in mu_values:
        z0 = np.ones(n_rep) + 0.01 * rng.normal(size=n_rep)
        zc, zf = z0.copy(), z0.copy()
        amp = sigma0 * mu ** (s_exp - 1.0)
        for _ in range(burn):
            n1, n2 = rng.normal(size=n_rep), rng.normal(size=n_rep)
            zc += (zc - zc**3) * dt_coarse + amp * np.sqrt(dt_fine) * (n1 + n2)
            zf += (zf - zf**3) * dt_fine + amp * np.sqrt(dt_fine) * n1
            zf += (zf - zf**3) * dt_fine + amp * np.sqrt(dt_fine) * n2
        sample_c, sample_f = [], []
        for k in range(keep):
            n1, n2 = rng.normal(size=n_rep), rng.normal(size=n_rep)
            zc += (zc - zc**3) * dt_coarse + amp * np.sqrt(dt_fine) * (n1 + n2)
            zf += (zf - zf**3) * dt_fine + amp * np.sqrt(dt_fine) * n1
            zf += (zf - zf**3) * dt_fine + amp * np.sqrt(dt_fine) * n2
            if k % stride == 0:
                sample_c.append(np.sqrt(mu) * zc.copy())
                sample_f.append(np.sqrt(mu) * zf.copy())
        xc, xf = np.asarray(sample_c), np.asarray(sample_f)
        vc.append(np.var(xc, axis=0)); vf.append(np.var(xf, axis=0))
        gc.append(np.var(xc**3, axis=0)); gf.append(np.var(xf**3, axis=0))

    def maximum_median_change(coarse: list[np.ndarray], fine: list[np.ndarray]) -> float:
        mc = np.median(np.asarray(coarse).T, axis=0)
        mf = np.median(np.asarray(fine).T, axis=0)
        return float(np.max(np.abs(mc / mf - 1.0)))

    return maximum_median_change(vc, vf), maximum_median_change(gc, gf)


def localized_ensemble_audit() -> dict[str, float]:
    mu = np.geomspace(1.0, 1.0e-3, 14)
    sigma0, s_exp = 0.20, 1.25
    var_ou = sigma0**2 * mu ** (2 * s_exp - 1) / 4.0
    var_g3 = 9.0 * mu**2 * var_ou

    base = g.localized_pitchfork_statistics(mu, n_rep=32, seed=1441)
    m32x = np.median(base["var_x"], axis=0)
    m16x = np.median(base["var_x"][:16], axis=0)
    m32g = np.median(base["var_g3"], axis=0)
    m16g = np.median(base["var_g3"][:16], axis=0)
    subset_change = max(np.max(np.abs(m16x / m32x - 1)), np.max(np.abs(m16g / m32g - 1)))

    sigma_max = 0.0
    for sigma, seed in [(0.12, 1561), (0.20, 1641), (0.28, 1721)]:
        stats = g.localized_pitchfork_statistics(mu, n_rep=32, sigma0=sigma, seed=seed)
        vx = sigma**2 * mu ** (2 * s_exp - 1) / 4.0
        vg = 9.0 * mu**2 * vx
        sigma_max = max(
            sigma_max,
            float(np.max(np.abs(np.median(stats["var_x"], axis=0)[-6:] / vx[-6:] - 1))),
            float(np.max(np.abs(np.median(stats["var_g3"], axis=0)[-6:] / vg[-6:] - 1))),
        )

    med_x, med_g = [], []
    for seed in [1441, 2441, 3441, 4441]:
        stats = g.localized_pitchfork_statistics(mu, n_rep=32, seed=seed)
        med_x.append(np.median(stats["var_x"], axis=0) / var_ou)
        med_g.append(np.median(stats["var_g3"], axis=0) / var_g3)
    med_x, med_g = np.asarray(med_x), np.asarray(med_g)

    def spread(a: np.ndarray) -> np.ndarray:
        return (np.max(a, axis=0) - np.min(a, axis=0)) / np.median(a, axis=0)

    full_spread = max(float(np.max(spread(med_x))), float(np.max(spread(med_g))))
    near_spread = max(float(np.max(spread(med_x)[-6:])), float(np.max(spread(med_g)[-6:])))
    return {
        "subset_change": float(subset_change),
        "sigma_max_near": float(sigma_max),
        "seed_spread_full": full_spread,
        "seed_spread_near": near_spread,
    }


def lyapunov_audit() -> float:
    residuals = []
    for mu in np.geomspace(0.8, 0.005, 180):
        v = g.rotating_model(float(mu))
        A, Sigma, D = np.asarray(v["A"]), np.asarray(v["Sigma"]), np.asarray(v["D"])
        residuals.append(np.linalg.norm(A @ Sigma + Sigma @ A.T + D, 2) / np.linalg.norm(D, 2))
    return float(max(residuals))


def finite_history_audit() -> tuple[float, float, float]:
    mu = np.geomspace(0.8, 0.02, 18)
    a = 2.0 * mu
    fisher = 1.5**2 * mu**0.5
    horizons = [np.ones_like(mu), 5.0 / a]
    continuous = [
        fisher * (1 - np.exp(-2 * a * horizons[0])) / (2 * a),
        fisher * (1 - np.exp(-2 * a * horizons[1])) / (2 * a),
    ]
    rng = np.random.default_rng(7007)
    max_error = 0.0
    max_discretization = 0.0
    covered = 0
    total = 0
    for horizon, target_info in zip(horizons, continuous):
        mse = np.empty((20, len(mu)))
        discrete_info = np.empty(len(mu))
        for j, (aj, fj, hj) in enumerate(zip(a, fisher, horizon)):
            mse[:, j], discrete_info[j] = g._gls_mse_from_synthetic_records(
                float(aj), float(fj), float(hj), rng,
                n_batches=20, trials_per_batch=400, n_time=160,
            )
        target = 1 / target_info
        median = np.median(mse, axis=0)
        low, high = np.percentile(mse, [10, 90], axis=0)
        max_error = max(max_error, float(np.max(np.abs(median / target - 1))))
        max_discretization = max(
            max_discretization,
            float(np.max(np.abs(discrete_info / target_info - 1))),
        )
        covered += int(np.sum((low <= target) & (target <= high)))
        total += len(mu)
    return max_error, covered / total, max_discretization


def fixed_noise_seed_audit() -> dict[str, tuple[float, float] | float]:
    n_steps, window = 30000, 1500
    time = np.arange(n_steps) * 0.01
    time_roll = time[window - 1 :]
    maps = {"linear": lambda x: x, "bounded": lambda x: np.tanh(2 * x), "cubic": lambda x: x**3}
    slopes = {key: [] for key in maps}

    for base in [3000, 4000, 5000, 6000]:
        variances = {key: [] for key in maps}
        for offset in range(24):
            x, _ = g.simulate_fixed_noise(base + offset)
            rng = np.random.default_rng(9000 + base + offset)
            for key, fn in maps.items():
                eps = g.SE * rng.normal(size=n_steps)
                variances[key].append(g.rolling_var(fn(x) + eps, window))
        medians = {key: np.median(np.asarray(values), axis=0) for key, values in variances.items()}
        fit = slice(int(0.1 * len(time_roll)), int(0.8 * len(time_roll)))
        for key, curve in medians.items():
            slopes[key].append(float(np.polyfit(time_roll[fit], np.log(np.maximum(curve[fit], 1e-12)), 1)[0]))

    return {
        key: (min(values), max(values)) for key, values in slopes.items()
    }


def main() -> None:
    dt_x, dt_g = coupled_timestep_audit()
    localized = localized_ensemble_audit()
    residual = lyapunov_audit()
    history_error, history_coverage, history_discretization = finite_history_audit()
    fixed = fixed_noise_seed_audit()

    print(f"Coupled timestep change, latent: {100*dt_x:.3f}%")
    print(f"Coupled timestep change, cubic:  {100*dt_g:.3f}%")
    print(f"Noise-amplitude max deviation near bifurcation: {100*localized['sigma_max_near']:.3f}%")
    print(f"16-vs-32 trajectory median change: {100*localized['subset_change']:.3f}%")
    print(f"Four-seed full-grid spread: {100*localized['seed_spread_full']:.3f}%")
    print(f"Four-seed near-grid spread: {100*localized['seed_spread_near']:.3f}%")
    print(f"Maximum normalized Lyapunov residual: {residual:.3e}")
    print(f"Finite-history batch-median discrepancy: {100*history_error:.3f}%")
    print(f"CR bound coverage by 10-90% batch bands: {100*history_coverage:.1f}%")
    print(f"Finite-history midpoint information error: {100*history_discretization:.4f}%")
    print(f"Fixed-noise slope ranges: {fixed['linear']}, {fixed['bounded']}, {fixed['cubic']}")

    assert dt_x < 0.006 and dt_g < 0.0061
    assert localized["sigma_max_near"] < 0.05
    assert localized["subset_change"] < 0.035
    assert localized["seed_spread_full"] < 0.060
    assert localized["seed_spread_near"] < 0.044
    assert residual < 6e-14
    assert history_error < 0.08
    assert history_coverage >= 0.90
    assert history_discretization < 0.001
    assert fixed["linear"][0] > 0 and fixed["bounded"][0] > 0 and fixed["cubic"][1] < 0
    print("All numerical audit assertions passed.")


if __name__ == "__main__":
    main()
