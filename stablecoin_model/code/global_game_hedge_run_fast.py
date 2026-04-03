import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.special import erf

# Fast screening version of the global-game hedge-run model.
# Purpose:
# - preserve the same economic structure as the fully nested solver,
# - run quickly enough for calibration screening and sign checks,
# - provide a practical quantitative layer before moving to the full solver.
#
# Differences from the fully nested benchmark:
# - coarser theta and posterior grids,
# - fewer Gauss-Hermite nodes for integrating over the public signal,
# - shorter fixed-point iterations,
# - intended for calibration search and sign verification rather than final tables.


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def norm_pdf(x, mu=0.0, sigma=1.0):
    x = np.asarray(x)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def norm_cdf(x, mu=0.0, sigma=1.0):
    x = np.asarray(x)
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


theta_grid = np.linspace(0.0, 1.4, 11)
theta_prior_grid = np.linspace(-0.1, 1.7, 21)
gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(3)

BASE = {
    "mu_theta": 0.5,
    "sigma_theta": 0.35,
    "sigma_x": 0.70,
    "sigma_s_bar": 0.90,
    "rho": 8.0,
    "alpha": 0.82,
    "delta": 0.10,
    "chi_h": 0.95,
    "eta_h": 0.70,
    "rbar": 0.22,
    "B": 0.06,
    "kappa_r": 0.12,
    "lambda_loss": 2.5,
    "xi_coord": 1.5,
    "a_c": 0.04,
    "b_c": 0.08,
    "a_s": 0.02,
    "b_s": 0.05,
    "kappa_c": 0.14,
    "kappa_s": 0.04,
    "nu_share": 7.5,
    "phi_share": 1.2,
    "zeta": 0.8,
    "run_weight_normal": 0.25,
}


def info_weight(Qs, p):
    prec_theta = 1.0 / p["sigma_theta"] ** 2
    prec_x = 1.0 / p["sigma_x"] ** 2
    sigma_s2 = p["sigma_s_bar"] ** 2 / (1.0 + p["rho"] * max(Qs, 0.0))
    prec_s = 1.0 / sigma_s2
    omega0 = prec_theta / (prec_theta + prec_x + prec_s)
    omegax = prec_x / (prec_theta + prec_x + prec_s)
    omegas = prec_s / (prec_theta + prec_x + prec_s)
    return omega0, omegax, omegas, sigma_s2


def costs(Qc, Qs, p):
    c_c = p["a_c"] + p["b_c"] * np.sqrt(max(Qc, 0.0)) + p["kappa_c"]
    c_s = p["a_s"] + p["b_s"] * np.sqrt(max(Qs, 0.0)) + p["kappa_s"]
    return c_c, c_s


def hedge_demand_fast(theta, s, Qs, Qc, allow_stablecoins, p):
    c_c, c_s = costs(Qc, Qs, p)
    omega0, omegax, omegas, sigma_s2 = info_weight(Qs if allow_stablecoins else 0.0, p)
    omega_s = omegas if allow_stablecoins else 0.0
    share_s = logistic(p["nu_share"] * (c_c - c_s) + p["phi_share"] * omega_s) if allow_stablecoins else 0.0
    eff_cost = (1.0 - share_s) * c_c + share_s * c_s

    mu_m = omega0 * p["mu_theta"] + omegax * theta + omegas * s
    sig_m = abs(omegax) * p["sigma_x"]
    a = p["chi_h"]
    b = p["eta_h"] * eff_cost
    mu_y = a * mu_m - b
    sig_y = max(a * sig_m, 1e-10)
    z = mu_y / sig_y
    q_h = sig_y * norm_pdf(z) + mu_y * norm_cdf(z)
    return float(q_h), float(share_s), float(eff_cost), float(omega_s), float(sigma_s2)


def posterior_weights_theta_fast(xstar, s, Qs, p):
    _, _, _, sigma_s2 = info_weight(Qs, p)
    prior = norm_pdf(theta_prior_grid, p["mu_theta"], p["sigma_theta"])
    like_x = norm_pdf(xstar, theta_prior_grid, p["sigma_x"])
    like_s = norm_pdf(s, theta_prior_grid, np.sqrt(sigma_s2))
    w = prior * like_x * like_s
    w = np.maximum(w, 1e-300)
    w /= np.sum(w)
    return w


def expected_marginal_gain_fast(xstar, s, Qs, Qc, allow_stablecoins, p):
    Qs_eff = Qs if allow_stablecoins else 0.0
    w = posterior_weights_theta_fast(xstar, s, Qs_eff, p)
    q_h_arr = np.empty_like(theta_prior_grid)
    omega_s_arr = np.empty_like(theta_prior_grid)
    for idx, th in enumerate(theta_prior_grid):
        q_h_arr[idx], _, _, omega_s_arr[idx], _ = hedge_demand_fast(th, s, Qs, Qc, allow_stablecoins, p)
    q_run_arr = p["rbar"] * (1.0 - norm_cdf(xstar, theta_prior_grid, p["sigma_x"]))
    q_fx_arr = q_h_arr + q_run_arr
    crisis_arr = (q_fx_arr >= (p["alpha"] - p["delta"] * theta_prior_grid)).astype(float)
    loss_arr = p["lambda_loss"] * np.maximum(theta_prior_grid, 0.0) * (1.0 + p["xi_coord"] * omega_s_arr)
    return float(p["B"] - p["kappa_r"] + np.sum(w * crisis_arr * loss_arr))


def solve_cutoff_fast(s, Qs, Qc, allow_stablecoins, p):
    lo, hi = -1.0, 2.5
    h_lo = expected_marginal_gain_fast(lo, s, Qs, Qc, allow_stablecoins, p)
    h_hi = expected_marginal_gain_fast(hi, s, Qs, Qc, allow_stablecoins, p)
    if h_lo >= 0 and h_hi >= 0:
        return lo
    if h_lo <= 0 and h_hi <= 0:
        return hi
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        h_mid = expected_marginal_gain_fast(mid, s, Qs, Qc, allow_stablecoins, p)
        if h_mid >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_state_fast(theta, s, allow_stablecoins, p, max_iter=35, damp=0.40, tol=1e-6):
    Qc = 0.10
    Qs = 0.06 if allow_stablecoins else 0.0
    xstar = 0.5
    for _ in range(max_iter):
        if not allow_stablecoins:
            Qs = 0.0
        xstar_new = solve_cutoff_fast(s, Qs, Qc, allow_stablecoins, p)
        q_h, share_s, eff_cost, omega_s, sigma_s2 = hedge_demand_fast(theta, s, Qs, Qc, allow_stablecoins, p)
        q_run = float(p["rbar"] * (1.0 - norm_cdf(xstar_new, theta, p["sigma_x"])))
        q_fx = q_h + q_run
        Qs_new = share_s * q_fx if allow_stablecoins else 0.0
        Qc_new = q_fx - Qs_new
        diff = max(abs(Qs_new - Qs), abs(Qc_new - Qc), abs(xstar_new - xstar))
        Qs = (1 - damp) * Qs + damp * Qs_new
        Qc = (1 - damp) * Qc + damp * Qc_new
        xstar = (1 - damp) * xstar + damp * xstar_new
        if diff < tol:
            break

    q_h, share_s, eff_cost, omega_s, sigma_s2 = hedge_demand_fast(theta, s, Qs, Qc, allow_stablecoins, p)
    q_run = float(p["rbar"] * (1.0 - norm_cdf(xstar, theta, p["sigma_x"])))
    q_fx = q_h + q_run
    crisis = 1.0 if q_fx >= (p["alpha"] - p["delta"] * theta) else 0.0

    c_normal = 1.0 + p["zeta"] * np.log1p(q_h + p["run_weight_normal"] * q_run) - eff_cost * q_fx
    crisis_loss = p["lambda_loss"] * theta * (1.0 + p["xi_coord"] * omega_s) * (1.0 + q_run)
    c_normal = max(c_normal, 1e-6)
    c_crisis = max(c_normal - crisis_loss, 1e-6)
    welfare = (1.0 - crisis) * np.log(c_normal) + crisis * np.log(c_crisis)

    return {
        "Qc": Qc,
        "Qs": Qs,
        "Qfx": q_fx,
        "Qhedge": q_h,
        "Qrun": q_run,
        "xstar": xstar,
        "share_s": share_s,
        "omega_s": omega_s,
        "sigma_s2": sigma_s2,
        "crisis": crisis,
        "welfare": welfare,
    }


def integrate_over_public_signal_fast(theta, allow_stablecoins, p, max_outer=10, damp=0.5, tol=1e-6):
    sigma_s2_agg = p["sigma_s_bar"] ** 2
    avg_Qs = 0.0
    weights = None
    states = None
    for _ in range(max_outer):
        sigma_s = np.sqrt(sigma_s2_agg)
        s_nodes = theta + np.sqrt(2.0) * sigma_s * gh_nodes
        w = gh_weights / np.sqrt(np.pi)
        w = np.maximum(w, 1e-300)
        w /= np.sum(w)
        new_states = [solve_state_fast(theta, s, allow_stablecoins, p) for s in s_nodes]
        new_avg_Qs = float(np.sum([wi * st["Qs"] for wi, st in zip(w, new_states)]))
        sigma_s2_new = p["sigma_s_bar"] ** 2 / (1.0 + p["rho"] * (new_avg_Qs if allow_stablecoins else 0.0))
        diff = max(abs(new_avg_Qs - avg_Qs), abs(sigma_s2_new - sigma_s2_agg))
        avg_Qs = (1 - damp) * avg_Qs + damp * new_avg_Qs
        sigma_s2_agg = (1 - damp) * sigma_s2_agg + damp * sigma_s2_new
        weights = w
        states = new_states
        if diff < tol:
            break
    out = {}
    for k in states[0].keys():
        out[k] = float(np.sum([wi * st[k] for wi, st in zip(weights, states)]))
    out["sigma_s2_agg"] = sigma_s2_agg
    return out


def profiles_fast(p=None):
    if p is None:
        p = BASE
    keys = ["Qc", "Qs", "Qfx", "Qhedge", "Qrun", "xstar", "share_s", "omega_s", "sigma_s2", "crisis", "welfare", "sigma_s2_agg"]
    cash = {k: [] for k in keys}
    full = {k: [] for k in keys}
    for theta in theta_grid:
        rc = integrate_over_public_signal_fast(theta, False, p)
        rf = integrate_over_public_signal_fast(theta, True, p)
        for k in keys:
            cash[k].append(rc[k])
            full[k].append(rf[k])
    cash = {k: np.array(v) for k, v in cash.items()}
    full = {k: np.array(v) for k, v in full.items()}
    return cash, full


def run(output_dir="../../output_fast_global_game"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cash, full = profiles_fast(BASE)
    welfare_diff = full["welfare"] - cash["welfare"]

    summary = pd.DataFrame({
        "metric": [
            "avg crisis cash-only",
            "avg crisis stablecoins",
            "monotone stablecoin demand",
            "monotone total FX demand",
            "max welfare gain low-mid theta",
            "min welfare diff high theta",
        ],
        "value": [
            float(np.mean(cash["crisis"])),
            float(np.mean(full["crisis"])),
            float(np.mean(np.diff(full["Qs"]) >= -1e-10)),
            float(np.mean(np.diff(full["Qfx"]) >= -1e-10)),
            float(np.max(welfare_diff[:4])),
            float(np.min(welfare_diff[-3:])),
        ],
    })
    summary.to_csv(output_dir / "summary_fast_global_game.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, full["Qs"], linewidth=2, label="Stablecoin demand")
    plt.plot(theta_grid, full["Qc"], linewidth=2, label="Cash demand")
    plt.plot(theta_grid, full["Qfx"], linewidth=2, label="Total FX demand")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Demand")
    plt.title("Fast global-game hedge-run model: FX demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_fx_demand_fast.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, cash["crisis"], linewidth=2, label="Cash-only")
    plt.plot(theta_grid, full["crisis"], linewidth=2, label="With stablecoins")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Crisis probability")
    plt.title("Fast global-game hedge-run model: crises")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_crisis_fast.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, welfare_diff, linewidth=2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Welfare difference")
    plt.title("Fast global-game hedge-run model: welfare")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_welfare_fast.png", dpi=220)
    plt.close()

    return summary


if __name__ == "__main__":
    print(run())
