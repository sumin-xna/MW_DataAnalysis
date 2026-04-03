import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from math import erf, sqrt

# Global-game hedge-run prototype
# The hedge block remains smooth and fundamentals-driven.
# The run block is determined by a cutoff equilibrium.

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def norm_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / (sigma * np.sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


theta_grid = np.linspace(0.0, 1.4, 51)
x_grid = np.linspace(-0.6, 2.2, 181)

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
    "tau_belief": 0.06,
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


def posterior_mean(x, s, Qs, p):
    omega0, omegax, omegas, _ = info_weight(Qs, p)
    return omega0 * p["mu_theta"] + omegax * x + omegas * s


def prices(Qc, Qs, p):
    p_c = p["a_c"] + p["b_c"] * np.sqrt(max(Qc, 0.0))
    p_s = p["a_s"] + p["b_s"] * np.sqrt(max(Qs, 0.0))
    c_c = p_c + p["kappa_c"]
    c_s = p_s + p["kappa_s"]
    return p_c, p_s, c_c, c_s


def aggregate_hedge(theta, s, Qs, Qc, p):
    dx = x_grid[1] - x_grid[0]
    fx = norm_pdf(x_grid, theta, p["sigma_x"])
    fx /= np.sum(fx) * dx
    _, _, c_c, c_s = prices(Qc, Qs, p)
    omega0, omegax, omegas, _ = info_weight(Qs, p)
    omega_s = omegas
    share_s = logistic(p["nu_share"] * (c_c - c_s) + p["phi_share"] * omega_s)
    eff_cost = (1.0 - share_s) * c_c + share_s * c_s
    m = omega0 * p["mu_theta"] + omegax * x_grid + omegas * s
    q_h = np.maximum(0.0, p["chi_h"] * m - p["eta_h"] * eff_cost)
    return float(np.sum(q_h * fx) * dx), share_s, eff_cost, omega_s


def perceived_crisis_prob(xstar, s, Qs, Qc, p):
    # Plug-in belief at the marginal runner based on the posterior mean.
    mstar = posterior_mean(xstar, s, Qs, p)
    q_h_hat, _, _, omega_s = aggregate_hedge(mstar, s, Qs, Qc, p)
    q_run_hat = p["rbar"] * (1.0 - norm_cdf(xstar, mstar, p["sigma_x"]))
    q_fx_hat = q_h_hat + q_run_hat
    threshold_hat = p["alpha"] - p["delta"] * mstar
    pi_hat = logistic((q_fx_hat - threshold_hat) / p["tau_belief"])
    loss_hat = p["lambda_loss"] * max(mstar, 0.0) * (1.0 + p["xi_coord"] * omega_s)
    return pi_hat, loss_hat


def solve_cutoff(s, Qs, Qc, p):
    # Solve the marginal-runner indifference condition by bisection.
    lo, hi = -1.0, 2.5

    def H(x):
        pi_hat, loss_hat = perceived_crisis_prob(x, s, Qs, Qc, p)
        return p["B"] - p["kappa_r"] + pi_hat * loss_hat

    h_lo, h_hi = H(lo), H(hi)
    if h_lo >= 0.0 and h_hi >= 0.0:
        return lo
    if h_lo <= 0.0 and h_hi <= 0.0:
        return hi

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        h_mid = H(mid)
        if h_mid >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_state(theta, s, allow_stablecoins=True, p=None, max_iter=200, damp=0.25, tol=1e-8):
    if p is None:
        p = BASE
    Qc = 0.10
    Qs = 0.06 if allow_stablecoins else 0.0
    xstar = 0.5

    for _ in range(max_iter):
        if not allow_stablecoins:
            Qs = 0.0
        xstar_new = solve_cutoff(s, Qs, Qc, p)
        q_h, share_s, eff_cost, omega_s = aggregate_hedge(theta, s, Qs, Qc, p)
        q_run = p["rbar"] * (1.0 - norm_cdf(xstar_new, theta, p["sigma_x"]))
        q_fx = q_h + q_run

        if allow_stablecoins:
            Qs_new = share_s * q_fx
        else:
            Qs_new = 0.0
        Qc_new = q_fx - Qs_new

        diff = max(abs(Qs_new - Qs), abs(Qc_new - Qc), abs(xstar_new - xstar))
        Qs = (1.0 - damp) * Qs + damp * Qs_new
        Qc = (1.0 - damp) * Qc + damp * Qc_new
        xstar = (1.0 - damp) * xstar + damp * xstar_new
        if diff < tol:
            break

    q_h, share_s, eff_cost, omega_s = aggregate_hedge(theta, s, Qs, Qc, p)
    q_run = p["rbar"] * (1.0 - norm_cdf(xstar, theta, p["sigma_x"]))
    q_fx = q_h + q_run
    crisis = 1.0 if q_fx >= (p["alpha"] - p["delta"] * theta) else 0.0

    c_normal = 1.0 + p["zeta"] * np.log1p(q_h + p["run_weight_normal"] * q_run) - eff_cost * q_fx
    crisis_loss = p["lambda_loss"] * theta * (1.0 + p["xi_coord"] * omega_s) * (1.0 + q_run)
    c_crisis = max(c_normal - crisis_loss, 1e-6)
    c_normal = max(c_normal, 1e-6)
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
        "sigma_s2": info_weight(Qs, p)[3],
        "crisis": crisis,
        "welfare": welfare,
    }


def profiles(p=None):
    if p is None:
        p = BASE
    out_cash = {k: [] for k in ["Qc", "Qs", "Qfx", "Qhedge", "Qrun", "xstar", "share_s", "omega_s", "sigma_s2", "crisis", "welfare"]}
    out_full = {k: [] for k in ["Qc", "Qs", "Qfx", "Qhedge", "Qrun", "xstar", "share_s", "omega_s", "sigma_s2", "crisis", "welfare"]}
    for theta in theta_grid:
        s = theta
        rc = solve_state(theta, s, False, p)
        rf = solve_state(theta, s, True, p)
        for k in out_cash:
            out_cash[k].append(rc[k])
            out_full[k].append(rf[k])
    out_cash = {k: np.array(v) for k, v in out_cash.items()}
    out_full = {k: np.array(v) for k, v in out_full.items()}
    return out_cash, out_full


def run(output_dir="../../output_global_game"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cash, full = profiles(BASE)
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
            float(np.max(welfare_diff[:20])),
            float(np.min(welfare_diff[-15:])),
        ],
    })
    summary.to_csv(output_dir / "summary_global_game.csv", index=False)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, full["Qs"], linewidth=2, label="Stablecoin demand")
    plt.plot(theta_grid, full["Qc"], linewidth=2, label="Cash demand")
    plt.plot(theta_grid, full["Qfx"], linewidth=2, label="Total FX demand")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Demand")
    plt.title("Global-game hedge-run model: FX demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_fx_demand_global_game.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, cash["crisis"], linewidth=2, label="Cash-only")
    plt.plot(theta_grid, full["crisis"], linewidth=2, label="With stablecoins")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Crisis indicator / probability")
    plt.title("Global-game hedge-run model: crises")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_crisis_global_game.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, welfare_diff, linewidth=2)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Welfare difference")
    plt.title("Global-game hedge-run model: welfare")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_welfare_global_game.png", dpi=220)
    plt.close()

    return summary


if __name__ == "__main__":
    run()
