import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------
# Baseline grids: overvaluation region only
# ------------------------------------------------------------

theta_grid = np.linspace(0.0, 1.4, 71)
x_grid = np.linspace(-0.3, 1.9, 81)
s_grid = np.linspace(0.0, 1.4, 31)


# ------------------------------------------------------------
# Calibrated parameters: monotone demand + interior crisis risk
# ------------------------------------------------------------

params = {
    "mu_theta": 0.55,
    "sigma_theta": 0.30,
    "sigma_x": 0.70,
    "sigma_s_bar": 0.85,
    "rho": 7.0,
    "alpha": 0.94,
    "delta": 0.18,
    "a_c": 0.03,
    "b_c": 0.10,
    "a_s": 0.015,
    "b_s": 0.06,
    "kappa_c": 0.16,
    "kappa_s": 0.04,
    "Delta": 0.40,
    "mu_choice": 0.18,
    "lambda_theta": 0.35,
    "tau_crisis": 0.10,
}


# ------------------------------------------------------------
# Cash-only model
# ------------------------------------------------------------

def solve_cash(theta_grid=theta_grid, x_grid=x_grid, p=params, max_iter=120, damp=0.22, tol=1e-5):
    dtheta = theta_grid[1] - theta_grid[0]
    dx = x_grid[1] - x_grid[0]

    prior = norm_pdf((theta_grid - p["mu_theta"]) / p["sigma_theta"]) / p["sigma_theta"]
    prior /= prior.sum() * dtheta

    fx_theta = np.array([
        norm_pdf((x_grid - th) / p["sigma_x"]) / p["sigma_x"]
        for th in theta_grid
    ])

    prec_theta = 1.0 / p["sigma_theta"] ** 2
    prec_x = 1.0 / p["sigma_x"] ** 2
    omega0 = prec_theta / (prec_theta + prec_x)
    omegax = prec_x / (prec_theta + prec_x)
    post_mean_x = omega0 * p["mu_theta"] + omegax * x_grid
    post_var = 1.0 / (prec_theta + prec_x)

    p_c = p["a_c"] + 0.02
    pi_x = np.full(len(x_grid), 0.10)

    for _ in range(max_iter):
        U_D = -p["lambda_theta"] * post_mean_x - p["Delta"] * pi_x
        U_C = np.full(len(x_grid), -p_c - p["kappa_c"])

        vmax = np.maximum(U_D, U_C)
        eD = np.exp((U_D - vmax) / p["mu_choice"])
        eC = np.exp((U_C - vmax) / p["mu_choice"])
        P_C = eC / (eD + eC)

        Qc_theta = fx_theta @ (P_C * dx)
        threshold = p["alpha"] - p["delta"] * theta_grid
        crisis_theta = logistic((Qc_theta - threshold) / p["tau_crisis"])

        numer = prior[:, None] * fx_theta
        denom = numer.sum(axis=0) * dtheta
        posterior = numer / denom
        pi_new = (crisis_theta[:, None] * posterior).sum(axis=0) * dtheta

        p_c_new = p["a_c"] + p["b_c"] * (Qc_theta @ prior * dtheta)
        diff = max(np.max(np.abs(pi_new - pi_x)), abs(p_c_new - p_c))
        pi_x = (1 - damp) * pi_x + damp * pi_new
        p_c = (1 - damp) * p_c + damp * p_c_new
        if diff < tol:
            break

    return {
        "prior": prior,
        "p_c": p_c,
        "pi_x": pi_x,
        "Qc_theta": Qc_theta,
        "crisis_theta": crisis_theta,
        "post_mean_x": post_mean_x,
        "post_var": post_var,
    }


# ------------------------------------------------------------
# Full model with stablecoins
# ------------------------------------------------------------

def solve_full(theta_grid=theta_grid, x_grid=x_grid, s_grid=s_grid, p=params, max_iter=140, damp=0.20, tol=1e-5):
    dtheta = theta_grid[1] - theta_grid[0]
    dx = x_grid[1] - x_grid[0]

    prior = norm_pdf((theta_grid - p["mu_theta"]) / p["sigma_theta"]) / p["sigma_theta"]
    prior /= prior.sum() * dtheta

    fx_theta = np.array([
        norm_pdf((x_grid - th) / p["sigma_x"]) / p["sigma_x"]
        for th in theta_grid
    ])

    p_c_s = np.full(len(s_grid), p["a_c"] + 0.02)
    p_s_s = np.full(len(s_grid), p["a_s"] + 0.015)
    sigma_s2_s = np.full(len(s_grid), p["sigma_s_bar"] ** 2)
    pi_xs = np.full((len(x_grid), len(s_grid)), 0.10)

    post_mean_xs = np.zeros((len(x_grid), len(s_grid)))
    omega_s_s = np.zeros(len(s_grid))
    post_var_s = np.zeros(len(s_grid))

    for _ in range(max_iter):
        new_p_c_s = np.zeros_like(p_c_s)
        new_p_s_s = np.zeros_like(p_s_s)
        new_sigma_s2_s = np.zeros_like(sigma_s2_s)
        new_pi_xs = np.zeros_like(pi_xs)
        new_post_mean_xs = np.zeros_like(post_mean_xs)
        new_omega_s_s = np.zeros_like(omega_s_s)
        new_post_var_s = np.zeros_like(post_var_s)

        Qc_theta_s = np.zeros((len(theta_grid), len(s_grid)))
        Qs_theta_s = np.zeros((len(theta_grid), len(s_grid)))
        crisis_theta_s = np.zeros((len(theta_grid), len(s_grid)))

        for js, s in enumerate(s_grid):
            prec_theta = 1.0 / p["sigma_theta"] ** 2
            prec_x = 1.0 / p["sigma_x"] ** 2
            prec_s = 1.0 / sigma_s2_s[js]
            denom_prec = prec_theta + prec_x + prec_s

            omega0 = prec_theta / denom_prec
            omegax = prec_x / denom_prec
            omegas = prec_s / denom_prec

            new_post_mean_xs[:, js] = omega0 * p["mu_theta"] + omegax * x_grid + omegas * s
            new_omega_s_s[js] = omegas
            new_post_var_s[js] = 1.0 / denom_prec

            U_D = -p["lambda_theta"] * new_post_mean_xs[:, js] - p["Delta"] * pi_xs[:, js]
            U_C = np.full(len(x_grid), -p_c_s[js] - p["kappa_c"])
            U_S = np.full(len(x_grid), -p_s_s[js] - p["kappa_s"])

            vmax = np.maximum.reduce([U_D, U_C, U_S])
            eD = np.exp((U_D - vmax) / p["mu_choice"])
            eC = np.exp((U_C - vmax) / p["mu_choice"])
            eS = np.exp((U_S - vmax) / p["mu_choice"])
            denom = eD + eC + eS

            P_C = eC / denom
            P_S = eS / denom

            Qc_theta = fx_theta @ (P_C * dx)
            Qs_theta = fx_theta @ (P_S * dx)
            Qc_theta_s[:, js] = Qc_theta
            Qs_theta_s[:, js] = Qs_theta

            threshold = p["alpha"] - p["delta"] * theta_grid
            crisis_theta = logistic(((Qc_theta + Qs_theta) - threshold) / p["tau_crisis"])
            crisis_theta_s[:, js] = crisis_theta

            fs_theta = norm_pdf((s - theta_grid) / np.sqrt(sigma_s2_s[js])) / np.sqrt(sigma_s2_s[js])
            post_theta_given_s = prior * fs_theta
            post_theta_given_s /= post_theta_given_s.sum() * dtheta

            numer = (prior * fs_theta)[:, None] * fx_theta
            denom_x = numer.sum(axis=0) * dtheta
            post_theta_given_xs = numer / denom_x[None, :]
            new_pi_xs[:, js] = (crisis_theta[:, None] * post_theta_given_xs).sum(axis=0) * dtheta

            EQc = (Qc_theta * post_theta_given_s).sum() * dtheta
            EQs = (Qs_theta * post_theta_given_s).sum() * dtheta
            new_p_c_s[js] = p["a_c"] + p["b_c"] * EQc
            new_p_s_s[js] = p["a_s"] + p["b_s"] * EQs
            new_sigma_s2_s[js] = p["sigma_s_bar"] ** 2 / (1.0 + p["rho"] * EQs)

        diff = max(
            np.max(np.abs(new_p_c_s - p_c_s)),
            np.max(np.abs(new_p_s_s - p_s_s)),
            np.max(np.abs(new_sigma_s2_s - sigma_s2_s)),
            np.max(np.abs(new_pi_xs - pi_xs)),
        )
        p_c_s = (1 - damp) * p_c_s + damp * new_p_c_s
        p_s_s = (1 - damp) * p_s_s + damp * new_p_s_s
        sigma_s2_s = (1 - damp) * sigma_s2_s + damp * new_sigma_s2_s
        pi_xs = (1 - damp) * pi_xs + damp * new_pi_xs
        post_mean_xs = (1 - damp) * post_mean_xs + damp * new_post_mean_xs
        omega_s_s = (1 - damp) * omega_s_s + damp * new_omega_s_s
        post_var_s = (1 - damp) * post_var_s + damp * new_post_var_s
        if diff < tol:
            break

    return {
        "prior": prior,
        "p_c_s": p_c_s,
        "p_s_s": p_s_s,
        "sigma_s2_s": sigma_s2_s,
        "pi_xs": pi_xs,
        "post_mean_xs": post_mean_xs,
        "omega_s_s": omega_s_s,
        "post_var_s": post_var_s,
        "Qc_theta_s": Qc_theta_s,
        "Qs_theta_s": Qs_theta_s,
        "crisis_theta_s": crisis_theta_s,
    }


# ------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------

def weights_over_s_given_theta(theta_idx, theta_grid, s_grid, full):
    like = norm_pdf((s_grid - theta_grid[theta_idx]) / np.sqrt(full["sigma_s2_s"])) / np.sqrt(full["sigma_s2_s"])
    return like / like.sum()


def compute_profiles(cash, full):
    Qs_full = np.zeros(len(theta_grid))
    Qc_full = np.zeros(len(theta_grid))
    Qfx_full = np.zeros(len(theta_grid))
    crisis_full = np.zeros(len(theta_grid))
    omegaS = np.zeros(len(theta_grid))
    postvar_full = np.zeros(len(theta_grid))

    for i in range(len(theta_grid)):
        w = weights_over_s_given_theta(i, theta_grid, s_grid, full)
        Qs_full[i] = full["Qs_theta_s"][i, :] @ w
        Qc_full[i] = full["Qc_theta_s"][i, :] @ w
        Qfx_full[i] = Qs_full[i] + Qc_full[i]
        crisis_full[i] = full["crisis_theta_s"][i, :] @ w
        omegaS[i] = full["omega_s_s"] @ w
        postvar_full[i] = full["post_var_s"] @ w

    postvar_cash = np.full(len(theta_grid), cash["post_var"])
    return Qs_full, Qc_full, Qfx_full, crisis_full, omegaS, postvar_cash, postvar_full


# ------------------------------------------------------------
# Main routine: generate figures
# ------------------------------------------------------------

def make_figures(output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cash = solve_cash()
    full = solve_full()
    Qs_full, Qc_full, Qfx_full, crisis_full, omegaS, postvar_cash, postvar_full = compute_profiles(cash, full)

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, Qs_full, linewidth=2, label="Stablecoin demand")
    plt.plot(theta_grid, Qc_full, linewidth=2, label="Cash demand")
    plt.plot(theta_grid, Qfx_full, linewidth=2, label="Total FX demand")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Demand share")
    plt.title("Asset allocation in the overvaluation region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_asset_allocation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, 1 / postvar_cash, linewidth=2, label="Cash-only posterior precision")
    plt.plot(theta_grid, 1 / postvar_full, linewidth=2, label="With stablecoins posterior precision")
    plt.plot(theta_grid, omegaS, linewidth=2, label="Weight on public signal")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Precision / weight")
    plt.title("Information channel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_information.png", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cash_map = np.tile(cash["crisis_theta"], (len(s_grid), 1))
    im0 = axes[0].imshow(cash_map, origin="lower", aspect="auto",
                         extent=[theta_grid[0], theta_grid[-1], s_grid[0], s_grid[-1]])
    axes[0].set_title("Cash-only")
    axes[0].set_xlabel("Overvaluation theta")
    axes[0].set_ylabel("Public signal s")
    im1 = axes[1].imshow(full["crisis_theta_s"].T, origin="lower", aspect="auto",
                         extent=[theta_grid[0], theta_grid[-1], s_grid[0], s_grid[-1]])
    axes[1].set_title("With stablecoins")
    axes[1].set_xlabel("Overvaluation theta")
    axes[1].set_ylabel("Public signal s")
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Crisis probability")
    plt.suptitle("Crisis regions (interior in both regimes)")
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_crisis_regions.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(theta_grid, cash["crisis_theta"], linewidth=2, label="Cash-only")
    plt.plot(theta_grid, crisis_full, linewidth=2, label="With stablecoins")
    plt.xlabel("Overvaluation theta")
    plt.ylabel("Crisis probability")
    plt.title("Crisis probabilities by state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_crisis_prob.png", dpi=220)
    plt.close()

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    make_figures()
