# Stablecoin Parallel Market Model

This folder contains a standalone implementation of the corrected microfounded model used for the overvalued-exchange-rate paper draft.

## Contents
- `corrected_monotonic_model.py`: solves the cash-only and stablecoin economies on the overvaluation region `theta >= 0`.

## Key modeling choices
- Domestic-currency utility is microfounded as
  `U_D(x,s) = -lambda_theta * E[theta | x,s] - Delta * pi(x,s)`.
- Cash and stablecoins are safe foreign-currency instruments with premia and access costs:
  `U_C = -p_C - kappa_C`, `U_S = -p_S - kappa_S`.
- Crisis risk uses a **smooth crisis-probability mapping** instead of a hard threshold, which avoids artificial saturation and removes the inverted-U pattern in FX demand.
- The code focuses on the **overvaluation region only** and targets an **interior-risk calibration** in both the cash-only and stablecoin regimes.

## Intended outputs
The script generates:
- monotonic asset-allocation curves,
- information-channel curves,
- interior crisis probabilities in both regimes,
- welfare-difference curves.

The script writes PNG figures and CSV summary statistics into a local `outputs/` folder when run.
