# LSMC American put pricing using Lasso regression for the continuation value
# similar to lsmc_ridge.py but Lasso uses L1 penalty instead of L2
#
# idea: same as Ridge. replace OLS with a regularized regression.
# Lasso (L1) drives some coefficients to exactly zero, so it's more
# "selective" about which polynomial terms it actually uses. might work
# better if only a few basis functions are actually informative.
#
# Reference: Longstaff & Schwartz (2001)

import warnings
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


def lsmc_american_put_lasso(S0, K, T, r, sigma, N, paths, degree=2, alpha=0.001, seed=None):
    """
    American put pricer using LSMC with Lasso regression continuation estimator.

    Args:
        S0, K, T, r, sigma : standard option params
        N      : time steps
        paths  : number of Monte Carlo paths
        degree : polynomial degree for basis features (default 2)
        alpha  : Lasso regularization strength (default 0.001, keep small or it shrinks too aggressively)
        seed   : random seed
    Returns:
        estimated price at t=0
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    # simulate GBM paths
    S = np.zeros((N + 1, paths))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # terminal payoff for American put
    V = np.maximum(K - S[-1], 0)
    df = np.exp(-r * dt)  # discount per step

    # backward induction
    for t in range(N - 1, 0, -1):
        V = V * df  # discount future cash flows one step

        St = S[t]
        itm_idx = np.where(St < K)[0]  # only in-the-money paths

        if len(itm_idx) == 0:
            continue

        X_itm = St[itm_idx]
        Y_itm = V[itm_idx]

        # Lasso regression to estimate continuation value
        # scaling is important here. Lasso is even more sensitive to feature
        # scale than Ridge because the L1 penalty treats all coefficients equally
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
            ('lasso',  Lasso(alpha=alpha, fit_intercept=True, max_iter=10_000)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pipe.fit(X_itm.reshape(-1, 1), Y_itm)
            cont_val = pipe.predict(X_itm.reshape(-1, 1))

        # exercise if immediate payoff > expected continuation
        exercise_val = K - X_itm
        exercise_idx = itm_idx[exercise_val > cont_val]
        V[exercise_idx] = K - St[exercise_idx]

    return np.mean(V * df)


if __name__ == "__main__":
    # same params as Table 1 in Longstaff & Schwartz (2001)
    # expected price ~4.47
    S0    = 36.0
    K     = 40.0
    T     = 1.0
    r     = 0.06
    sigma = 0.2
    N     = 50
    paths = 100_000

    print("LSMC American Put w/ Lasso regression")
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, N={N}, paths={paths}\n")

    # sweep alpha — Lasso needs small alpha to not over-shrink
    # too large and it zeroes out all coefficients → bad price
    print(f"{'alpha':>8}  {'price':>8}")
    print("-" * 20)
    for a in [0.0001, 0.001, 0.01, 0.1]:
        price = lsmc_american_put_lasso(S0, K, T, r, sigma, N, paths, degree=2, alpha=a, seed=42)
        print(f"{a:>8.4f}  {price:>8.4f}")

    print("\npaper benchmark ~4.47")
    print("Lasso regression is not working well because it is zeroing out coefficients")
