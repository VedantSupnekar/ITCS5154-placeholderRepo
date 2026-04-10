# lsmc_gradient_boosting.py
# LSMC American put pricing using Gradient Boosting for the continuation value
# sequential ensemble of shallow trees. each tree corrects the residuals of the
# previous one. generally better bias variance tradeoff than random forest for
# smooth continuation value surfaces, but more sensitive to hyperparams.
#
# Reference: Longstaff & Schwartz (2001)

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def lsmc_american_put_gb(S0, K, T, r, sigma, N, paths,
                         n_estimators=100, max_depth=3, learning_rate=0.1,
                         seed=None):
   
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    # simulate GBM paths
    S = np.zeros((N + 1, paths))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # terminal payoff
    V = np.maximum(K - S[-1], 0)
    df = np.exp(-r * dt)

    # backward induction
    for t in range(N - 1, 0, -1):
        V = V * df

        St = S[t]
        itm_idx = np.where(St < K)[0]

        if len(itm_idx) == 0:
            continue

        X_itm = St[itm_idx].reshape(-1, 1)  # GBR expects 2D input
        Y_itm = V[itm_idx]

        # gradient boosting to estimate continuation value
        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        gbr.fit(X_itm, Y_itm)
        cont_val = gbr.predict(X_itm)

        # exercise if immediate payoff > expected continuation
        exercise_val = K - St[itm_idx]
        exercise_idx = itm_idx[exercise_val > cont_val]
        V[exercise_idx] = K - St[exercise_idx]

    return np.mean(V * df)


if __name__ == "__main__":
    # same params as Longstaff & Schwartz (2001), Table 1
    # expected price ~4.47
    S0    = 36.0
    K     = 40.0
    T     = 1.0
    r     = 0.06
    sigma = 0.2
    N     = 50

    print("LSMC American Put w/ Gradient Boosting regression")
    print(f"S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, N={N}\n")

    # GB is slower than linear methods but faster than RF at similar accuracy
    # testing with fewer paths first
    print(f"{'paths':>8}  {'n_est':>6}  {'lr':>6}  {'depth':>6}  {'price':>8}")
    print("-" * 45)
    for paths, n_est, lr, depth in [
        (5_000,   50, 0.1, 3),
        (10_000,  50, 0.1, 3),
        (10_000, 100, 0.1, 3),
        (10_000, 100, 0.05, 4),
    ]:
        price = lsmc_american_put_gb(S0, K, T, r, sigma, N, paths,
                                     n_estimators=n_est, max_depth=depth,
                                     learning_rate=lr, seed=42)
        print(f"{paths:>8,}  {n_est:>6}  {lr:>6.2f}  {depth:>6}  {price:>8.4f}")

    print("\npaper benchmark ~4.47")
