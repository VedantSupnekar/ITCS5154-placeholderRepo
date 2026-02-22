import numpy as np

def lsmc_american_put(S0, K, T, r, sigma, N, paths, degree=2, seed=42):
    np.random.seed(seed)
    dt = T / N
    
    # Simulate GBM paths
    S = np.zeros((N + 1, paths))
    S[0] = S0
    for t in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
    # Payoff at maturity (American Put)
    payoff = np.maximum(K - S[-1], 0)
    
    # Cash flows vector
    V = payoff.copy()
    
    # Discount factor per step
    df = np.exp(-r * dt)
    
    # Stepping backwards through time
    for t in range(N - 1, 0, -1):
        # Discounting the cash flows
        V = V * df
        
        St = S[t]
        
        # In-the-money (ITM) paths only
        itm_idx = np.where(St < K)[0]
        
        if len(itm_idx) > 0:
            # Immediate exercise value for ITM paths
            exercise_val = K - St[itm_idx]
            
            # predict continuation value
            # Regress discounted future cash flows
            X = St[itm_idx]
            Y = V[itm_idx]
            
            # simple OLS polynomial regression
            coeffs = np.polyfit(X, Y, degree)
            cont_val = np.polyval(coeffs, X)
            
            # Exercise decision
            exercise_idx = itm_idx[exercise_val > cont_val]
            
            # Update cash flows for paths where we exercise
            V[exercise_idx] = K - St[exercise_idx]
            
    # Discount back to t=0 and average
    V0 = np.mean(V * df)
    
    return V0

if __name__ == "__main__":
    S0 = 36.0
    K = 40.0
    T = 1.0
    r = 0.06
    sigma = 0.2
    N = 50
    paths = 100000

    print("running lsmc put pricing...")
    price = lsmc_american_put(S0, K, T, r, sigma, N, paths, degree=2, seed=42)
    print(f"simulated LSMC price: {price:.4f}")
    # paper value for these params is around ~4.47
