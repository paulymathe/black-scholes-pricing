import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# --- European Option Pricing (Black-Scholes Formula) ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Monte Carlo Pricing for European Options ---
def monte_carlo_price(S, K, T, r, sigma, N=100_000, option_type='call'):
    Z = np.random.normal(size=N)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    return np.exp(-r*T) * np.mean(payoff)

# --- Greeks Calculation ---
def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return delta, gamma, vega, theta, rho

# --- American Option Pricing (Binomial Tree) ---
def binomial_american_option(S, K, T, r, sigma, steps=100, option_type='call'):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)

    option_values = np.zeros_like(prices)
    for j in range(steps + 1):
        if option_type == 'call':
            option_values[j, steps] = max(prices[j, steps] - K, 0)
        else:
            option_values[j, steps] = max(K - prices[j, steps], 0)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = np.exp(-r * dt) * (p * option_values[j, i+1] + (1 - p) * option_values[j+1, i+1])
            if option_type == 'call':
                exercise = max(prices[j, i] - K, 0)
            else:
                exercise = max(K - prices[j, i], 0)
            option_values[j, i] = max(hold, exercise)

    return option_values[0, 0]

# --- Asian Option Pricing (Arithmetic Mean via Monte Carlo) ---
def asian_option_monte_carlo(S, K, T, r, sigma, steps=100, N=100_000, option_type='call'):
    dt = T / steps
    payoffs = []
    for _ in range(N):
        prices = [S]
        for _ in range(steps):
            z = np.random.normal()
            S_t = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            prices.append(S_t)
        A = np.mean(prices[1:])  # moyenne arithmétique
        payoff = max(A - K, 0) if option_type == 'call' else max(K - A, 0)
        payoffs.append(payoff)
    return np.exp(-r*T) * np.mean(payoffs)

# --- Implied Volatility Estimation ---
def implied_volatility(option_market_price, S, K, T, r, option_type='call'):
    func = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - option_market_price
    try:
        return brentq(func, 1e-6, 5.0)
    except ValueError:
        return np.nan
