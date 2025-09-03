# Install required libraries (run in Jupyter or terminal if needed)
# !pip install yfinance numpy pandas matplotlib seaborn arch

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.optimize import minimize
import os

# Set matplotlib style for professional plots
try:
    plt.style.use('seaborn-v0_8')  # Use Seaborn's legacy style
except:
    plt.style.use('ggplot')  # Fallback to Matplotlib's ggplot style

# Create directory for saving plots
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# 1. Download stock data (20 major stocks)
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "MA",
    "WMT", "PG", "KO", "PEP", "XOM", "CVX", "JNJ", "PFE", "BAC", "C"
]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31", auto_adjust=True)["Close"]

# Check data integrity
print("First 5 rows of data:\n", data.head())
print("\nMissing values:\n", data.isna().sum())

# Handle missing values (if any) by forward filling
data = data.fillna(method="ffill").dropna()

# 2. Calculate daily returns
returns = data.pct_change().dropna()

# 3. Optimize portfolio weights using Sharpe Ratio
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate=0.01):
    port_returns = returns.dot(weights)
    port_mean = port_returns.mean() * 252  # Annualize returns
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_mean - risk_free_rate) / port_std
    return -sharpe  # Minimize negative Sharpe Ratio

cov_matrix = returns.cov()
initial_weights = np.array([1/len(tickers)] * len(tickers))  # Equal weights as initial guess
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights = 1
bounds = tuple((0, 1) for _ in range(len(tickers)))
opt_result = minimize(portfolio_performance, initial_weights, args=(returns, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x
print("\nOptimal Portfolio Weights:", dict(zip(tickers, np.round(optimal_weights, 4))))

# Calculate portfolio returns with optimized weights
port_returns = returns.dot(optimal_weights)

# 4. Historical VaR (1-day and 5-day)
VaR_95_hist_1d = np.percentile(port_returns, 5)
VaR_99_hist_1d = np.percentile(port_returns, 1)
port_returns_5d = returns.rolling(window=5).sum().dropna().dot(optimal_weights)
VaR_95_hist_5d = np.percentile(port_returns_5d, 5)
VaR_99_hist_5d = np.percentile(port_returns_5d, 1)

# 5. Parametric VaR (Variance-Covariance)
mu_1d = port_returns.mean()
sigma_1d = port_returns.std()
VaR_95_param_1d = mu_1d - 1.65 * sigma_1d
VaR_99_param_1d = mu_1d - 2.33 * sigma_1d
mu_5d = port_returns_5d.mean()
sigma_5d = port_returns_5d.std()
VaR_95_param_5d = mu_5d - 1.65 * sigma_5d
VaR_99_param_5d = mu_5d - 2.33 * sigma_5d

# 6. Monte Carlo Simulation
np.random.seed(42)
sim_returns_1d = np.random.normal(mu_1d, sigma_1d, 100000)
VaR_95_mc_1d = np.percentile(sim_returns_1d, 5)
VaR_99_mc_1d = np.percentile(sim_returns_1d, 1)
sim_returns_5d = np.random.normal(mu_5d, sigma_5d, 100000)
VaR_95_mc_5d = np.percentile(sim_returns_5d, 5)
VaR_99_mc_5d = np.percentile(sim_returns_5d, 1)

# 7. GARCH model for volatility forecasting
garch_model = arch_model(port_returns * 100, vol="Garch", p=1, q=1)  # Scale for GARCH
garch_fit = garch_model.fit(disp="off")
volatility_forecast = garch_fit.conditional_volatility[-1] / 100  # Forecasted volatility
VaR_95_garch_1d = mu_1d - 1.65 * volatility_forecast
VaR_99_garch_1d = mu_1d - 2.33 * volatility_forecast

# 8. Print results
print("\n1-Day VaR Results:")
print(f"Historical VaR 95%: {VaR_95_hist_1d:.4f}")
print(f"Historical VaR 99%: {VaR_99_hist_1d:.4f}")
print(f"Parametric VaR 95%: {VaR_95_param_1d:.4f}")
print(f"Parametric VaR 99%: {VaR_99_param_1d:.4f}")
print(f"Monte Carlo VaR 95%: {VaR_95_mc_1d:.4f}")
print(f"Monte Carlo VaR 99%: {VaR_99_mc_1d:.4f}")
print(f"GARCH VaR 95%: {VaR_95_garch_1d:.4f}")
print(f"GARCH VaR 99%: {VaR_99_garch_1d:.4f}")
print("\n5-Day VaR Results:")
print(f"Historical VaR 95%: {VaR_95_hist_5d:.4f}")
print(f"Historical VaR 99%: {VaR_99_hist_5d:.4f}")
print(f"Parametric VaR 95%: {VaR_95_param_5d:.4f}")
print(f"Parametric VaR 99%: {VaR_99_param_5d:.4f}")
print(f"Monte Carlo VaR 95%: {VaR_95_mc_5d:.4f}")
print(f"Monte Carlo VaR 99%: {VaR_99_mc_5d:.4f}")

# 9. Plot 1-day Monte Carlo histogram (professional style)
plt.figure(figsize=(12, 7))
plt.hist(sim_returns_1d, bins=100, alpha=0.7, color="dodgerblue", edgecolor="black", label="1-Day Simulated Returns")
plt.axvline(VaR_95_mc_1d, color="red", linestyle="--", linewidth=2, label=f"VaR 95% (MC): {VaR_95_mc_1d:.4f}")
plt.axvline(VaR_99_mc_1d, color="darkgreen", linestyle="--", linewidth=2, label=f"VaR 99% (MC): {VaR_99_mc_1d:.4f}")
plt.title("Monte Carlo Simulation - 1-Day Portfolio Returns (20 Stocks)", fontsize=16, weight="bold")
plt.xlabel("Daily Returns", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("outputs/var_histogram_1d.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. Plot 5-day Monte Carlo histogram (professional style)
plt.figure(figsize=(12, 7))
plt.hist(sim_returns_5d, bins=100, alpha=0.7, color="coral", edgecolor="black", label="5-Day Simulated Returns")
plt.axvline(VaR_95_mc_5d, color="red", linestyle="--", linewidth=2, label=f"VaR 95% (MC): {VaR_95_mc_5d:.4f}")
plt.axvline(VaR_99_mc_5d, color="darkgreen", linestyle="--", linewidth=2, label=f"VaR 99% (MC): {VaR_99_mc_5d:.4f}")
plt.title("Monte Carlo Simulation - 5-Day Portfolio Returns (20 Stocks)", fontsize=16, weight="bold")
plt.xlabel("5-Day Returns", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("outputs/var_histogram_5d.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
plt.title("Correlation Matrix of Stock Returns (20 Stocks)", fontsize=16, weight="bold")
plt.savefig("outputs/correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()