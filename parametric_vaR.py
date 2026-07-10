import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#portfolio parameters
AUM_USD = 10_000_000
Confidence_level = 0.99
Z_Score = norm.ppf(Confidence_level)

start_date = "2024-01-01"
end_date = "2026-07-01"

tickers = ["MSFT", "PG", "SHEL.L", "ULVR.L"]
fx_ticker = ["GBPUSD=X"]

weights = np.array([0.10, 0.40, 0.10, 0.40])

raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
fx_data = yf.download(fx_ticker, start=start_date, end=end_date)['Close']

raw_data = raw_data.ffill().bfill()
fx_data = fx_data.ffill().bfill()

df = pd.merge(raw_data, fx_data, left_index=True, right_index=True, how='inner')

df["SHEL_USD"] = (df["SHEL.L"] / 100) * df["GBPUSD=X"]
df["ULVR_USD"] = (df["ULVR.L"] / 100) * df["GBPUSD=X"]

usd_portfolio_prices = df[["MSFT", "PG", "SHEL_USD", "ULVR_USD"]]

daily_returns = usd_portfolio_prices.pct_change().dropna()

cov_matrix_daily = daily_returns.cov()
cov_matrix_annualized = cov_matrix_daily * 252

portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_daily, weights))
portfolio_volatility_daily = np.sqrt(portfolio_variance)

parametric_var_percent = Z_Score * portfolio_volatility_daily
parametric_var_usd = AUM_USD * parametric_var_percent

portfolio_daily_returns = daily_returns.dot(weights)

#backtesting
violations = portfolio_daily_returns < -parametric_var_percent
num_violations = int(violations.sum())
total_days = len(portfolio_daily_returns)
expected_violations = total_days * (1 - Confidence_level)

if num_violations <= 4:
    basel_zone = "Safe"
elif num_violations <= 9:
    basel_zone = "Caution/Warning"
else:
    basel_zone = "Action Required"

# extreme risk scenarios
scenarios = {
    "Equity Meltdown": {"MSFT": -0.08, "PG": -0.04, "SHEL_USD": -0.12, "ULVR_USD": -0.05},
    "Currency Shock": {"MSFT": 0.00, "PG": 0.00, "SHEL_USD": -0.15, "ULVR_USD": -0.15},
    "Tech Sector Crash": {"MSFT": -0.15, "PG": 0.01, "SHEL_USD": -0.02, "ULVR_USD": 0.00}
}

stress_results = {}
for name, shocks in scenarios.items():
    shock_vector = np.array([shocks["MSFT"], shocks["PG"], shocks["SHEL_USD"], shocks["ULVR_USD"]])
    scenario_loss_percent = np.dot(weights, shock_vector)
    scenario_loss_usd = AUM_USD * abs(scenario_loss_percent) if scenario_loss_percent < 0 else 0
    stress_results[name] = scenario_loss_usd


# summary
print("\n")
print("Institutional Risk Report (USD)")
print("-"*60)
print(f"Total Portfolio Value (AUM):${AUM_USD:,.2f}")
print(f"Confidence Level: {Confidence_level*100}% / 1-Day")
print(f"Daily Portfolio Volatility:{portfolio_volatility_daily * 100:.4f}%")
print(f"1-Day Parametric VaR (%):{parametric_var_percent * 100:.4f}%")
print(f"1-Day Parametric VaR (USD):${parametric_var_usd:,.2f}")
print("\n")
print("Regulatory/Backtesting Summary")
print("-"*60)
print(f"Total Backtesting Days:{total_days} days")
print(f"Expected Violations:{expected_violations:.2f} days")
print(f"Observed Violations:{num_violations} days")
print(f"Model Status:{basel_zone}")
print("\n")
print("Stress Testing Scenarios & Estimated Losses (USD)")
print("-"*60)
for scenario, loss in stress_results.items():
    print(f" -> {scenario}: Estimated Loss ${loss:,.2f}")
print("\n")

# visualizations

# Chart 1: Asset Covariance Heatmap Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix_annualized * 100, annot=True, cmap="Blues", fmt=".2f", 
            xticklabels=usd_portfolio_prices.columns, yticklabels=usd_portfolio_prices.columns)
plt.title("Annualized Covariance Matrix (%)", fontweight="bold")
plt.tight_layout()
plt.show()

# Chart 2: Backtesting Over Time & VaR Violations
plt.figure(figsize=(10, 5))
plt.plot(portfolio_daily_returns.index, portfolio_daily_returns * 100, color="teal", label="Daily Portfolio Return")
plt.axhline(-parametric_var_percent * 100, color="crimson", linestyle="--", linewidth=2, label="99% VaR Threshold")

violation_dates = portfolio_daily_returns[violations].index
violation_values = portfolio_daily_returns[violations] * 100
plt.scatter(violation_dates, violation_values, color="black", marker="x", s=50, zorder=5, label="Basel Breaches")

plt.title("Historical Model Backtesting & Breaches", fontweight="bold")
plt.ylabel("Return (%)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Chart 3: Distribution Histogram & Parametric Cutoff
plt.figure(figsize=(8, 5))
plt.hist(portfolio_daily_returns * 100, bins=50, color="gainsboro", edgecolor="gray", alpha=0.7)
plt.axvline(-parametric_var_percent * 100, color="crimson", linestyle="-", linewidth=2.5, label="Parametric VaR Line")
plt.title("99% Parametric VaR Distribution Tail", fontweight="bold")
plt.xlabel("Daily Portfolio Return (%)")
plt.legend()
plt.tight_layout()
plt.show()

# Chart 4: Stress Test Loss Comparison Bar Chart
plt.figure(figsize=(8, 4))
plt.barh(list(stress_results.keys()), [v / 1e6 for v in stress_results.values()], color=["darkred", "chocolate", "midnightblue"])
plt.xlabel("Loss Magnitude (Millions USD)")
plt.title("Macroeconomic Scenario Stress Losses", fontweight="bold")
plt.tight_layout()
plt.show()
