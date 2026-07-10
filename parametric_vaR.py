import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#portfolio parameters
AUM_USD = 10_000_000
CONFIDENCE_LEVEL = 0.99
Z_SCORE = norm.ppf(CONFIDENCE_LEVEL) #what is norm?

START_DATE = "2024-01-01"
END_DATE = "2026-07-01"

tickers = ["MSFT", "PG", "SHEL.L", "ULVR.L"] #why shell?
fx_ticker = "GBPUSD=X" #what is this?

weights = np.array([0.25, 0.25, 0.25, 0.25]) #set unequal weights.

print("Extracting multi-asset historical data streams...") #do we need this?
raw_data = yf.download(tickers, start=START_DATE, end=END_DATE)['Close']
fx_data = yf.download(fx_ticker, start=START_DATE, end=END_DATE)['Close']

raw_data = raw_data.ffill().bfill() #learn these methods
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

parametric_var_percent = Z_SCORE * portfolio_volatility_daily
parametric_var_usd = AUM_USD * parametric_var_percent

portfolio_daily_returns = daily_returns.dot(weights)

violations = portfolio_daily_returns < -parametric_var_percent
num_violations = int(violations.sum())
total_days = len(portfolio_daily_returns)
expected_violations = total_days * (1 - CONFIDENCE_LEVEL)

if num_violations <= 4:
    basel_zone = "GREEN (Acceptable Model)"
elif num_violations <= 9:
    basel_zone = "YELLOW (Caution / Review Model)"
else:
    basel_zone = "RED (Action Required / Model Miscalibrated)" #change these terms.

# extreme risk scenarios
scenarios = {
    "Lehman-Style Equity Meltdown": {"MSFT": -0.08, "PG": -0.04, "SHEL_USD": -0.12, "ULVR_USD": -0.05},
    "GBP Flash Crash (Currency Shock)": {"MSFT": 0.00, "PG": 0.00, "SHEL_USD": -0.15, "ULVR_USD": -0.15},
    "Tech Sector Crash": {"MSFT": -0.15, "PG": 0.01, "SHEL_USD": -0.02, "ULVR_USD": 0.00} #change these terms.
}

stress_results = {}
for name, shocks in scenarios.items():
    shock_vector = np.array([shocks["MSFT"], shocks["PG"], shocks["SHEL_USD"], shocks["ULVR_USD"]])
    scenario_loss_percent = np.dot(weights, shock_vector)
    scenario_loss_usd = AUM_USD * abs(scenario_loss_percent) if scenario_loss_percent < 0 else 0
    stress_results[name] = scenario_loss_usd


# summary
print("\n")
print("Institutional Risk Report (USD)") #change the look.
print("-"*60)
print(f"Total Portfolio Value (AUM):      ${AUM_USD:,.2f}")
print(f"Confidence Level / Horizon:       {CONFIDENCE_LEVEL*100}% / 1-Day")
print(f"Daily Portfolio Volatility:       {portfolio_volatility_daily * 100:.4f}%")
print(f"1-Day Parametric VaR (%):         {parametric_var_percent * 100:.4f}%")
print(f"1-Day Parametric VaR (USD):       ${parametric_var_usd:,.2f}")
print("-"*60)
print("Regulatory/Backtesting Summary (Basel Framework)")
print(f"Total Backtesting Days:           {total_days} days")
print(f"Expected Violations:              {expected_violations:.2f} days")
print(f"Observed Violations:              {num_violations} days")
print(f"Basel Traffic Light Zone:         {basel_zone}")
print("-"*60)
print("Stress Testing Scenarios & Estimated Losses (USD)")
for scenario, loss in stress_results.items():
    print(f" -> {scenario:<32}: Estimated Loss ${loss:,.2f}")
print("\n")

# visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(10)) #change the look.

# Chart 1: Asset Covariance Heatmap Matrix
sns.heatmap(cov_matrix_annualized * 100, annot=True, cmap="coolwarm", fmt=".2f", 
            xticklabels=usd_portfolio_prices.columns, yticklabels=usd_portfolio_prices.columns, ax=axes[0,0])
axes[0,0].set_title("Annualized Covariance Matrix (%)", fontweight="bold")

# Chart 2: Backtesting Over Time & VaR Violations
axes[0,1].plot(portfolio_daily_returns.index, portfolio_daily_returns * 100, color="teal", label="Daily Portfolio Return")
axes[0,1].axhline(-parametric_var_percent * 100, color="crimson", linestyle="--", linewidth=2, label=f"99% VaR Threshold")
violation_dates = portfolio_daily_returns[violations].index
violation_values = portfolio_daily_returns[violations] * 100
axes[0,1].scatter(violation_dates, violation_values, color="black", marker="x", s=50, zorder=5, label="Basel Breaches")
axes[0,1].set_title("Historical Model Backtesting & Breaches", fontweight="bold")
axes[0,1].set_ylabel("Return (%)")
axes[0,1].legend()

# Chart 3: Distribution Histogram & Parametric Cutoff
axes[1,0].hist(portfolio_daily_returns * 100, bins=50, color="gainsboro", edgecolor="gray", alpha=0.7)
axes[1,0].axvline(-parametric_var_percent * 100, color="crimson", linestyle="-", linewidth=2.5, label=f"Parametric VaR Line")
axes[1,0].set_title("99% Parametric VaR Distribution Tail", fontweight="bold")
axes[1,0].set_xlabel("Daily Portfolio Return (%)")
axes[1,0].legend()

# Chart 4: Stress Test Loss Comparison Bar Chart
axes[1,1].barh(list(stress_results.keys()), [v / 1e6 for v in stress_results.values()], color=["darkred", "chocolate", "midnightblue"])
axes[1,1].set_xlabel("Loss Magnitude (Millions USD)")
axes[1,1].set_title("Macroeconomic Scenario Stress Losses", fontweight="bold")

plt.tight_layout()
plt.show()
