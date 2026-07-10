# Parametric VaR Model

## About The Project

A 1-day 99% Parametric Value at Risk (VaR) model with Basel regulatory backtesting and historical stress testing. It analyzes market risk on a cross-currency equity portfolio, spanning tech, consumer, and energy stocks listed on the LSE and NYSE. 

Disclaimer: All data and modeling methodologies are based on public and sample sources.

## Methodology
This model estimates the portfolio's daily market risk using a Parametric (Variance-Covariance) VaR framework. The covariance-based portfolio risk logic is rooted in Modern Portfolio Theory (MPT). The portfolio maps multi-currency equity assets directly onto a unified covariance matrix to isolate linear risk dependencies.

#### Assumptions:
- Asset returns are assumed to follow a stationary multivariate normal distribution.
- Portfolio variance is assumed to be a stable linear combination of the asset weights and the covariance matrix.
- LSE asset prices were converted to USD daily using historical rates. Consequently, GBP/USD volatility is embedded in equity returns rather than as an isolated risk factor.
- Structural shifts in asset correlations over the lookback window are outside the scope of this static parametric calculation.

## Output
A 1-day Parametric VaR threshold at a 99% confidence level (Z-score of ~2.326) is calculated using the standard normal inverse cumulative distribution function (CDF).

## Backtesting and Stress Testing
Model validation is conducted via a 250-day regulatory exception framework based on Basel standards. Realized exceptions, where daily portfolio losses exceed the VaR threshold, are used to determine the model's calibration status (Traffic Light zone).

Macroeconomic stress testing is applied by subjecting the asset matrix to extreme, non-normal joint-shock scenarios to quantify tail risk exposure in USD.

## Conclusion and Future Work
The model flags a "Caution/Warning" (Yellow zone) due to an excessive number of exceptions. This is an expected limitation, as the model assumes a normal distribution bell curve for market returns, underestimating fat tails.

In future iterations I plan to incorporate a broader spectrum of risk dimensions and techniques like GARCH and Monte Carlo.

