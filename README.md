# Adaptive Portfolio Optimization (MASI)

This repository contains the complete quantitative pipeline developed for the robust optimization of financial portfolios, applied to the Casablanca Stock Exchange (MASI) over the 2015-2026 period.

## Abstract
Traditional Markowitz mean-variance optimization often fails out-of-sample due to its extreme sensitivity to input estimation errors ("error maximizer" problem). This project solves this structural flaw by combining Bayesian updates, matrix shrinkage, robust minimax hedging, and online machine learning.

### Key Methodological Innovations
1. **Endogenous Views Generation**: An EGARCH-M model captures asymmetric volatility and market shocks, feeding endogenous return and uncertainty views to a Black-Litterman Bayesian framework.
2. **Minimax Robust Optimization**: A Hansen-Sargent minimax algorithm hedges the allocation against the worst-case covariance scenario within a defined uncertainty ellipsoid.
3. **Online Convex Optimization (FTRL)**: To avoid static risk-aversion calibrations, a Follow-The-Regularized-Leader (FTRL) algorithm dynamically adapts the portfolio's risk aversion ($\delta$) sequentially.
4. **PAC-Bayes Generalization**: The algorithmic stability is formally monitored using PAC-Bayes bounds (McAllester) based on the Kullback-Leibler divergence between the empirical views and the CAPM equilibrium prior.

## Results (Out-of-Sample: 2023-2026)
Incorporating realistic constraints (20 basis points transaction costs, 21-day rebalancing frequency), the adaptive FTRL algorithm dramatically outperforms static equilibrium models:
- **Net Annualized Return**: 39.04%
- **Sharpe Ratio**: 1.625
- **Maximum Drawdown**: -20.86%

## Architecture
- `data_pipeline.py`: Financial data engineering, arithmetic return composition, and data anomaly filtering.
- `covariance.py`: Ledoit-Wolf shrinkage estimator.
- `views_egarch.py`: EGARCH-M forecasting.
- `black_litterman.py`: Bayesian posterior calculation.
- `optimizers.py`: SciPy-based SLSQP solvers for Markowitz, Black-Litterman, and Minimax.
- `online_ftrl.py`: Custom grid-search FTRL optimization engine.
- `pac_bayes.py`: KL Divergence tracking.
- `main.py`: The full walk-forward out-of-sample backtester.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the backtest pipeline: `python main.py`
