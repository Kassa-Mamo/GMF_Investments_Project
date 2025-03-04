import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Fetch data
def fetch_data():
    assets = ['TSLA', 'BND', 'SPY']
    data = {}
    for asset in assets:
        data[asset] = yf.download(asset, start="2015-01-01", end="2025-01-31")['Adj Close']
    return pd.DataFrame(data)

# Calculate returns and risk (volatility)
def calculate_returns(data):
    return data.pct_change().dropna()

# Portfolio metrics
def portfolio_metrics(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    return portfolio_return, portfolio_volatility

# Sharpe Ratio
def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return, portfolio_volatility = portfolio_metrics(weights, mean_returns, cov_matrix)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative because we are minimizing

# Optimization function
def optimize_portfolio():
    data = fetch_data()
    returns = calculate_returns(data)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(data.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for asset in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights sum to 1
    
    result = minimize(sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    portfolio_return, portfolio_volatility = portfolio_metrics(optimal_weights, mean_returns, cov_matrix)
    
    print(f"Optimal Portfolio Weights: {optimal_weights}")
    print(f"Expected Portfolio Return: {portfolio_return:.2f}")
    print(f"Expected Portfolio Volatility: {portfolio_volatility:.2f}")
    
    # Visualize the portfolio performance
    plt.bar(data.columns, optimal_weights, color='green', alpha=0.6)
    plt.title('Optimal Portfolio Asset Allocation')
    plt.show()
    
    return optimal_weights, portfolio_return, portfolio_volatility

if __name__ == "__main__":
    optimize_portfolio()
