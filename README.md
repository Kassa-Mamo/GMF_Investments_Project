 1. Overview
This report presents the preprocessing and exploratory data analysis (EDA) performed on historical financial data of TSLA, BND, and SPY. The objective is to clean, analyze, and extract insights from the data, preparing it for further modeling.
2. Objective
The primary objectives of this task are:
•	Extract historical stock market data for TSLA, BND, and SPY using the YFinance API.
•	Clean and preprocess the data to ensure quality and consistency.
•	Conduct exploratory data analysis to uncover trends, patterns, and volatility.
•	Identify seasonality and trends in the time series data.
•	Analyze risk and returns using metrics like Value at Risk (VaR) and Sharpe Ratio.
3. Methodology
3.1 Data Collection
•	Used the yfinance Python library to extract daily stock price data for:
o	TSLA (Tesla): High volatility, potential high returns.
o	BND (Bond ETF): Stability and low risk.
o	SPY (S&P 500 ETF): Diversified, moderate-risk market exposure.
3.2 Data Cleaning and Understanding
•	Checked for missing values and handled them using interpolation or removal where necessary.
•	Verified data types and ensured numerical consistency.
•	Generated summary statistics to understand distribution (mean, median, standard deviation, min/max values).
•	Normalized data where required to ensure consistency across datasets.
3.3 Exploratory Data Analysis (EDA)
•	Time Series Visualization:
o	Plotted closing prices to observe historical trends.
o	Compared stock performances over time.
•	Daily Returns Analysis:
o	Calculated percentage change to measure daily volatility.
o	Visualized daily returns distribution to identify high-risk days.
•	Volatility Analysis:
o	Computed rolling mean and rolling standard deviation to examine short-term fluctuations.
o	Used a 30-day rolling window to track changing volatility trends.
•	Outlier Detection:
o	Identified days with extreme returns (unusually high/low changes in price).
o	Analyzed anomalies in TSLA’s price movement.
3.4 Seasonality and Trend Analysis
•	Used time series decomposition (via statsmodels) to separate:
o	Trend (long-term direction of stock prices).
o	Seasonality (recurring patterns in returns).
o	Residuals (unexplained variations).
3.5 Risk and Return Assessment
•	Value at Risk (VaR):
o	Estimated potential losses using a 95% confidence interval.
•	Sharpe Ratio Calculation:
o	Measured risk-adjusted returns by comparing average returns to volatility.
4. Results
•	TSLA exhibited high volatility with frequent price spikes and dips.
•	BND remained stable with minor fluctuations, confirming its low-risk nature.
•	SPY showed moderate fluctuations but maintained steady long-term growth.
•	Daily returns analysis revealed:
o	TSLA had extreme price movements on specific days.
o	BND had the least fluctuations.
o	SPY had balanced movement between risk and return.
•	Volatility Analysis:
o	TSLA’s rolling standard deviation was the highest, indicating unpredictability.
o	BND maintained a near-flat volatility curve.
o	SPY showed moderate volatility with occasional spikes.
•	Trend Analysis:
o	TSLA displayed strong upward momentum despite short-term fluctuations.
o	BND remained steady with minor seasonal effects.
o	SPY followed a stable upward trend over time.
•	Risk Assessment:
o	TSLA had the highest risk exposure (VaR showed significant potential losses).
o	BND had minimal risk.
o	SPY had moderate risk, balancing return potential.
5. Conclusion
•	TSLA is highly volatile with significant fluctuations but offers high returns.
•	BND is a safe investment with minimal risk and low volatility.
•	SPY provides balanced exposure with moderate risk and stable long-term growth.
•	Volatility and daily return analysis help in understanding stock movement risks.
•	Sharpe Ratio & VaR calculations give insights into risk-adjusted returns.
•	This analysis prepares the data for further modeling, forecasting, and portfolio optimization in the next phases.


