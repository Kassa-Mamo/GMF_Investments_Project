  import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data (make sure to use the preprocessed TSLA data, if not already fetched)
def load_data():
    tsla = pd.read_csv('data/TSLA.csv', index_col='Date', parse_dates=True)
    tsla = tsla['Close']
    return tsla

# Forecasting function (using ARIMA as an example)
def forecast(tsla_data, steps=180):  # Forecast for 6 months (180 days)
    model = ARIMA(tsla_data, order=(5, 1, 0))  # ARIMA model parameters
    model_fit = model.fit()
    forecasted_values = model_fit.forecast(steps=steps)
    
    forecast_index = pd.date_range(tsla_data.index[-1], periods=steps+1, freq='B')[1:]
    forecast_series = pd.Series(forecasted_values, index=forecast_index)
    
    return forecast_series

# Plotting the forecast
def plot_forecast(tsla_data, forecast_series):
    plt.figure(figsize=(10, 6))
    plt.plot(tsla_data, label='Historical TSLA Prices')
    plt.plot(forecast_series, label='Forecasted TSLA Prices', color='red')
    plt.title('Tesla Stock Price Forecast (6 Months)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    tsla_data = load_data()
    forecast_series = forecast(tsla_data)
    plot_forecast(tsla_data, forecast_series)

