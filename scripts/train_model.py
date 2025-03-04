# train_model.py

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima

# Load the data
tsla_data = pd.read_csv("C:/Users/user/Desktop/10 Academy- Machine-Learning/10 Academy W11/GMF_Investments_Project/data/TSLA.csv")
tsla_data['Date'] = pd.to_datetime(tsla_data['Date'])
tsla_data.set_index('Date', inplace=True)

# Train and Test Data Split
train_size = int(len(tsla_data) * 0.8)
train_data, test_data = tsla_data['Close'][:train_size], tsla_data['Close'][train_size:]

# ARIMA Model
model_arima = ARIMA(train_data, order=(5, 1, 0))
model_arima_fit = model_arima.fit()

# Forecast
forecast_arima = model_arima_fit.forecast(steps=30)

# SARIMA Model
model_sarima = SARIMAX(train_data, order=(5, 1, 0), seasonal_order=(1, 1, 0, 5))
model_sarima_fit = model_sarima.fit()
forecast_sarima = model_sarima_fit.forecast(steps=30)

# LSTM Model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(tsla_data['Close'].values.reshape(-1, 1))

def prepare_data(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = prepare_data(scaled_data[:train_size])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)

scaled_test_data = scaler.transform(tsla_data['Close'][train_size:].values.reshape(-1, 1))
X_test, y_test = prepare_data(scaled_test_data)

predicted_stock_price = model_lstm.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Save Results
forecast_results = pd.DataFrame({
    'Date': test_data.index[:30],
    'ARIMA Forecast': forecast_arima,
    'SARIMA Forecast': forecast_sarima,
    'LSTM Forecast': predicted_stock_price.flatten()
})

forecast_results.to_csv("C:/Users/user/Desktop/10 Academy- Machine-Learning/10 Academy W11/GMF_Investments_Project/reports/forecast_results.csv", index=False)

# Model Evaluation
mae_arima = mean_absolute_error(test_data[:30], forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test_data[:30], forecast_arima))
mape_arima = mean_absolute_percentage_error(test_data[:30], forecast_arima)

mae_sarima =
