# pip install pandas matplotlib statsmodels
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load the data
data = pd.read_csv('data.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Step 2: Fit an ARIMA model
model = ARIMA(data['Passengers'], order=(2, 1, 2))  # (p,d,q) values can be tuned
model_fit = model.fit()

# Step 3: Forecast the next 12 months
forecast = model_fit.forecast(steps=12)

# Step 4: Plot the original data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Passengers'], label='Original Data', color='blue')
plt.plot(pd.date_range(data.index[-1], periods=12, freq='M'), forecast, label='Forecast', color='red', linestyle='--')
plt.title('Time Series Forecasting of Air Passengers')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.legend()
plt.grid(True)
plt.show()
