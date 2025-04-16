# pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 🔹 Load CSV Data
df = pd.read_csv('data.csv')

# 🔹 Features & Target
X = df[['Age', 'Depression']]
y = df['CGPA']

# 🔹 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🔹 Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 🔹 Output
print("📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

