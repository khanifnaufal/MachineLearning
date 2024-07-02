# linear_regression_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Define features and target
X = df[['lb', 'lt', 'kt', 'km', 'grs']].values
y = df['harga'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Visualize regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='orange', label='Data Latih')
plt.scatter(y_test, y_pred, color='red', label='Data Uji')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='gray', label='Garis Identitas')
plt.xlabel('Nilai Sebenarnya')
plt.ylabel('Nilai Prediksi')
plt.title('Prediksi Model Regresi Linier')
plt.legend()
plt.show()
