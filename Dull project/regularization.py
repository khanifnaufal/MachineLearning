# regularization.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Define features and target
X = df[['lb', 'lt', 'kt', 'km', 'grs']].values
y = df['harga'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge model
model_ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
model_ridge.fit(X_train, y_train)

# Make predictions
y_train_pred_ridge = model_ridge.predict(X_train)
y_pred_ridge = model_ridge.predict(X_test)

# Calculate evaluation metrics
mae_train_ridge = mean_absolute_error(y_train, y_train_pred_ridge)
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
r2_train_ridge = r2_score(y_train, y_train_pred_ridge)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Print evaluation results
print("Evaluasi Model Ridge pada Data Pelatihan:")
print(f'MAE: {mae_train_ridge}')
print(f'MSE: {mse_train_ridge}')
print(f'R2 Score: {r2_train_ridge}')
print("\nEvaluasi Model Ridge pada Data Pengujian:")
print(f'MAE: {mae_ridge}')
print(f'MSE: {mse_ridge}')
print(f'R2 Score: {r2_ridge}')
