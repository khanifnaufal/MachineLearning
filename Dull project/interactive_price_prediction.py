# interactive_price_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import ipywidgets as widgets
from IPython.display import display

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

# Define function to predict house price based on selected features
def predict_house_price(lb, lt, kt, km, grs):
    # Predict house price using linear regression model
    predicted_price = model.predict([[lb, lt, kt, km, grs]])
    
    # Display the predicted price
    print("Harga rumah impian anda diperkirakan sekitar IDR {:,.3f} juta".format(predicted_price[0]))

# Create sliders for each feature
slider_lb = widgets.FloatSlider(value=100, min=df['lb'].min(), max=df['lb'].max(), step=10, description='LB:')
slider_lt = widgets.FloatSlider(value=300, min=20, max=df['lt'].max(), step=10, description='LT:')
slider_kt = widgets.FloatSlider(value=3, min=1, max=df['kt'].max(), step=1, description='KT:')
slider_km = widgets.FloatSlider(value=2, min=1, max=df['km'].max(), step=1, description='KM:')
slider_grs = widgets.FloatSlider(value=2, min=1, max=df['grs'].max(), step=1, description='GRS:')

# Create interactive display
interactive_display = widgets.interactive(predict_house_price, lb=slider_lb, lt=slider_lt, kt=slider_kt, km=slider_km, grs=slider_grs)

# Display the interactive widget
display(interactive_display)
