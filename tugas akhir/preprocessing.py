# import_libraries.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# data_preprocessing.py

import pandas as pd

# Load data
df = pd.read_excel('D:/kuliah/kuliah/Machine Learning/DATA RUMAH.xlsx')

# Display basic information about the dataframe
print(df.head())
df.info()
print("Shape of data:", df.shape)
print("Jumlah data duplicated:", df.duplicated().sum())
print(df.isna().sum())

# Rename columns for better readability
df = df.rename(columns={
    'NO': 'nomor',
    'NAMA RUMAH': 'nama_rumah',
    'HARGA': 'harga',
    'LB': 'lb',
    'LT': 'lt',
    'KT': 'kt',
    'KM': 'km',
    'GRS': 'grs'
})

# Convert harga to millions
df['harga'] = (df['harga'] / 1000000).astype(int)
df.drop(columns=['nomor'], inplace=True)

# Define price classification function
q1 = df['harga'].quantile(0.25)
median = df['harga'].median()
q3 = df['harga'].quantile(0.75)

def classification_harga(harga):
    if harga <= q1:
        return 'Murah'
    elif harga <= median:
        return 'Menengah'
    else:
        return 'Mahal'

# Add new column for price classification
df['tingkat_harga'] = df['harga'].apply(classification_harga)

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
