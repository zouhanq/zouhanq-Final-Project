# data_preparation.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import BollingerBands
from ta.trend import MACD
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def align_to_trading_hours(data):
    """Align data to trading hours (9:30 AM to 2:59 PM)."""
    data.index = data.index + pd.DateOffset(minutes=30)
    data = data.between_time('9:30', '14:59')
    return data

def handle_missing_data(data):
    """Fill missing values to maintain data consistency."""
    data.fillna(method='ffill', inplace=True)
    return data

# Feature Engineering
def feature_engineering(data):
    """Calculate technical indicators for feature extraction and handle NaNs."""
    # RSI
    data['rsi'] = RSIIndicator(close=data['close'], window=14).rsi()
    print("NaNs after RSI:", data['rsi'].isna().sum())

    # VWAP
    vwap = VolumeWeightedAveragePrice(high=data['high'], low=data['low'], close=data['close'], volume=data['volume']).vwap
    data['vwap'] = vwap
    print("NaNs after VWAP:", data['vwap'].isna().sum())

    # Moving Averages
    data['short_ma'] = data['close'].rolling(window=5).mean()
    data['long_ma'] = data['close'].rolling(window=10).mean()
    print("NaNs after moving averages:", data[['short_ma', 'long_ma']].isna().sum().sum())

    # Bollinger Bands
    bollinger = BollingerBands(data['close'], window=20, window_dev=2)
    data['bollinger_mid'] = bollinger.bollinger_mavg()
    data['bollinger_upper'] = bollinger.bollinger_hband()  # Corrected method name
    data['bollinger_lower'] = bollinger.bollinger_lband()  # Corrected method name
    data['bollinger_width'] = data['bollinger_upper'] - data['bollinger_lower']
    print("NaNs after Bollinger Bands:", data[['bollinger_mid', 'bollinger_upper', 'bollinger_lower', 'bollinger_width']].isna().sum().sum())

    # MACD
    macd = MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
    data['30min_Diff'] = macd.macd_diff()
    data['30min_Dea'] = macd.macd_signal()
    print("NaNs after MACD:", data[['30min_Diff', '30min_Dea']].isna().sum().sum())

    # Drop rows with any NaN values after all features are calculated
    data.dropna(inplace=True)
    print("Total NaNs after dropping:", data.isna().sum().sum())

    return data
# Apply SVD
def apply_svd(data, n_components=3):
    """Apply SVD for dimensionality reduction on selected technical indicators."""
    features = ['close', 'volume', 'short_ma', 'long_ma', 'bollinger_mid', 
                'bollinger_upper', 'bollinger_lower', 'bollinger_width', 
                '30min_Diff', '30min_Dea', 'rsi', 'vwap']
    
    # Drop rows with NaN values in selected features
    data_subset = data[features].dropna()
    print("NaNs before SVD:", data_subset.isna().sum().sum())

    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_subset)

    # Apply SVD
    svd = TruncatedSVD(n_components=n_components)
    data_svd = svd.fit_transform(data_scaled)

    # Create DataFrame for SVD components, align indices with original data
    svd_df = pd.DataFrame(data_svd, columns=[f'svd_component_{i+1}' for i in range(n_components)], index=data_subset.index)
    data = data.join(svd_df, how='left')

    # Drop any remaining NaN values in the SVD columns
    print("NaNs after SVD components:", data[[f'svd_component_{i+1}' for i in range(n_components)]].isna().sum().sum())
    data.dropna(subset=[f'svd_component_{i+1}' for i in range(n_components)], inplace=True)
    
    return data
