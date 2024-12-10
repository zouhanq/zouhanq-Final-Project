# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from model_training import train_decision_tree
from data_preparation import feature_engineering, handle_missing_data
from regime_detection import detect_regimes
from sklearn.cluster import KMeans

# Parameters
CUT_OFF_DATE = datetime(2024, 1, 1)
N_COMPONENTS = 3

# Load Data
df = pd.read_csv('data/mink.csv')
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

# Split
train_data = df.loc[df.index < CUT_OFF_DATE].copy()
test_data = df.loc[df.index >= CUT_OFF_DATE].copy()

# Handle missing in train
train_data = handle_missing_data(train_data)

# Feature engineering on train
train_data = feature_engineering(train_data)

features = [
    'close', 'volume', 'short_ma', 'long_ma', 'bollinger_mid',
    'bollinger_upper', 'bollinger_lower', 'bollinger_width',
    '30min_Diff', '30min_Dea', 'rsi', 'vwap'
]

train_subset = train_data[features].dropna()
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_subset)

svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
train_svd = svd.fit_transform(train_scaled)
svd_cols = [f'svd_component_{i+1}' for i in range(N_COMPONENTS)]
for i in range(N_COMPONENTS):
    train_data[svd_cols[i]] = np.nan
train_data.loc[train_subset.index, svd_cols] = train_svd
train_data.dropna(subset=svd_cols, inplace=True)

# Regime detection on train
train_data, kmeans = detect_regimes(train_data, n_clusters=3)

# Label creation for training model
train_data['future_ret'] = train_data['close'].shift(-1) - train_data['close']
train_data['label'] = np.where(train_data['future_ret'] > 0, 1, -1)
train_data.dropna(subset=['label'], inplace=True)

X_train = train_data[features + svd_cols + ['regime']]
y_train = train_data['label']

model = train_decision_tree(X_train, y_train)

# Test data processing
test_data = handle_missing_data(test_data)
test_data = feature_engineering(test_data)

positions = 0  # 0 = no position, 1 = long
entry_price = 0
profit_pct_list = []
cum_profit = 1.0

test_index = sorted(test_data.index)
cum_profit_series = pd.Series(index=test_index, dtype=float)
cum_profit_series[:] = np.nan

signals = []
trades = []  # to store trade details

for i, t in enumerate(test_index):
    row = test_data.loc[[t]]
    # Skip if missing features
    if row[features].isna().any().any():
        signals.append(0)
        if i == 0:
            cum_profit_series[t] = cum_profit
        else:
            cum_profit_series[t] = cum_profit_series[test_index[i-1]]
        continue

    # Scale & SVD
    row_scaled = scaler.transform(row[features])
    row_svd = svd.transform(row_scaled)
    for c_i in range(N_COMPONENTS):
        row[svd_cols[c_i]] = row_svd[0, c_i]

    if row[svd_cols].isna().any().any():
        signals.append(0)
        if i == 0:
            cum_profit_series[t] = cum_profit
        else:
            cum_profit_series[t] = cum_profit_series[test_index[i-1]]
        continue

    # Predict regime
    regime = kmeans.predict(row[svd_cols])[0]
    row['regime'] = regime

    # Predict signal
    X_test = row[features + svd_cols + ['regime']]
    pred_signal = model.predict(X_test)[0]

    # Enforce buy-before-sell rule
    if pred_signal == -1 and positions == 0:
        pred_signal = 0
    if pred_signal == 1 and positions == 1:
        pred_signal = 0

    if pred_signal == 1 and positions == 0:
        # buy
        positions = 1
        entry_price = row['close'].values[0]
        entry_time = t
        entry_regime = regime
    elif pred_signal == -1 and positions == 1:
        # sell
        exit_price = row['close'].values[0]
        trade_return = (exit_price - entry_price) / entry_price
        cum_profit *= (1 + trade_return)
        profit_pct_list.append(trade_return)
        positions = 0
        exit_time = t
        # Record the trade
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'regime': entry_regime,
            'profit_pct': trade_return
        })
        entry_price = 0

    signals.append(pred_signal)
    if i == 0:
        cum_profit_series[t] = cum_profit
    else:
        cum_profit_series[t] = cum_profit

final_cum_profit = cum_profit - 1
print("Final Cumulative Profit on Test Data:", final_cum_profit)
print("Number of trades:", len(profit_pct_list))

total_buys = signals.count(1)
total_sells = signals.count(-1)
print("Total Buy Signals:", total_buys)
print("Total Sell Signals:", total_sells)

# Convert trades to DataFrame if not empty
if trades:
    trades_df = pd.DataFrame(trades)
else:
    trades_df = pd.DataFrame(columns=['entry_time', 'exit_time', 'regime', 'profit_pct'])

# Plot cumulative profit over time
plt.figure(figsize=(14,7))
plt.plot(cum_profit_series.index, cum_profit_series - 1, label='Cumulative Profit')
plt.xlabel('Time')
plt.ylabel('Cumulative Profit')
plt.title('Cumulative Profit Over Time')
plt.legend()
plt.show()

# Additional plots

## 1. Regimes Over Time in Test Data
if 'regime' in test_data.columns and not test_data['regime'].isna().all():
    plt.figure(figsize=(14,5))
    plt.plot(test_data.index, test_data['regime'], marker='o', linestyle='-', ms=2)
    plt.xlabel('Time')
    plt.ylabel('Regime Label')
    plt.title('Test Data Regimes Over Time')
    plt.show()

trades_df['cumulative_wins'] = (trades_df['profit_pct'] > 0).cumsum()
trades_df['cumulative_count'] = range(1, len(trades_df)+1)
trades_df['cumulative_win_rate'] = trades_df['cumulative_wins'] / trades_df['cumulative_count']

plt.figure(figsize=(10,5))
plt.plot(trades_df['exit_time'], trades_df['cumulative_win_rate'], marker='o', linestyle='-')
plt.title('Cumulative Win Rate Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Win Rate')
plt.ylim(0,1)
plt.show()
