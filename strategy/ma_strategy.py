import base as strat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ta
import data_preparation as dp
import regime_detection as rd
# Initialize Tushare

def align_to_trading_hours(data):
    """Ensure that the data aligns to start at 9:30 AM and then resample correctly."""
    data.index = data.index + pd.DateOffset(minutes=30)
    data = data.between_time('9:30', '14:59')
    return data

def calculate_daily_average(data):
    """Calculate the daily average price."""
    data['daily_avg'] = data.groupby(data.index.date)['close'].transform('mean')
    return data

def calculate_daily_open(data):
    """Calculate the daily opening price."""
    data['daily_open'] = data.groupby(data.index.date)['open'].transform('first')
    return data


def calculate_conservative_profit_target(open_price, ma10):
    """Calculate 特别保守版最低盈利目标线."""
    return open_price + (open_price - ma10) * 5

def calculate_moving_stop_loss(current_price, prev_low, prev_ma10, post_open_max):
    """Calculate 移动盈损线."""
    return min(prev_low, prev_ma10, post_open_max - 5)

def calculate_holding_time(data):
    """Calculate holding time in minutes based on the index intervals."""
    holding_time = []
    initial_time = data.index[0]
    
    for current_time in data.index:
        time_diff = (current_time - initial_time).total_seconds() / 60
        holding_time.append(time_diff)
        
    data['holding_time'] = holding_time
    return data

def generate_regime_based_signals(data):
    """Generate buy/sell signals based on market regimes."""
    data['buy_signal'] = 0
    data['sell_signal'] = 0

    for i in range(len(data)):
        regime = data['regime'].iloc[i]

        if regime == 0:  # Mean-reversion regime
            # Generate buy/sell signals using Bollinger Bands
            if data['close'].iloc[i] < data['bollinger_lower'].iloc[i]:
                data['buy_signal'].iloc[i] = 1
            elif data['close'].iloc[i] > data['bollinger_upper'].iloc[i]:
                data['sell_signal'].iloc[i] = -1
            
        elif regime == 1:  # Trending regime
            # Generate buy/sell signals using moving average crossover
            if data['short_ma'].iloc[i] > data['long_ma'].iloc[i]:
                data['buy_signal'].iloc[i] = 1
            elif data['short_ma'].iloc[i] < data['long_ma'].iloc[i]:
                data['sell_signal'].iloc[i] = -1
            
        elif regime == 2:  # Moderate regime
            # Apply a more conservative strategy
            if data['close'].iloc[i] > data['daily_open'].iloc[i] and data['rsi'].iloc[i] > 50:
                data['buy_signal'].iloc[i] = 1
            elif data['close'].iloc[i] < data['daily_open'].iloc[i] and data['rsi'].iloc[i] < 50:
                data['sell_signal'].iloc[i] = -1

    return data


def ma_strategy(data):
    """Moving Average strategy with regime-based signal generation."""
    data = pd.DataFrame(data)
    data.set_index('Time', inplace=True)

    data = dp.align_to_trading_hours(data)
    data = dp.handle_missing_data(data)

    data = dp.feature_engineering(data)
    data = dp.apply_svd(data)

    data, kmeans = rd.detect_regimes(data, n_clusters=3)
    rd.plot_regimes(data)
    
    data = generate_regime_based_signals(data)
    
    return data

def calculate_performance_metrics(data):
    """Calculate performance metrics for the strategy."""
    # Calculate profit pct
    data = strat.calculate_prof_pct(data)

    # Calculate cumulative profit
    data = strat.calculate_cum_prof(data)

    # Total return
    total_return = data['cum_profit'].iloc[-1]
    
    # Annualized return
    trading_days = 252  # Assuming 252 trading days in a year
    periods_per_day = 48  # Assuming 48 30-minute periods in a trading day
    annual_return = (1 + data['profit_pct'].mean()) ** (trading_days * periods_per_day) - 1

    # Maximum drawdown
    data = strat.caculate_max_drawdown(data, window=trading_days * periods_per_day)
    max_drawdown = data['max_dd'].min()


    # Sharpe ratio
    sharpe, annual_sharpe = strat.calculate_sharpe(data)

    # Win rate
    win_rate = strat.calculate_win_rate(data)

    # Compile the results
    results = {
        'Total Return': total_return,
        'Annualized Return': annual_return,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': annual_sharpe,
        'Win Rate': win_rate
    }

    # Print the results
    for key, value in results.items():
        print(f"{key}: {value}")

    return results

def read_data_from_csv(file_path):
    """
    Reads data from a CSV file.

    :param file_path: str, the path to the CSV file
    :return: DataFrame, the read data
    """
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df
# Example usage
if __name__ == "__main__":
    # Load your CSV data
    df = read_data_from_csv(r'C:\Users\Hanqi\Documents\WPS Cloud Files\WPSDrive\1013884632\WPS云盘\期货交易\均线策略\data\mink.csv')

    # Ensure the necessary columns exist
    required_columns = ['Time', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert 'Time' column to datetime if necessary
    df['Time'] = pd.to_datetime(df['Time'])

    # Apply the MA strategy
    result = ma_strategy(df)

    # Calculate performance metrics
    result = strat.calculate_prof_pct(result)
    result = strat.calculate_cum_prof(result)

    # Plot the cumulative profit percentage
    plt.figure(figsize=(14, 7))
    plt.plot(result['Time'], result['cum_profit'], label='Cumulative Profit Percentage')

    # Annotate the final cumulative profit percentage on the plot
    final_cum_profit = result['cum_profit'].iloc[-1]
    plt.annotate(f'Final Cumulative Profit: {final_cum_profit:.2%}', 
                 xy=(result['Time'].iloc[-1], final_cum_profit),
                 xytext=(result['Time'].iloc[-1], final_cum_profit + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit Percentage')
    plt.title('Cumulative Profit Percentage Over Time')
    plt.legend()
    plt.show()