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

def generate_open_position_signals(data_30min):
    """Generate open position signals based on the defined strategy."""
    # Generate buy signals for the first case
    data_30min['buy_signal_1_1'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['close'] > data_30min['bollinger_mid']),  # Latest price > BOLL(MID)
        1, 0)

    # Generate buy signals for the second case, Condition 1
    data_30min['buy_signal_2_1'] = np.where(
        (data_30min['close'] < data_30min['bollinger_mid']) &  # Latest price < BOLL(MID)
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['short_ma'] > data_30min['long_ma']),  # MA5 > MA10
        1, 0)

    # Generate buy signals for the second case, Condition 2
    data_30min['buy_signal_2_2'] = np.where(
        (data_30min['close'] < data_30min['bollinger_mid']) &  # Latest price < BOLL(MID)
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['close'] >= data_30min['long_ma']) &  # Latest price >= MA10
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['long_ma'] > data_30min['long_ma'].shift(1)),  # MA10 is in an upward trend
        1, 0)

    # Generate buy signals for the third case
    data_30min['buy_signal_3'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['close'] >= data_30min['long_ma']) &  # Latest price >= MA10
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)),  # MA5 is in an upward trend
        1, 0)

    # Generate buy signals for the fourth case, Condition 1
    data_30min['buy_signal_4_1'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['short_ma'] > data_30min['long_ma']),  # MA5 > MA10
        1, 0)

    # Generate buy signals for the fourth case, Condition 2
    data_30min['buy_signal_4_2'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['close'] >= data_30min['long_ma']) &  # Latest price >= MA10
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['long_ma'] > data_30min['long_ma'].shift(1)),  # MA10 is in an upward trend
        1, 0)

    # Generate buy signals for the fifth case
    data_30min['buy_signal_5'] = np.where(
        (data_30min['close'] > data_30min['daily_open']) &  # Latest price > daily open price
        (data_30min['close'] > data_30min['short_ma']) &  # Latest price > MA5
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['close'] > data_30min['bollinger_upper']) &  # Latest price > BOLL(UPPER)
        (data_30min['bollinger_width'] >= data_30min['bollinger_width'].shift(30)),  # Bollinger Band width is increasing
        1, 0)

    # Generate buy signals for the sixth case, Condition 1
    data_30min['buy_signal_6_1'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['short_ma'] > data_30min['long_ma']),  # MA5 > MA10
        1, 0)

    # Generate buy signals for the sixth case, Condition 2
    data_30min['buy_signal_6_2'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['close'] >= data_30min['long_ma']) &  # Latest price >= MA10
        (data_30min['short_ma'] > data_30min['short_ma'].shift(1)) &  # MA5 is in an upward trend
        (data_30min['long_ma'] > data_30min['long_ma'].shift(1)),  # MA10 is in an upward trend
        1, 0)

    # Generate buy signals for the sixth case, Condition 3
    data_30min['buy_signal_6_3'] = np.where(
        (data_30min['close'] >= data_30min['short_ma']) &  # Latest price >= MA5
        (data_30min['close'] >= data_30min['high'].shift(1)),  # Latest price >= highest price of the last 30-minute candle
        1, 0)

    # Combine buy signals
    data_30min['buy_signal'] = data_30min[['buy_signal_1_1', 'buy_signal_2_1', 'buy_signal_2_2', 'buy_signal_3', 'buy_signal_4_1', 'buy_signal_4_2', 'buy_signal_5', 'buy_signal_6_1', 'buy_signal_6_2', 'buy_signal_6_3']].max(axis=1)

    # Apply the overall condition for the first four cases
    overall_condition = (data_30min['close'] > data_30min['daily_avg']) & (data_30min['close'] > data_30min['daily_open'])
    data_30min.loc[~overall_condition, ['buy_signal_1_1', 'buy_signal_2_1', 'buy_signal_2_2', 'buy_signal_3', 'buy_signal_4_1', 'buy_signal_4_2']] = 0

    return data_30min

def avoid_open_position_signals(data_30min):
    """Generate signals to avoid opening positions based on defined criteria."""
    # Generate avoid open signals based on specified criteria
    avoid_open_conditions = (
        (data_30min['close'] >= data_30min['high'].rolling(window=30).max()) &  # Latest price >= 30min max (MA)
        (data_30min['bollinger_width'] > data_30min['bollinger_width'].shift(60)) &  # 60min Bollinger Band width is increasing
        (data_30min['30min_Diff'] > data_30min['30min_Dea']) &  # 30min Diff > Dea
        (data_30min['30min_Dea'] > data_30min['30min_Dea'].shift(1))  # 30min Dea is in an upward trend
    )

    # Set buy signal to 0 where avoid open conditions are met
    data_30min.loc[avoid_open_conditions, 'buy_signal'] = 0

    return data_30min

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

def close_position_signals(data_30min):
    """Generate close position signals based on the defined criteria."""
    data_30min['conservative_profit_target'] = calculate_conservative_profit_target(
        data_30min['open'], data_30min['long_ma'])

    data_30min['moving_stop_loss'] = np.nan

    for i in range(1, len(data_30min)):
        data_30min.loc[data_30min.index[i], 'moving_stop_loss'] = calculate_moving_stop_loss(
            data_30min['close'].iloc[i], 
            data_30min['low'].shift(1).iloc[i], 
            data_30min['long_ma'].shift(1).iloc[i], 
            data_30min['high'].cummax().iloc[i])
    
    data_30min.index = pd.to_datetime(data_30min.index) 
    data_30min['sell_signal'] = 0
    
    # Use the new calculate_holding_time function
    data_30min = calculate_holding_time(data_30min)
    data_30min.loc[data_30min['holding_time'] > 15, 'sell_signal'] = -1

    data_30min['sell_signal'] = np.where(
        (data_30min['close'] >= data_30min['conservative_profit_target']) |
        (data_30min['close'] <= data_30min['moving_stop_loss']),
        -1, data_30min['sell_signal']
    )

    data_30min['sell_signal'] = np.where(
        (data_30min['close'] > data_30min['moving_stop_loss']) &
        (data_30min['close'] < data_30min['conservative_profit_target']),
        -0.5, data_30min['sell_signal']
    )

    end_of_day = '14:59'
    data_30min['sell_signal'] = np.where(
        data_30min.index.time == pd.to_datetime(end_of_day).time(),
        -1, data_30min['sell_signal']
    )

    return data_30min

def generate_regime_based_signals(data):
    """Generate buy/sell signals based on detected market regime."""
    data['signal'] = 0
    
    for regime in data['regime'].unique():
        regime_data = data[data['regime'] == regime]

        if regime == 0:  # Trending regime
            regime_data = strat.generate_signals_trending(regime_data)
        elif regime == 1:  # Range-bound regime
            regime_data = strat.generate_signals_range_bound(regime_data)
        elif regime == 2:  # Volatile regime
            regime_data = strat.generate_signals_volatile(regime_data)

        data.loc[data['regime'] == regime, 'signal'] = regime_data['signal']
    
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