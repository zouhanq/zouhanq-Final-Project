import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_strategy(data):
    """
    Evaluate the performance of the strategy
    :param data: dataframe, contains single trade returns data
    :return results: dict, evaluation metrics
    """
    # Evaluate strategy performance: total return, annualized return, maximum drawdown, Sharpe ratio
    data = calculate_cum_prof(data)

    # Get total return
    total_return = data['cum_profit'].iloc[-1]
    # Calculate annualized return (assuming monthly trades)
    annual_return = data['profit_pct'].mean() * 12

    # Calculate maximum drawdown over the past year
    data = caculate_max_drawdown(data, window=12)
    # print(data)
    # Get maximum drawdown over the past year
    max_drawdown = data['max_dd'].iloc[-1]

    # Calculate Sharpe ratio
    sharpe, annual_sharpe = calculate_sharpe(data)

    # Store metrics in a dictionary
    results = {'Total Return': total_return, 'Annualized Return': annual_return,
               'Max Drawdown': max_drawdown, 'Sharpe Ratio': annual_sharpe}

    # Print evaluation metrics
    for key, value in results.items():
        print(key, value)

    return data


def compose_signal(data):
    """
    Combine signals
    :param data: DataFrame
    :return: DataFrame
    """
    # Initialize position state
    position = 0  # 0: no position, 1: full position, 0.5: half position

    # Lists to store signals
    buy_signals = []
    sell_signals = []

    for i in range(len(data)):
        if data['buy_signal'].iloc[i] == 1 and position == 0:
            buy_signals.append(1)
            sell_signals.append(0)
            position = 1
        elif data['sell_signal'].iloc[i] == -1 and position == 1:
            buy_signals.append(0)
            sell_signals.append(-1)
            position = 0
        elif data['sell_signal'].iloc[i] == -0.5 and position == 1:
            buy_signals.append(0)
            sell_signals.append(-0.5)
            position = 0.5
        elif data['sell_signal'].iloc[i] == -0.5 and position == 0.5:
            buy_signals.append(0)
            sell_signals.append(-0.5)
            position = 0
        else:
            buy_signals.append(0)
            sell_signals.append(0)

    # Add signals back to the DataFrame
    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals

    # Combine buy and sell signals into a single signal column
    data['signal'] = data['buy_signal'] + data['sell_signal']

    return data


def calculate_prof_pct(data):
    """
    Calculate single trade returns: opening and closing (for the entire position size)
    :param data: DataFrame, contains signals and closing prices
    :return: DataFrame, contains returns for each trade
    """
    data['profit_pct'] = np.nan
    position = 0  # Track the current position size: 1 for full position, 0.5 for half position
    entry_price = 0  # Track the entry price for the current position

    for i in range(len(data)):
        if data['signal'].iloc[i] == 1 and position == 0:
            # Entering a full position
            position = 1
            entry_price = data['close'].iloc[i]
        elif data['signal'].iloc[i] == -0.5 and position == 1:
            # Exiting half of the position
            exit_price = data['close'].iloc[i]
            data['profit_pct'].iloc[i] = (exit_price - entry_price) / entry_price / 2  # Half of the position
            entry_price = exit_price  # Update entry price for remaining position
            position = 0.5
        elif data['signal'].iloc[i] == -1 and position == 1:
            # Exiting the full position
            exit_price = data['close'].iloc[i]
            data['profit_pct'].iloc[i] = (exit_price - entry_price) / entry_price
            position = 0
            entry_price = 0
        elif data['signal'].iloc[i] == -1 and position == 0.5:
            # Exiting the remaining half position
            exit_price = data['close'].iloc[i]
            data['profit_pct'].iloc[i] = (exit_price - entry_price) / entry_price / 2
            position = 0
            entry_price = 0

    data = data.dropna(subset=['profit_pct'])  # Drop rows where 'profit_pct' is NaN
    return data


def calculate_cum_prof(data):
    """
    Calculate cumulative returns (individual stock returns)
    """
    # Cumulative return
    data['cum_profit'] = pd.DataFrame(1 + data['profit_pct']).cumprod() - 1
    return data


def caculate_max_drawdown(data, window=252):
    """
    Calculate maximum drawdown
    :param data:
    :param window: int, time window setting, default is 252 (daily)
    :return:
    """
    # Simulate holding amount: total invested amount * (1 + return rate)
    data['close'] = 10000 * (1 + data['cum_profit'])
    # Select the maximum net value within the time period
    data['roll_max'] = data['close'].rolling(window=window, min_periods=1).max()
    # Calculate daily drawdown ratio = (trough - peak) / peak = trough / peak - 1
    data['daily_dd'] = data['close'] / data['roll_max'] - 1
    # Select the maximum drawdown within the time period
    data['max_dd'] = data['daily_dd'].rolling(window, min_periods=1).min()

    return data


def calculate_sharpe(data):
    """
    Calculate Sharpe ratio, returns annualized Sharpe ratio
    :param data: dataframe, stock
    :return: float
    """
    # Formula: sharpe = (mean return - risk-free rate) / standard deviation of returns
    daily_return = data['profit_pct']  # Returns after applying the strategy
    avg_return = daily_return.mean()
    sd_return = daily_return.std()
    # Calculate Sharpe: daily return * 252 = annualized return
    sharpe = avg_return / sd_return
    sharpe_year = sharpe * np.sqrt(252)
    return sharpe, sharpe_year

def calculate_win_rate(data):
    """
    Calculate win rate: number of positive trades / total trades
    :param data: dataframe, contains single trade returns data
    :return win_rate: float, win rate
    """
    total_trades = data['profit_pct'].count()
    winning_trades = data[data['profit_pct'] > 0]['profit_pct'].count()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    return win_rate

def apply_risk_management(data, stop_loss=0.02, take_profit=0.05):
    data['stop_loss'] = data['close'] * (1 - stop_loss)
    data['take_profit'] = data['close'] * (1 + take_profit)
    position = 0  # Track the position state: 1 for long, 0 for no position

    for i in range(1, len(data)):
        if data.loc[i, 'signal'] == 1 and position == 0:  # Buy signal
            position = 1  # Enter position
            entry_price = data.loc[i, 'close']
            data.loc[i, 'stop_loss'] = entry_price * (1 - stop_loss)
            data.loc[i, 'take_profit'] = entry_price * (1 + take_profit)
        elif position == 1:  # If in position, check risk management criteria
            if data.loc[i, 'low'] < data.loc[i, 'stop_loss']:  # Stop loss condition
                data.loc[i, 'signal'] = -1  # Exit position
                position = 0  # Reset position
            elif data.loc[i, 'high'] > data.loc[i, 'take_profit']:  # Take profit condition
                data.loc[i, 'signal'] = -1  # Exit position
                position = 0  # Reset position
            else:
                data.loc[i, 'signal'] = 0  # Maintain position, reset the signal to 0
        else:
            data.loc[i, 'signal'] = 0  # No position or buy signal, reset the signal to 0

    return data

def generate_signals_trending(data):
    """Generate buy/sell signals for a trending market."""
    data['signal'] = np.where(data['short_ma'] > data['long_ma'], 1, -1)
    return data

def generate_signals_range_bound(data):
    """Generate buy/sell signals for a range-bound market."""
    data['signal'] = np.where(data['close'] < data['bollinger_lower'], 1,
                              np.where(data['close'] > data['bollinger_upper'], -1, 0))
    return data

def generate_signals_volatile(data):
    """Generate buy/sell signals for a volatile market."""
    data['signal'] = np.where((data['close'] > data['vwap']) & (data['rsi'] < 70), 1,
                              np.where((data['close'] < data['vwap']) & (data['rsi'] > 30), -1, 0))
    return data

class Strategy:
    def __init__(self):
        self.position = 0  # Track the current position size: 0 (no position), 0.5 (half position), 1 (full position)
        self.entry_price = 0

    def process_signal(self, current_price, buy_signal, sell_signal):
        """
        Process the signal for instant operation.
        
        :param current_price: float, current price of the asset
        :param buy_signal: int, 1 for buy, 0 otherwise
        :param sell_signal: float, -0.5 for half sell, -1 for full sell, 0 otherwise
        :return: tuple, (signal, new_position)
        """
        signal = 0

        if buy_signal == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            signal = 1  # Buy signal
        elif sell_signal == -1 and self.position == 1:
            self.position = 0
            signal = -1  # Full sell signal
        elif sell_signal == -0.5 and self.position == 1:
            self.position = 0.5
            signal = -0.5  # Half sell signal
        elif sell_signal == -0.5 and self.position == 0.5:
            self.position = 0
            signal = -0.5  # Half sell signal to close remaining position

        return signal, self.position
