# backtesting.py

import matplotlib.pyplot as plt
import pandas as pd
import base as strat
from ma_strategy import ma_strategy  # Import the ma_strategy function

def backtest(data):
    """Backtest the strategy by calculating returns and cumulative profit."""
    # Calculate profit percentages for each trade
    data = strat.calculate_prof_pct(data)
    
    # Calculate cumulative profit over time
    data = strat.calculate_cum_prof(data)

    # Plot cumulative profit
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['cum_profit'], label='Cumulative Profit')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Profit')
    plt.title('Backtest Performance')
    plt.legend()
    plt.show()

    return data

# Example usage in the main script
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r'data\mink.csv')
    df['Time'] = pd.to_datetime(df['Time'])

    # Apply the trading strategy
    result = ma_strategy(df)

    # Run backtesting on the strategy output
    backtest(result)
