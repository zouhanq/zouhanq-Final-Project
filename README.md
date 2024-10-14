# Development of Moving Average Strategies for Quantitative Trading

## **Project Description**
The aim of this project is to develop and evaluate a quantitative trading strategy using Moving Averages (MA) focused on China's stock index futures, specifically the CSI 300 (沪深300). The project will involve backtesting the strategy on historical data and refining the parameters to achieve consistent positive returns. This study will explore various configurations of MA strategies to identify the optimal approach for trading in the CSI 300 futures market.

## **Goals**
- Successfully develop a Moving Average strategy that achieves a positive return on China's stock index futures (CSI 300).
- Backtest the strategy on historical data to evaluate its effectiveness.
- Analyze the profitability, risk, and robustness of the strategy under different market conditions.

## **Data Collection**
- **Data Source**: Historical data of CSI 300 futures will be collected using publicly available datasets from financial data platforms like Wind, CSMAR, or brokerage data services such as OpenCTP, as well as other APIs that provide Chinese stock market data.
- **Data Frequency**: Minute-level or daily data will be used, depending on the feasibility and computational requirements.
- **Data Attributes**: OHLC (Open, High, Low, Close), trading volume, and contract-specific details such as expiration date and tick size will be gathered.

## **Data Modeling Approach**
- **Trading Strategy**: Implement Moving Average strategies such as:
  - **Simple Moving Average (SMA)**: Comparing short-term and long-term SMAs to generate trading signals.
  - **Exponential Moving Average (EMA)**: Using EMA for more responsive signal generation.
  - **Crossover Strategy**: Implementing the Golden Cross/Death Cross strategy (short-term MA crossing over/under long-term MA) to identify potential entry and exit points.
  - **Market Conditions**: The SMA and EMA strategies will be tested under trending and range-bound markets to determine which configuration works better under each condition. In trending markets, the crossover strategy might generate more reliable signals, while in range-bound markets, shorter MAs may reduce noise.
- **Hyperparameter Tuning**: Stop-loss and take-profit levels will be optimized through hyperparameter tuning. This will involve testing different values for MA lengths, stop-loss thresholds, and profit-taking levels to maximize the Sharpe ratio and minimize drawdowns.

## **Data Visualization**
- **Performance Metrics**: Visualize the cumulative returns of the strategy compared to a benchmark (e.g., buy-and-hold approach of CSI 300 futures) using line charts.
- **MA Signals**: Overlay MA lines on price charts to show entry and exit points.
- **Risk Analysis**: Utilize drawdown charts to visualize periods of loss and recovery.
- **Interactive Tools**: Create an interactive dashboard to adjust MA parameters and observe their impact on backtest results.

## **Test Plan**
- **Data Split**: Use historical data from January 2018 to December 2022:
  - **Training Data**: January 2018 to December 2021 (80%)
  - **Testing Data**: January 2022 to December 2022 (20%)
- **Handling Data Gaps**: Missing data points or anomalies (such as extreme outliers) will be addressed by forward-filling missing values or using statistical methods like interpolation. Significant anomalies will be ignored or flagged for manual review.
- **Validation**: Evaluate the model's performance using metrics such as annualized return, Sharpe ratio, maximum drawdown, and win rate.
- **Robustness Testing**: Test the strategy under different market conditions, including bullish, bearish, and sideways trends.

## **GitHub Repository**
This repository contains all project code, datasets, documentation, and visualization tools. The README.md will be updated regularly to reflect progress and findings.
