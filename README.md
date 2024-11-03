# Regime-Based Moving Average Strategy for CSI 300 Futures

## 1. Video Presentation Link
[YouTube Presentation Link](your_youtube_link_here)

---

## 2. Preliminary Visualizations

### Cumulative Profit Over Time
This chart shows the cumulative profit over time for the regime-based moving average strategy. Although initially positive, the performance has been steadily declining, which suggests that the strategy may require further tuning.

![Cumulative Profit](link_to_your_image)

### Market Regimes Detected by K-means
Using K-means clustering, we identified three distinct market regimes represented by clusters. Each cluster may represent different market conditions, such as trending, mean-reverting, and moderate conditions.

![K-means Clustering](link_to_your_image)

```python
# Code snippet for visualizations

# Plotting cumulative profit over time
plt.plot(result['Time'], result['cum_profit'], label='Cumulative Profit')
plt.xlabel('Time')
plt.ylabel('Cumulative Profit')
plt.title('Backtest Performance')
plt.legend()
plt.show()

# Plotting K-means clustering results
plt.scatter(data['svd_component_1'], data['svd_component_2'], c=data['regime'], cmap='viridis')
plt.colorbar(label='Regime')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.title('Market Regimes Detected by K-means')
plt.show()


## 3. Data Processing

### Data Loading and Cleaning
- **Data Source**: Historical price data for CSI 300 futures.
- **Cleaning**: Checked for missing values and forward-filled any gaps to ensure continuity.
  
### Feature Engineering
To better understand and respond to market conditions, several technical indicators were added to the dataset:
  - **RSI**: Used to gauge overbought and oversold conditions.
  - **Moving Averages (MA)**: Short (5-period) and Long (10-period) moving averages were used to identify trends.
  - **Bollinger Bands**: To detect potential mean-reverting opportunities.
  - **SVD Components**: Performed Singular Value Decomposition on key features to reduce dimensionality for clustering.

### K-means Clustering for Regime Detection
Using the SVD components, we applied K-means clustering to segment the data into three market regimes:
  - **Regime 0 (Purple Cluster)**: This regime appears to represent stable, low-volatility conditions.
  - **Regime 1 (Yellow Cluster)**: Represents higher volatility, potentially trending conditions.
  - **Regime 2 (Teal Cluster)**: Indicates moderate volatility or mixed conditions.

Each row in the dataset was labeled with its corresponding regime, allowing us to apply tailored strategies for each regime type.

---

## 4. Data Modeling Methods

### Regime-Specific Strategies
Different strategies were implemented for each detected regime:
  - **Trending Regime**: In a trending regime, a Moving Average crossover strategy is applied, where buy signals are generated when the short MA crosses above the long MA, and sell signals when it crosses below.
  - **Mean-Reverting Regime**: Bollinger Bands are used to capture mean-reversion opportunities. Buy signals are generated when the price is below the lower band, and sell signals when it is above the upper band.
  - **Moderate/Conservative Regime**: A more cautious strategy is applied using daily open price and RSI levels to confirm signals.

### Signal Generation
For each data point, buy or sell signals are generated based on the detected regime:
  - If the current regime is trending, signals follow a Moving Average crossover approach.
  - If the regime is mean-reverting, Bollinger Bands dictate the entry/exit signals.
  - For moderate regimes, a more conservative approach is taken to minimize risk in uncertain conditions.

Each strategy is optimized to align with the characteristics of its respective regime, allowing for a more responsive and potentially profitable trading approach.

---

## 5. Preliminary Results

### K-means Clustering Results
The K-means clustering produced three distinct regimes. Preliminary analysis suggests that these regimes could correspond to different market conditions, such as trending, mean-reverting, and moderate states. The clustering provides a basis for applying tailored strategies to better align with current market conditions.

### Backtest Performance
The cumulative profit over time chart reveals an overall decline in performance, indicating that the strategy may not yet be optimized for each regime. Initial gains were observed, but as the backtest continued, the performance steadily declined. This suggests potential misalignment between the applied strategies and the actual market behavior within each regime.

### Challenges and Next Steps
1. **Challenges**:
   - **Parameter Tuning**: The current parameters for each strategy may not be optimal for the different regimes. Further tuning is necessary to improve performance.
   - **Lagging Indicators**: Moving averages and other indicators used may have lagged behind rapid market changes, resulting in delayed entries and exits.
   
2. **Next Steps**:
   - **Parameter Optimization**: We plan to fine-tune the Moving Average window sizes, RSI thresholds, and Bollinger Band widths for each regime.
   - **Exploring Additional Features**: Considering adding volatility indicators like ATR to better capture market conditions.
   - **Enhanced Regime Detection**: Experimenting with alternative clustering methods to see if more refined regimes can be identified.



