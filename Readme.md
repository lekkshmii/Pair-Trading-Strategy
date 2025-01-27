# Pair Trading Strategy

A Python implementation of a **pair trading strategy** that combines statistical analysis, fundamental metrics, and backtesting to identify optimal stock pairs for trading. This tool is designed to assist in the analysis and evaluation of potential trading opportunities using a systematic approach.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7+
- Required Python packages (listed in `requirements.txt`). Install them with:
  ```bash
  pip install -r requirements.txt
  ```

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/lekkshmii/Pair-Trading-Strategy.git
   cd Pair-Trading-Strategy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:
   ```bash
   python pair_trading.py
   ```

4. Follow the interactive menu to explore pair trading analyses.

---

## How It Works

1. **Fetch Data**: Retrieves historical prices and calculates sector-wide metrics for the selected stocks.
2. **Analyze Pairs**: Combines statistical and fundamental metrics to rank pairs based on performance and compatibility.
3. **Backtest Strategies**: Simulates trading strategies with configurable parameters.
4. **Visualize Insights**: Provides intuitive charts and heatmaps for better decision-making.

---

## Example Usage

1. Define your stock tickers in the `main()` function:
   ```python
   tickers = ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM']
   ```

2. Analyze the best pair:
   - Use the "Show Best Pair Analysis" menu option to view detailed metrics and visualizations for the top-performing pair.

3. Explore sector-wide analysis:
   - View the correlation matrix and sector performance metrics to understand broader trends.

---

## Dependencies

- `yfinance`: For fetching stock data.
- `pandas`: For data manipulation.
- `numpy`: For numerical computations.
- `matplotlib`: For visualizations.
- `seaborn`: For heatmaps and advanced plotting.
- `statsmodels`: For cointegration tests.

---

## Project Structure

```
Pair-Trading-Strategy/
├── pair_trading.py           # Main program file
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
└── data/                     # (Optional) Folder to store fetched data
```

---

## Future Enhancements

- Add support for live market data.
- Integrate machine learning models for pair selection.
- Include automated trade execution using APIs.

---

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## Author

Developed by Lekshmi. If you have any questions or suggestions, feel free to reach out or submit an issue on GitHub.

---

## Acknowledgements

- Special thanks to the developers of `yfinance`, `statsmodels`, and `seaborn` for providing the tools that made this project possible.

