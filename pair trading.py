import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

class EnhancedPairSelector:
    def __init__(self, tickers, start_date='2019-01-01', end_date='2023-12-31'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.sector_data = None
        self.all_pairs_results = []

    def fetch_data(self):
        """Fetch price data and calculate sector metrics"""
        df_list = []
        fundamental_data = {}
        
        print("Fetching data for all stocks...")
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(f'{ticker}.NS')
                data = stock.history(start=self.start_date, end=self.end_date)
                df_list.append(data['Close'].rename(ticker))
                
                info = stock.info
                fundamental_data[ticker] = {
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'profit_margins': info.get('profitMargins', 0)
                }
                print(f"✓ {ticker} data fetched successfully")
            except Exception as e:
                print(f"✗ Error fetching {ticker}: {e}")
        
        self.data = pd.concat(df_list, axis=1).ffill().dropna()
        
        # Calculate sector metrics
        sector_returns = self.data.pct_change()
        self.sector_data = {
            'sector_return': sector_returns.mean(axis=1),
            'sector_vol': sector_returns.std(axis=1),
            'fundamentals': fundamental_data
        }
        return self.data

    def backtest_pair(self, pair, price_data, window=30, entry_z=2.0, exit_z=0.5):
        """Backtest a pair trading strategy"""
        prices = price_data[list(pair)]
        norm_prices = prices.div(prices.iloc[0])
        spread = norm_prices[pair[0]] - norm_prices[pair[1]]
        
        # Calculate z-score
        roll_mean = spread.rolling(window=window, min_periods=1).mean()
        roll_std = spread.rolling(window=window, min_periods=1).std()
        z_score = (spread - roll_mean) / roll_std
        
        # Generate positions
        positions = pd.Series(0, index=prices.index)
        positions.loc[z_score < -entry_z] = 1
        positions.loc[z_score > entry_z] = -1
        positions.loc[abs(z_score) < exit_z] = 0
        
        # Calculate returns
        pair_returns = prices.pct_change()
        strategy_returns = positions.shift(1) * (pair_returns[pair[0]] - pair_returns[pair[1]])
        strategy_returns = strategy_returns.dropna()
        
        # Calculate metrics
        if len(strategy_returns) > 0:
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
            max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        else:
            total_return = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': (positions.diff() != 0).sum() // 2
        }

    def compare_fundamentals(self, pair):
        """Compare fundamental metrics between companies"""
        fund_data = self.sector_data['fundamentals']
        stock1, stock2 = pair
        
        # Calculate relative differences
        metrics = {
            'market_cap_diff': abs(fund_data[stock1]['market_cap'] - fund_data[stock2]['market_cap']) / max(fund_data[stock1]['market_cap'], fund_data[stock2]['market_cap']),
            'pe_diff': abs(fund_data[stock1]['pe_ratio'] - fund_data[stock2]['pe_ratio']) / max(fund_data[stock1]['pe_ratio'], fund_data[stock2]['pe_ratio']) if fund_data[stock1]['pe_ratio'] and fund_data[stock2]['pe_ratio'] else 1,
            'growth_diff': abs(fund_data[stock1]['revenue_growth'] - fund_data[stock2]['revenue_growth']),
            'margin_diff': abs(fund_data[stock1]['profit_margins'] - fund_data[stock2]['profit_margins'])
        }
        
        # Calculate similarity score (lower is better)
        similarity_score = np.mean(list(metrics.values()))
        
        return {
            'metrics': metrics,
            'similarity_score': similarity_score
        }

    def analyze_pair(self, pair):
        """Comprehensive analysis of a single pair"""
        prices = self.data[list(pair)]
        returns = prices.pct_change().dropna()
        
        # Statistical Analysis
        correlation = returns.corr().iloc[0, 1]
        score, pvalue, _ = coint(prices[pair[0]], prices[pair[1]])
        
        # Performance Analysis
        metrics = self.backtest_pair(pair, self.data)
        
        # Fundamental Comparison
        fundamental_comparison = self.compare_fundamentals(pair)
        
        return {
            'pair': pair,
            'correlation': correlation,
            'cointegration_pvalue': pvalue,
            'metrics': metrics,
            'fundamentals': fundamental_comparison
        }

    def find_optimal_pairs(self):
        """Find and rank all possible pairs"""
        if self.data is None:
            self.fetch_data()
        
        print("\nAnalyzing all possible pairs...")
        pairs_results = []
        
        for pair in combinations(self.tickers, 2):
            print(f"Analyzing {pair[0]} - {pair[1]}...")
            analysis = self.analyze_pair(pair)
            
            # Calculate composite score
            score = (
                0.3 * analysis['metrics']['sharpe'] +
                0.3 * (1 - analysis['cointegration_pvalue']) +
                0.2 * abs(analysis['correlation']) +
                0.2 * analysis['metrics']['total_return']
            )
            
            analysis['composite_score'] = score
            pairs_results.append(analysis)
        
        # Sort by composite score
        self.all_pairs_results = sorted(pairs_results, 
                                      key=lambda x: x['composite_score'], 
                                      reverse=True)
        
        return self.all_pairs_results

    def show_best_pair(self):
        """Display detailed analysis of the best pair"""
        best_pair = self.all_pairs_results[0]
        pair = best_pair['pair']
        
        print(f"\nBest Pair Analysis: {pair[0]} - {pair[1]}")
        print("="*50)
        
        print("\nKey Metrics:")
        print(f"Composite Score: {best_pair['composite_score']:.4f}")
        print(f"Correlation: {best_pair['correlation']:.4f}")
        print(f"Cointegration p-value: {best_pair['cointegration_pvalue']:.4f}")
        
        print("\nPerformance Metrics:")
        metrics = best_pair['metrics']
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        self.plot_pair_analysis(pair)

    def show_all_pairs(self):
        """Display ranking of all pairs"""
        print("\nAll Pairs Ranking")
        print("="*50)
        
        for i, result in enumerate(self.all_pairs_results, 1):
            pair = result['pair']
            print(f"\n{i}. {pair[0]} - {pair[1]}")
            print(f"   Composite Score: {result['composite_score']:.4f}")
            print(f"   Sharpe Ratio: {result['metrics']['sharpe']:.2f}")
            print(f"   Total Return: {result['metrics']['total_return']:.2%}")

    def show_sector_analysis(self):
        """Display sector-wide analysis"""
        print("\nSector Analysis")
        print("="*50)
        
        # Calculate sector metrics
        returns = self.data.pct_change()
        corr_matrix = returns.corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
        print("\nSector Statistics:")
        print(f"Average Daily Return: {returns.mean().mean():.4%}")
        print(f"Average Volatility: {returns.std().mean():.4%}")
        
        # Plot sector correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Sector Correlation Heatmap')
        plt.show()

    def show_pair_selection(self):
        """Allow user to select and analyze specific pair"""
        print("\nAvailable Stocks:")
        for i, ticker in enumerate(self.tickers, 1):
            print(f"{i}. {ticker}")
        
        try:
            idx1 = int(input("\nSelect first stock (number): ")) - 1
            idx2 = int(input("Select second stock (number): ")) - 1
            
            if 0 <= idx1 < len(self.tickers) and 0 <= idx2 < len(self.tickers):
                pair = (self.tickers[idx1], self.tickers[idx2])
                analysis = self.analyze_pair(pair)
                
                print(f"\nAnalysis for {pair[0]} - {pair[1]}")
                print("="*50)
                print(f"Correlation: {analysis['correlation']:.4f}")
                print(f"Cointegration p-value: {analysis['cointegration_pvalue']:.4f}")
                print(f"Sharpe Ratio: {analysis['metrics']['sharpe']:.2f}")
                print(f"Total Return: {analysis['metrics']['total_return']:.2%}")
                
                self.plot_pair_analysis(pair)
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter valid numbers!")

    def show_performance_comparison(self):
        """Display performance comparison of all pairs"""
        returns = [result['metrics']['total_return'] for result in self.all_pairs_results]
        sharpes = [result['metrics']['sharpe'] for result in self.all_pairs_results]
        pairs = [f"{p['pair'][0]}-{p['pair'][1]}" for p in self.all_pairs_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot returns comparison
        ax1.bar(pairs, returns)
        ax1.set_title('Total Returns Comparison')
        ax1.set_xticklabels(pairs, rotation=45)
        ax1.grid(True)
        
        # Plot Sharpe ratio comparison
        ax2.bar(pairs, sharpes)
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_xticklabels(pairs, rotation=45)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_pair_analysis(self, pair):
        """Create and display analysis plots for a pair"""
        fig = plt.figure(figsize=(15, 10))
        
        # Price plot
        plt.subplot(311)
        norm_prices = self.data[list(pair)] / self.data[list(pair)].iloc[0]
        norm_prices.plot()
        plt.title(f'Normalized Prices: {pair[0]} vs {pair[1]}')
        plt.grid(True)
        
        # Spread plot
        plt.subplot(312)
        spread = norm_prices[pair[0]] - norm_prices[pair[1]]
        spread.plot()
        plt.title('Price Spread')
        plt.grid(True)
        
        # Return correlation plot
        plt.subplot(313)
        returns = self.data[list(pair)].pct_change()
        plt.scatter(returns[pair[0]], returns[pair[1]], alpha=0.5)
        plt.xlabel(f'{pair[0]} Returns')
        plt.ylabel(f'{pair[1]} Returns')
        plt.title('Return Correlation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def display_menu(self):
        """Display interactive menu for user"""
        while True:
            print("\n" + "="*50)
            print("Pair Trading Analysis Menu")
            print("="*50)
            print("1. Show Best Pair Analysis")
            print("2. Show All Pairs Ranking")
            print("3. Show Sector Analysis")
            print("4. Show Detailed Analysis for Specific Pair")
            print("5. Show Performance Comparison Plot")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                self.show_best_pair()
            elif choice == '2':
                self.show_all_pairs()
            elif choice == '3':
                self.show_sector_analysis()
            elif choice == '4':
                self.show_pair_selection()
            elif choice == '5':
                self.show_performance_comparison()
            elif choice == '6':
                print("\nExiting program...")
                break
            else:
                print("\nInvalid choice. Please try again.")

def main():
    # Define IT sector stocks
    tickers = ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM']
    
    # Initialize and run analysis
    selector = EnhancedPairSelector(tickers)
    selector.find_optimal_pairs()
    
    # Start interactive menu
    selector.display_menu()

if __name__ == "__main__":
    main()