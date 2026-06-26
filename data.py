# data.py
# Downloads and caches historical price data for a basket of stocks

import yfinance as yf
import pandas as pd

# Our universe - a diversified basket of large-cap stocks across sectors 
# This is our "investment universe" - the stocks the strategy can trade
TICKERS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', # Technology
  'JPM', 'GS', 'BAC', 'MS', 'BLK', # Finance
  'XOM', 'CVX', 'COP', 'SLB', 'EOG', # Energy
  'JNJ', 'PFE', 'UNH', 'ABT', 'MRK', # Healthcare
  'SPY'                               # S&P 500 benchmark ETF
]

def download_prices(tickers=TICKERS, start='2018-01-01', end='2024-12-31'):
  """
  Download adjusted closing prices for all tickers.
  Adjusted close accounts for dividends & stock splits - always use this, not raw close, for backtesting (common interview question)
  """
print (f"Downloading price data for {len(tickers)} tickers...")

raw = yf.download(tickers, start=start, end=end, auto_adjust=True)
prices = raw['Close']

# Drop any ticker that has more than 20% missing data
    threshold = 0.8
    prices = prices.dropna(thresh=int(len(prices) * threshold), axis=1)
    
    # Forward-fill remaining gaps (e.g. holidays where some stocks didn't trade)
    prices = prices.ffill()
    
    print(f"Downloaded {len(prices)} trading days of data.")
    print(f"Tickers available: {list(prices.columns)}")
    
    return prices


def compute_returns(prices):
    """
    Compute daily log returns.
    We use log returns (not simple returns) because:
    - They are time-additive: We can sum daily log returns to get the period return
    - They are more statistically well-behaved (closer to normally distributed)
    - This is standard in quantitative finance
    """
    import numpy as np
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


if __name__ == '__main__':
    prices = download_prices()
    returns = compute_returns(prices)
    print("\nFirst 5 rows of returns:")
    print(returns.head())
    print(f"\nShape: {returns.shape} (days x stocks)")
