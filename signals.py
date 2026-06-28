#signals.py
#computes momentum signals and volatility estimates

import pandas as pd
import numpy as np

def momentum_signal(returns, lookback=126):
  """
  Momentum signal: past N-day cumulative return for each stock.

  The intuition: stocks that have performed well recently tend to 
  continue performing well in the short-to-medium term. This is one 
  of the most well-documented anomalies in academic finance (Jegadeesh 
  & Titman, 1993 - a classic paper worth knowing for interviews).

  We use 126 days (~6 months) as the default lookback.
  We skip the most recent month (21 days) to avoid short-term reversal, 
  which is also well-documented in the literature.

  Parameters:
    returns: DataFrame of momentum scores, same shape as returns
  """
  # Sum log returns over lookback window, skipping most recent 21 days
  momentum = returns.shift(21).rolling(window=lookback).sum()
  return momentum

def rolling_volatility(returns, window=21):
  """
  Rolling annualised volatility for each stock.

  Volatility = standard deviation of daily returns * sqrt(252)
  We annualise by multiplying by sqrt(252) because there are ~252
  trading days in a year. This is standard in finance.

  We use this to:
  1. Scale position sizes (volatility targeting)
  2. Assess strategy risk
  """
  daily_vol = returns.rolling(window=window).std()
  annualised_vol = daily_vol * np.sqrt(252)
  return annualised_vol

def rank_signal(signal_df):
  """
  Cross-sectional rank of the momentum signal.

  Instead of using raw momentum scores, we rank stocks each day.
  Rank 1 = lowest momentum, Rank N = highest momentum.
  This makes the signal robust to outliers and market regimes.

  This is called "cross-sectional normalisation" - a standard
  technique in systematic trading.
  """
  return signal_df.rank(axis=1, pct=True) # pct=True gives rank as percentile (0 to 1)

if __name__ == '__main__':
  from data import download_prices, compute_returns

  prices = download_prices()
  returns = compute_returns(prices)

  # Exclude benchmark from signal calculation
  stock_returns = returns.drop(columns=['SPY'], errors='ignore')

  mom = momentum_signal(stock_returns)
  vol = rolling_volatility(stock_returns)
  ranked = rank_signal(mom)

  print("Momentum signal (last 5 rows):")
  print(mom.tail())
  print("\nRanked signal (last 5 rows, higher = stronger momentum):")
  print(ranked.tail())
