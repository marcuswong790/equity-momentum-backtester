# backtest.py
# Portfolio construction and strategy simulation

import pandas as pd
import numpy as np
from data import download_prices, compute_returns
from signals import momentum_signal, rolling_volatility, rank_signal


def build_portfolio(ranked_signal, returns, top_n=5, vol_df=None):
    """
    Long-only momentum portfolio.
    
    Each day, we:
    1. Rank all stocks by momentum signal
    2. Buy equal weights in the top N stocks
    3. Hold until the next rebalance
    
    We rebalance monthly (every 21 trading days) — daily rebalancing
    would incur too much in transaction costs in the real world.
    
    Parameters:
        ranked_signal: DataFrame of percentile-ranked momentum scores
        returns: DataFrame of daily log returns
        top_n: number of stocks to hold (default 5)
        vol_df: optional volatility DataFrame for vol-targeted weighting
    
    Returns:
        portfolio_returns: Series of daily portfolio returns
        weights_history: DataFrame tracking portfolio weights over time
    """
    
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    weights_history = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    
    rebalance_dates = returns.index[::21]  # Rebalance every 21 trading days
    current_weights = {}
    
    for date in returns.index:
        # Rebalance on rebalance dates
        if date in rebalance_dates:
            signal_today = ranked_signal.loc[date].dropna()
            
            if len(signal_today) >= top_n:
                # Select top N stocks by momentum
                top_stocks = signal_today.nlargest(top_n).index.tolist()
                
                if vol_df is not None:
                    # Volatility-targeted weighting: weight = 1/vol, then normalise
                    # This gives lower-volatility stocks slightly more weight,
                    # improving risk-adjusted returns. Called "inverse vol weighting".
                    vols = vol_df.loc[date, top_stocks].replace(0, np.nan).dropna()
                    if len(vols) > 0:
                        inv_vol = 1 / vols
                        weights = (inv_vol / inv_vol.sum()).to_dict()
                    else:
                        weights = {s: 1/top_n for s in top_stocks}
                else:
                    # Equal weighting — simpler, often surprisingly competitive
                    weights = {s: 1/top_n for s in top_stocks}
                
                current_weights = weights
        
        # Calculate today's portfolio return
        if current_weights:
            daily_ret = 0
            for stock, weight in current_weights.items():
                if stock in returns.columns and date in returns.index:
                    daily_ret += weight * returns.loc[date, stock]
            portfolio_returns[date] = daily_ret
        
        # Record weights
        for stock in returns.columns:
            weights_history.loc[date, stock] = current_weights.get(stock, 0)
    
    return portfolio_returns.dropna(), weights_history


def compute_performance_stats(portfolio_returns, benchmark_returns, risk_free_rate=0.04):
    """
    Compute standard performance statistics used in quant finance.
    
    These are the metrics any quant interviewer will ask about.
    Know what each one means and how to interpret it.
    
    Parameters:
        portfolio_returns: Series of daily portfolio log returns
        benchmark_returns: Series of daily benchmark (SPY) log returns
        risk_free_rate: annual risk-free rate (default 4% — approx US T-bill rate)
    
    Returns:
        dict of performance statistics
    """
    
    daily_rf = risk_free_rate / 252  # Convert annual to daily
    excess_returns = portfolio_returns - daily_rf
    
    # Annualised return: mean daily return * 252 trading days
    ann_return = portfolio_returns.mean() * 252
    
    # Annualised volatility: std of daily returns * sqrt(252)
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio: risk-adjusted return
    # Sharpe = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
    # Higher is better. Above 1.0 is good. Above 2.0 is exceptional.
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    
    # Maximum Drawdown: largest peak-to-trough decline
    # This measures the worst-case loss an investor could have experienced
    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio: annualised return / max drawdown
    # Penalises strategies with large drawdowns
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Beta: correlation of strategy to benchmark
    # Beta = 1 means strategy moves 1-for-1 with market
    # Beta < 1 means lower market exposure (generally desirable)
    aligned = pd.DataFrame({'port': portfolio_returns, 'bench': benchmark_returns}).dropna()
    if len(aligned) > 1:
        cov_matrix = aligned.cov()
        beta = cov_matrix.loc['port', 'bench'] / cov_matrix.loc['bench', 'bench']
        
        # Alpha: return not explained by market exposure
        # Positive alpha means the strategy genuinely adds value beyond market beta
        alpha = ann_return - (risk_free_rate + beta * (benchmark_returns.mean() * 252 - risk_free_rate))
    else:
        beta = np.nan
        alpha = np.nan
    
    # Win rate: % of days with positive return
    win_rate = (portfolio_returns > 0).mean()
    
    return {
        'Annualised Return':  f"{ann_return:.2%}",
        'Annualised Volatility': f"{ann_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Beta to SPY': f"{beta:.2f}",
        'Alpha (annualised)': f"{alpha:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Total Trading Days': len(portfolio_returns)
    }


if __name__ == '__main__':
    print("=== Running Backtest ===\n")
    
    # Load data
    prices = download_prices()
    returns = compute_returns(prices)
    
    # Separate stocks from benchmark
    benchmark_returns = returns['SPY']
    stock_returns = returns.drop(columns=['SPY'], errors='ignore')
    
    # Build signals
    print("Computing signals...")
    mom = momentum_signal(stock_returns)
    vol = rolling_volatility(stock_returns)
    ranked = rank_signal(mom)
    
    # Run backtest — equal weight
    print("Running backtest...")
    port_returns, weights = build_portfolio(ranked, stock_returns, top_n=5)
    
    # Align benchmark
    bench_aligned = benchmark_returns.reindex(port_returns.index).dropna()
    port_aligned = port_returns.reindex(bench_aligned.index).dropna()
    
    # Print stats
    print("\n=== STRATEGY PERFORMANCE ===")
    stats = compute_performance_stats(port_aligned, bench_aligned)
    for k, v in stats.items():
        print(f"  {k:<30} {v}")
    
    print("\n=== BENCHMARK (SPY) PERFORMANCE ===")
    bench_stats = compute_performance_stats(bench_aligned, bench_aligned)
    for k, v in bench_stats.items():
        print(f"  {k:<30} {v}")
