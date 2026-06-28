# visualise.py
# Generates all charts for the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from data import download_prices, compute_returns
from signals import momentum_signal, rolling_volatility, rank_signal
from backtest import build_portfolio

# Set clean visual style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
COLORS = {'strategy': '#1B3A6B', 'benchmark': '#E87C3E', 'drawdown': '#C0392B'}


def plot_cumulative_returns(port_returns, bench_returns, save=True):
    """Chart 1: Cumulative returns — strategy vs benchmark"""
    
    cumulative_port = (1 + port_returns).cumprod()
    cumulative_bench = (1 + bench_returns).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(cumulative_port.index, cumulative_port.values,
            label='Momentum Strategy', color=COLORS['strategy'], linewidth=2)
    ax.plot(cumulative_bench.index, cumulative_bench.values,
            label='S&P 500 (SPY)', color=COLORS['benchmark'], linewidth=2, linestyle='--')
    
    ax.set_title('Cumulative Returns: Momentum Strategy vs S&P 500', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (starting at $1)')
    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    
    if save:
        plt.savefig('chart_1_cumulative_returns.png', dpi=150, bbox_inches='tight')
        print("Saved: chart_1_cumulative_returns.png")
    plt.show()


def plot_drawdown(port_returns, bench_returns, save=True):
    """Chart 2: Drawdown over time — shows risk profile"""
    
    def get_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        return (cumulative - rolling_max) / rolling_max
    
    dd_port = get_drawdown(port_returns)
    dd_bench = get_drawdown(bench_returns)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.fill_between(dd_port.index, dd_port.values, 0,
                    alpha=0.4, color=COLORS['strategy'], label='Strategy Drawdown')
    ax.fill_between(dd_bench.index, dd_bench.values, 0,
                    alpha=0.3, color=COLORS['benchmark'], label='SPY Drawdown')
    ax.plot(dd_port.index, dd_port.values, color=COLORS['strategy'], linewidth=1)
    
    ax.set_title('Drawdown Profile: Strategy vs Benchmark', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown from Peak')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        plt.savefig('chart_2_drawdown.png', dpi=150, bbox_inches='tight')
        print("Saved: chart_2_drawdown.png")
    plt.show()


def plot_rolling_sharpe(port_returns, save=True):
    """Chart 3: Rolling 252-day Sharpe ratio — shows strategy consistency"""
    
    rolling_mean = port_returns.rolling(252).mean() * 252
    rolling_std = port_returns.rolling(252).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=COLORS['strategy'], linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axhline(y=1, color='green', linewidth=0.8, linestyle='--', alpha=0.7, label='Sharpe = 1.0')
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe > 0), alpha=0.2, color='green')
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe < 0), alpha=0.2, color='red')
    
    ax.set_title('Rolling 12-Month Sharpe Ratio', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        plt.savefig('chart_3_rolling_sharpe.png', dpi=150, bbox_inches='tight')
        print("Saved: chart_3_rolling_sharpe.png")
    plt.show()


def plot_monthly_returns_heatmap(port_returns, save=True):
    """Chart 4: Monthly returns heatmap — the classic quant tearsheet visual"""
    
    monthly = port_returns.resample('ME').sum()
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values
    })
    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                center=0, linewidths=0.5, ax=ax,
                annot_kws={'size': 9})
    
    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    
    if save:
        plt.savefig('chart_4_monthly_heatmap.png', dpi=150, bbox_inches='tight')
        print("Saved: chart_4_monthly_heatmap.png")
    plt.show()


if __name__ == '__main__':
    print("=== Generating Charts ===\n")
    
    prices = download_prices()
    returns = compute_returns(prices)
    benchmark_returns = returns['SPY']
    stock_returns = returns.drop(columns=['SPY'], errors='ignore')
    
    mom = momentum_signal(stock_returns)
    vol = rolling_volatility(stock_returns)
    ranked = rank_signal(mom)
    
    port_returns, _ = build_portfolio(ranked, stock_returns, top_n=5, vol_df=vol)
    bench_aligned = benchmark_returns.reindex(port_returns.index).dropna()
    port_aligned = port_returns.reindex(bench_aligned.index).dropna()
    
    plot_cumulative_returns(port_aligned, bench_aligned)
    plot_drawdown(port_aligned, bench_aligned)
    plot_rolling_sharpe(port_aligned)
    plot_monthly_returns_heatmap(port_aligned)
    
    print("\nAll charts saved. Add these PNG files to your GitHub repo.")
