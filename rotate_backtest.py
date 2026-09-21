# rotate_backtest.py
# 多标的相对强弱动量轮动 + 情绪因子择时（无未来函数：t 日信号 -> t+1 日执行）
import pandas as pd
import numpy as np
from metrics import calc_return_metrics, calc_trade_metrics, print_report

# ========== 加载数据 ==========
prices = pd.read_csv("./data/raw/etf_prices/all_close.csv", parse_dates=['date'])
prices = prices.sort_values('date').reset_index(drop=True)

sent = pd.read_csv("./data/raw/sentiment_indices.csv", parse_dates=['date'])
prices = prices.merge(sent, on='date', how='left')

codes = [c for c in prices.columns if c not in
         ('date', 'daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment')]

close_df = prices[codes]
daily_rets = close_df.pct_change()          # 每个标的日收益率（含上市前 NaN）
bench_ret = daily_rets.mean(axis=1).fillna(0).values   # 等权买入持有基准

for c in ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']:
    prices[c] = prices[c].ffill().fillna(0)

n = len(prices)
print("=" * 70)
print(f"多标的轮动回测框架  |  标的数: {len(codes)}  |  交易日: {n}")
print(f"标的: {codes}")
print("=" * 70)


# ========== 轮动策略 ==========
def run_rotation(momentum_window=60, rebalance=20, top_k=2,
                 sent_col='daily_sentiment', sent_thresh=None,
                 use_sentiment=True, trend_filter=True):
    """
    相对强弱动量轮动 + 情绪开关。

    每日：
      1) 若启用情绪开关且当日情绪 <= 阈值，则空仓（防守）
      2) 调仓日（每 rebalance 天）选择动量最强 top_k 个标的（趋势过滤下只选正动量）
      3) 组合日收益 = 持有标的等权平均收益
    """
    mom_np = close_df.pct_change(momentum_window).values          # n x 9，t 日动量
    rets_np = daily_rets.values                                   # n x 9（上市前 NaN）
    sent_arr = prices[sent_col].values

    strat_ret = np.zeros(n)
    current_hold = []

    for t in range(n):
        # 1) t 日收益 = 基于 t-1 日收盘信号的持仓（无未来函数）
        if current_hold:
            vals = rets_np[t, current_hold]
            vals = vals[~np.isnan(vals)]
            strat_ret[t] = vals.mean() if len(vals) else 0.0

        # 2) t 日收盘后生成信号，决定 t+1 日持仓
        if use_sentiment and sent_arr[t] <= sent_thresh:
            current_hold = []
        elif t % rebalance == 0 or len(current_hold) == 0:
            mom_t = mom_np[t]
            valid_idx = np.where(~np.isnan(mom_t))[0]
            if trend_filter:
                valid_idx = valid_idx[mom_t[valid_idx] > 0]
            if len(valid_idx) == 0:
                current_hold = []
            else:
                top = valid_idx[np.argsort(-mom_t[valid_idx])[:top_k]]
                current_hold = top.tolist()

    return strat_ret


def evaluate(strat_ret, name, bench=bench_ret):
    m = calc_return_metrics(strat_ret, benchmark_returns=bench)
    if m is None:
        return None
    # 交易级指标：连续持仓段
    pos = (strat_ret != 0).astype(int)
    trades = []
    in_pos = False
    cum = 1.0
    for t in range(n):
        r = strat_ret[t]
        if pos[t] == 1:
            if not in_pos:
                in_pos = True
                cum = 1.0 + r
            else:
                cum *= (1.0 + r)
        else:
            if in_pos:
                in_pos = False
                trades.append(cum - 1.0)
    if in_pos:
        trades.append(cum - 1.0)
    t = calc_trade_metrics(np.array(trades))
    m.update(t)
    m['name'] = name
    return m


# ========== 基准 ==========
print_report(calc_return_metrics(bench_ret), "基准：等权买入持有全部标的")


# ========== 参数扫描 ==========
print("\n" + "=" * 70)
print("轮动策略参数扫描（无未来函数）")
print("=" * 70)

results = []
sent_levels = {
    'daily_sentiment': [None, 0.10, 0.15, 0.20, 0.25, 0.30],
    'ewma_7d': [None, 0.10, 0.15, 0.20, 0.25],
}
for mw in [20, 40, 60, 120]:
    for rb in [10, 20, 40]:
        for k in [1, 2, 3]:
            for tf in [True, False]:
                for sc, ths in sent_levels.items():
                    for th in ths:
                        use_s = th is not None
                        sr = run_rotation(mw, rb, k, sc, th if use_s else 0.0, use_s, tf)
                        r = evaluate(sr, f"mw{mw}_rb{rb}_k{k}_tf{tf}_{sc if use_s else 'nosent'}_{th if use_s else ''}")
                        if r:
                            results.append(r)

results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)

print(f"\n  {'策略':<34}{'夏普':>8}{'年化%':>8}{'回撤%':>8}{'超额%':>8}{'IR':>6}{'交易':>5}{'胜率%':>7}{'盈亏比':>6}")
for r in results[:25]:
    print(f"  {r['name']:<34}{r['sharpe_ratio']:>8.3f}{r['annual_return_pct']:>8.2f}"
          f"{r['max_drawdown_pct']:>8.2f}{r['annual_excess_pct']:>8.2f}{r['information_ratio']:>6.3f}"
          f"{r['num_trades']:>5}{r['win_rate_pct']:>7.2f}{r['profit_loss_ratio']:>6.2f}")

# 同时给出最优的完整报告
if results:
    best = results[0]
    print_report(best, f"★ 最优轮动策略 ({best['name']})")
