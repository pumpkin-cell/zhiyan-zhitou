# rotate_backtest_v2.py
# 股债轮动：趋势+情绪择时，牛市持强势行业ETF，熊市切债券/货币/黄金防守（无未来函数）
import pandas as pd
import numpy as np
from metrics import calc_return_metrics, calc_trade_metrics, print_report

# ========== 数据 ==========
prices = pd.read_csv("./data/raw/etf_prices/all_close.csv", parse_dates=['date'])
prices = prices.sort_values('date').reset_index(drop=True)
sent = pd.read_csv("./data/raw/sentiment_indices.csv", parse_dates=['date'])
prices = prices.merge(sent, on='date', how='left')

EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
DEFENSIVE = ['511260.SH', '511880.SH', '518880.SH']   # 十年国债、货币、黄金
ALL_CODES = EQUITY + DEFENSIVE

close_df = prices[ALL_CODES]
daily_rets = close_df.pct_change()
# 基准：沪深300（A股择时标准基准）
hs300_ret = prices['510300.SH'].pct_change().fillna(0).values
bench_ret = hs300_ret

for c in ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']:
    prices[c] = prices[c].ffill().fillna(0)

n = len(prices)
hs300 = prices['510300.SH'].values

# 索引映射
eq_idx = [ALL_CODES.index(c) for c in EQUITY]
def_idx = [ALL_CODES.index(c) for c in DEFENSIVE]
cash_idx = ALL_CODES.index('511880.SH')

print("=" * 72)
print(f"股债轮动框架 | 股票池 {len(EQUITY)} 个 | 防御池 {len(DEFENSIVE)} 个 | 交易日 {n}")
print(f"基准：沪深300（510300.SH）")
print("=" * 72)
print_report(calc_return_metrics(bench_ret), "基准：沪深300 买入持有")


def run_strategy(trend_window=60, momentum_window=20, rebalance=10, top_k=2,
                 sent_col='daily_sentiment', sent_thresh=None, use_sentiment=True,
                 defensive_mode='cash'):
    """
    t 日收盘后判定：市场趋势（沪深300 vs 均线）+ 情绪
      - 趋势向上 且（无情绪开关或情绪好） -> 持有股票池中动量最强 top_k
      - 否则 -> 防守：货币（cash）或防御池动量最强 1 个（defensive_momentum）
    t+1 日执行。
    """
    mom_np = close_df.pct_change(momentum_window).values
    rets_np = daily_rets.values
    hs300_ma = pd.Series(hs300).rolling(trend_window).mean().values
    sent_arr = prices[sent_col].values

    strat_ret = np.zeros(n)
    current_hold = []

    for t in range(n):
        # 1) t 日收益（基于 t-1 日收盘决定的持仓）
        if current_hold:
            vals = rets_np[t, current_hold]
            vals = vals[~np.isnan(vals)]
            strat_ret[t] = vals.mean() if len(vals) else 0.0

        # 2) t 日收盘后生成信号
        trend_up = hs300[t] > hs300_ma[t] if not np.isnan(hs300_ma[t]) else False
        sent_ok = (not use_sentiment) or (sent_arr[t] > sent_thresh)

        if trend_up and sent_ok:
            # 持有股票：调仓选动量最强 top_k
            if t % rebalance == 0 or not current_hold or current_hold[0] in def_idx or current_hold[0] == cash_idx:
                mom_t = mom_np[t]
                vi = np.array(eq_idx)
                vi = vi[~np.isnan(mom_t[vi])]
                if len(vi) == 0:
                    current_hold = [cash_idx]
                else:
                    top = vi[np.argsort(-mom_t[vi])[:top_k]]
                    current_hold = top.tolist()
        else:
            # 防守
            if defensive_mode == 'cash':
                current_hold = [cash_idx]
            else:
                if t % rebalance == 0 or not current_hold or current_hold[0] in eq_idx:
                    mom_t = mom_np[t]
                    vi = np.array(def_idx)
                    vi = vi[~np.isnan(mom_t[vi])]
                    if len(vi) == 0:
                        current_hold = [cash_idx]
                    else:
                        current_hold = [vi[np.argmax(mom_t[vi])]]

    return strat_ret


def evaluate(strat_ret, name):
    m = calc_return_metrics(strat_ret, benchmark_returns=bench_ret)
    if m is None:
        return None
    pos = (strat_ret != 0).astype(int)
    trades, in_pos, cum = [], False, 1.0
    for t in range(n):
        r = strat_ret[t]
        if pos[t] == 1:
            cum = cum * (1 + r) if in_pos else (1 + r)
            in_pos = True
        else:
            if in_pos:
                trades.append(cum - 1.0)
                in_pos = False
                cum = 1.0
    if in_pos:
        trades.append(cum - 1.0)
    m.update(calc_trade_metrics(np.array(trades)))
    m['name'] = name
    return m


# ========== 扫描 ==========
results = []
for tw in [20, 40, 60, 120]:
    for mw in [20, 40, 60]:
        for rb in [10, 20]:
            for k in [1, 2, 3]:
                for dm in ['cash', 'defensive_momentum']:
                    for use_s, th in [(False, 0.0), (True, 0.15), (True, 0.20), (True, 0.25)]:
                        sr = run_strategy(tw, mw, rb, k, 'daily_sentiment', th, use_s, dm)
                        r = evaluate(sr, f"tw{tw}_mw{mw}_rb{rb}_k{k}_{dm}_sent{th if use_s else 'none'}")
                        if r:
                            results.append(r)

results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
print("\n" + "=" * 72)
print("股债轮动 Top25（按夏普排序，无未来函数）")
print("=" * 72)
print(f"  {'策略':<30}{'夏普':>7}{'年化%':>8}{'回撤%':>8}{'超额%':>8}{'IR':>6}{'胜率%':>7}{'盈亏比':>6}")
for r in results[:25]:
    print(f"  {r['name']:<30}{r['sharpe_ratio']:>7.3f}{r['annual_return_pct']:>8.2f}"
          f"{r['max_drawdown_pct']:>8.2f}{r['annual_excess_pct']:>8.2f}{r['information_ratio']:>6.3f}"
          f"{r['win_rate_pct']:>7.2f}{r['profit_loss_ratio']:>6.2f}")

if results:
    print_report(results[0], f"★ 最优股债轮动 ({results[0]['name']})")
