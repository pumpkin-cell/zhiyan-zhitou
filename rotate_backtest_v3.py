# rotate_backtest_v3.py
# 股债轮动 v3：精细防守方式（货币/债券/黄金/防御池轮动）+ 趋势窗口，基准=沪深300
import pandas as pd
import numpy as np
from metrics import calc_return_metrics, calc_trade_metrics, print_report

prices = pd.read_csv("./data/raw/etf_prices/all_close.csv", parse_dates=['date'])
prices = prices.sort_values('date').reset_index(drop=True)
sent = pd.read_csv("./data/raw/sentiment_indices.csv", parse_dates=['date'])
prices = prices.merge(sent, on='date', how='left')

EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
BOND, CASH, GOLD = '511260.SH', '511880.SH', '518880.SH'
ALL_CODES = EQUITY + [BOND, CASH, GOLD]

close_df = prices[ALL_CODES]
daily_rets = close_df.pct_change()
hs300_ret = prices['510300.SH'].pct_change().fillna(0).values
bench_ret = hs300_ret

for c in ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']:
    prices[c] = prices[c].ffill().fillna(0)

n = len(prices)
hs300 = prices['510300.SH'].values
idx = {c: ALL_CODES.index(c) for c in ALL_CODES}
eq_idx = [idx[c] for c in EQUITY]

print("=" * 72)
print(f"股债轮动 v3 | 基准=沪深300 | 交易日 {n}")
print("=" * 72)
print_report(calc_return_metrics(bench_ret), "基准：沪深300 买入持有")


def run_strategy(trend_window=40, momentum_window=40, rebalance=10, top_k=3,
                 sent_thresh=None, use_sentiment=True, defense='bond', vol_filter=0.0):
    """
    趋势（沪深300 vs MA）+ 情绪 择时：
      向上+情绪好 -> 股票池动量 top_k
      否则 -> 防守（defense: cash/bond/gold/defense_mom）
    vol_filter: 沪深300 近期年化波动超过该值(如0.35)则强制防守。
    """
    mom_np = close_df.pct_change(momentum_window).values
    rets_np = daily_rets.values
    hs300_ma = pd.Series(hs300).rolling(trend_window).mean().values
    hs300_vol = pd.Series(hs300_ret).rolling(20).std().values * np.sqrt(252)
    sent_arr = prices['daily_sentiment'].values

    defense_map = {
        'cash': [idx[CASH]],
        'bond': [idx[BOND]],
        'gold': [idx[GOLD]],
        'defense_mom': [idx[BOND], idx[CASH], idx[GOLD]],
    }

    strat_ret = np.zeros(n)
    current_hold = []

    for t in range(n):
        # t 日收益（t-1 日决定的持仓）
        if current_hold:
            vals = rets_np[t, current_hold]
            vals = vals[~np.isnan(vals)]
            strat_ret[t] = vals.mean() if len(vals) else 0.0

        # t 日收盘生成信号
        trend_up = hs300[t] > hs300_ma[t] if not np.isnan(hs300_ma[t]) else False
        sent_ok = (not use_sentiment) or (sent_arr[t] > sent_thresh)
        vol_ok = True
        if vol_filter > 0 and not np.isnan(hs300_vol[t]):
            vol_ok = hs300_vol[t] <= vol_filter

        if trend_up and sent_ok and vol_ok:
            if t % rebalance == 0 or not current_hold or current_hold[0] not in eq_idx:
                mom_t = mom_np[t]
                vi = np.array(eq_idx)
                vi = vi[~np.isnan(mom_t[vi])]
                if len(vi) == 0:
                    current_hold = [idx[CASH]]
                else:
                    current_hold = vi[np.argsort(-mom_t[vi])[:top_k]].tolist()
        else:
            pool = defense_map[defense]
            if defense == 'defense_mom':
                if t % rebalance == 0 or not current_hold or current_hold[0] in eq_idx:
                    mom_t = mom_np[t]
                    pi = np.array(pool)
                    pi = pi[~np.isnan(mom_t[pi])]
                    current_hold = [pi[np.argmax(mom_t[pi])]] if len(pi) else [idx[CASH]]
            else:
                current_hold = pool

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
for tw in [20, 30, 40, 60]:
    for mw in [20, 40, 60]:
        for rb in [10, 20]:
            for k in [2, 3]:
                for defense in ['cash', 'bond', 'gold', 'defense_mom']:
                    for use_s, th in [(False, 0.0), (True, 0.15), (True, 0.20), (True, 0.25)]:
                        for vf in [0.0, 0.35, 0.30]:
                            sr = run_strategy(tw, mw, rb, k, th, use_s, defense, vf)
                            r = evaluate(sr, f"tw{tw}_mw{mw}_rb{rb}_k{k}_{defense}_s{th if use_s else '-'}_v{vf}")
                            if r:
                                results.append(r)

results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
print("\n" + "=" * 72)
print("股债轮动 v3 Top25（按夏普排序，无未来函数，基准=沪深300）")
print("=" * 72)
print(f"  {'策略':<32}{'夏普':>7}{'年化%':>8}{'回撤%':>8}{'超额%':>8}{'IR':>6}{'胜率%':>7}{'盈亏比':>6}")
for r in results[:25]:
    print(f"  {r['name']:<32}{r['sharpe_ratio']:>7.3f}{r['annual_return_pct']:>8.2f}"
          f"{r['max_drawdown_pct']:>8.2f}{r['annual_excess_pct']:>8.2f}{r['information_ratio']:>6.3f}"
          f"{r['win_rate_pct']:>7.2f}{r['profit_loss_ratio']:>6.2f}")

# 同时按"回撤<=15%"过滤，找达标组合
print("\n" + "=" * 72)
print("满足「回撤≤15%」的最优组合（按夏普排序）")
print("=" * 72)
ok = [r for r in results if r['max_drawdown_pct'] >= -15.0]
ok.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
for r in ok[:10]:
    print(f"  {r['name']:<32} 夏普{r['sharpe_ratio']:.3f} 年化{r['annual_return_pct']:.2f}% "
          f"回撤{r['max_drawdown_pct']:.2f}% 超额{r['annual_excess_pct']:.2f}% IR{r['information_ratio']:.3f}")

if results:
    print_report(results[0], f"★ 最优 (夏普) ({results[0]['name']})")
