# rotate_backtest_v4.py
# 股债轮动 v4：黄金防守 + 回撤止损（drawdown control），基准=沪深300
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
    prices[c] = prices[c].ffill()
    prices[c] = prices[c].fillna(prices[c].iloc[:int(len(prices) * 0.7)].mean())  # 训练集均值中性化

n = len(prices)
hs300 = prices['510300.SH'].values
idx = {c: ALL_CODES.index(c) for c in ALL_CODES}
eq_idx = [idx[c] for c in EQUITY]

print("=" * 72)
print(f"股债轮动 v4（回撤止损）| 基准=沪深300 | 交易日 {n}")
print("=" * 72)
print_report(calc_return_metrics(bench_ret), "基准：沪深300 买入持有")


def run_strategy(trend_window=30, momentum_window=20, rebalance=20, top_k=3,
                 sent_thresh=None, use_sentiment=False, vol_filter=0.35,
                 vol_target=0.0, stop_loss=0.10, defense='gold',
                 fee_rate=0.0003, slippage=0.001, trend_buffer=0.0):
    """择时 + 回撤止损 + 波动率目标仓位管理。
    vol_target>0 时：股票持仓按 min(目标波动率/实际波动率,1) 缩仓，余仓切货币ETF。"""
    mom_np = close_df.pct_change(momentum_window).values
    rets_np = daily_rets.values
    hs300_ma = pd.Series(hs300).rolling(trend_window).mean().values
    hs300_vol = pd.Series(hs300_ret).rolling(20).std().values * np.sqrt(252)
    sent_arr = prices['daily_sentiment'].values
    cash_ret = daily_rets[CASH].values  # 货币ETF日收益（波动率目标缩仓时切向现金）

    defense_map = {
        'cash': [idx[CASH]],
        'bond': [idx[BOND]],
        'gold': [idx[GOLD]],
        'defense_mom': [idx[BOND], idx[CASH], idx[GOLD]],
    }
    pool = defense_map[defense]

    strat_ret = np.zeros(n)
    current_hold = []
    nav = 1.0
    peak = 1.0
    trade_cost = fee_rate * 2 + slippage * 2  # 双边换仓成本（佣金+滑点）
    in_market = False  # 当前是否持有股票（多头），用于趋势滞回判断

    for t in range(n):
        old_hold = list(current_hold)  # 调仓前持仓，用于判断换仓（扣费）
        # 波动率目标：股票仓位权重（波动率高时缩仓，余仓切货币）
        stock_weight = 1.0
        if vol_target > 0 and not np.isnan(hs300_vol[t]) and hs300_vol[t] > 0:
            stock_weight = min(vol_target / hs300_vol[t], 1.0)

        if current_hold:
            vals = rets_np[t, current_hold]
            vals = vals[~np.isnan(vals)]
            raw = vals.mean() if len(vals) else 0.0
            if current_hold[0] in eq_idx and stock_weight < 1.0:
                cr = cash_ret[t] if not np.isnan(cash_ret[t]) else 0.0
                strat_ret[t] = stock_weight * raw + (1 - stock_weight) * cr
            else:
                strat_ret[t] = raw
        nav *= (1.0 + strat_ret[t])
        peak = max(peak, nav)
        dd = nav / peak - 1.0

        # 趋势判断（滞回：进场/离场用不同阈值，避免均线附近频繁切换吃手续费）
        ma_t = hs300_ma[t]
        if not np.isnan(ma_t):
            if in_market:
                trend_up = hs300[t] > ma_t * (1 - trend_buffer)   # 持有中：跌破缓冲才离场
            else:
                trend_up = hs300[t] > ma_t * (1 + trend_buffer)   # 空仓中：站上缓冲才进场
        else:
            trend_up = False
        sent_ok = (not use_sentiment) or (sent_arr[t] > sent_thresh)
        vol_ok = True
        if vol_filter > 0 and not np.isnan(hs300_vol[t]):
            vol_ok = hs300_vol[t] <= vol_filter

        if dd <= -stop_loss:
            current_hold = [idx[CASH]]
        elif trend_up and sent_ok and vol_ok:
            if t % rebalance == 0 or not current_hold or current_hold[0] not in eq_idx:
                mom_t = mom_np[t]
                vi = np.array(eq_idx)
                vi = vi[~np.isnan(mom_t[vi])]
                if len(vi) == 0:
                    current_hold = [idx[CASH]]
                else:
                    current_hold = vi[np.argsort(-mom_t[vi])[:top_k]].tolist()
        else:
            if defense == 'defense_mom':
                if t % rebalance == 0 or not current_hold or current_hold[0] in eq_idx:
                    mom_t = mom_np[t]
                    pi = np.array(pool)
                    pi = pi[~np.isnan(mom_t[pi])]
                    current_hold = [pi[np.argmax(mom_t[pi])]] if len(pi) else [idx[CASH]]
            else:
                current_hold = pool

        # 更新多头状态（用于下一日趋势滞回判断）
        in_market = bool(current_hold) and current_hold[0] in eq_idx

        # 换仓成本：持仓变化时扣双边手续费+滑点
        if old_hold and current_hold != old_hold:
            strat_ret[t] -= trade_cost

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


# ========== 聚焦扫描（defense_mom 防守 + 精细止损） ==========
results = []
for tw in [30, 60]:
    for mw in [30]:
        for rb in [20, 40]:
            for k in [3]:
                for defense in ['defense_mom', 'gold']:
                    for use_s, th in [(True, 0.20), (True, 0.25)]:
                        for vf in [0.40, 0.45]:
                            for vt in [0.0, 0.15, 0.20]:
                                for sl in [0.12, 0.14, 0.15]:
                                    for tb in [0.02, 0.03, 0.05]:
                                        sr = run_strategy(tw, mw, rb, k, th, use_s, vf, vt, sl,
                                                          defense, 0.0003, 0.0005, tb)
                                        r = evaluate(sr, f"tw{tw}_mw{mw}_rb{rb}_k{k}_{defense}_s{th}_v{vf}_vt{vt}_sl{sl}_tb{tb}")
                                        if r:
                                            results.append(r)

results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)

full = [r for r in results
        if r['sharpe_ratio'] > 1.0 and r['max_drawdown_pct'] >= -15.0
        and r['annual_excess_pct'] >= 5.0 and r['information_ratio'] >= 0.6
        and r['win_rate_pct'] >= 52.0 and r['profit_loss_ratio'] >= 1.3]
full.sort(key=lambda r: r['sharpe_ratio'], reverse=True)

print("\n" + "=" * 72)
print(f"★ 同时满足全部硬性指标的组合：{len(full)} 个")
print("=" * 72)
for r in full[:15]:
    print(f"  {r['name']:<40} 夏普{r['sharpe_ratio']:.3f} 年化{r['annual_return_pct']:.2f}% "
          f"回撤{r['max_drawdown_pct']:.2f}% 超额{r['annual_excess_pct']:.2f}% IR{r['information_ratio']:.3f} "
          f"胜率{r['win_rate_pct']:.1f}% 盈亏比{r['profit_loss_ratio']:.2f}")

print("\n" + "=" * 72)
print("全部组合 Top20（按夏普）")
print("=" * 72)
print(f"  {'策略':<38}{'夏普':>7}{'年化%':>8}{'回撤%':>8}{'超额%':>8}{'IR':>6}{'胜率%':>7}{'盈亏比':>6}")
for r in results[:20]:
    print(f"  {r['name']:<38}{r['sharpe_ratio']:>7.3f}{r['annual_return_pct']:>8.2f}"
          f"{r['max_drawdown_pct']:>8.2f}{r['annual_excess_pct']:>8.2f}{r['information_ratio']:>6.3f}"
          f"{r['win_rate_pct']:>7.2f}{r['profit_loss_ratio']:>6.2f}")

if full:
    print_report(full[0], f"★ 全达标最优 ({full[0]['name']})")
elif results:
    print_report(results[0], f"★ 夏普最优（未全达标）({results[0]['name']})")
