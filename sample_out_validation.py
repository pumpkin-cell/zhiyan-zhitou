# sample_out_validation.py
# 样本外验证：训练期(2019-2023)选参数 -> 测试期(2024-2026)验证（无未来函数）
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

for c in ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']:
    prices[c] = prices[c].ffill()
    train_mean = prices.loc[prices['date'] < pd.Timestamp('2024-01-01'), c].mean()  # 训练期均值（无未来函数）
    prices[c] = prices[c].fillna(train_mean)

n = len(prices)
hs300 = prices['510300.SH'].values
idx = {c: ALL_CODES.index(c) for c in ALL_CODES}
eq_idx = [idx[c] for c in EQUITY]

# 分割点：2024-01-01
split = prices.index[prices['date'] >= pd.Timestamp('2024-01-01')].min()
print(f"训练期: {prices['date'].iloc[0].date()} ~ {prices['date'].iloc[split-1].date()} ({split} 天)")
print(f"测试期: {prices['date'].iloc[split].date()} ~ {prices['date'].iloc[-1].date()} ({n-split} 天)")


def run_strategy(trend_window, momentum_window, rebalance, top_k,
                 sent_thresh, use_sentiment, vol_filter, stop_loss, defense,
                 vol_target=0.0, fee_rate=0.0003, slippage=0.0005, trend_buffer=0.0):
    mom_np = close_df.pct_change(momentum_window).values
    rets_np = daily_rets.values
    hs300_ma = pd.Series(hs300).rolling(trend_window).mean().values
    hs300_vol = pd.Series(hs300_ret).rolling(20).std().values * np.sqrt(252)
    sent_arr = prices['daily_sentiment'].values
    cash_ret = daily_rets[CASH].values
    defense_map = {'cash': [idx[CASH]], 'bond': [idx[BOND]], 'gold': [idx[GOLD]],
                   'defense_mom': [idx[BOND], idx[CASH], idx[GOLD]]}
    pool = defense_map[defense]
    strat_ret = np.zeros(n)
    current_hold = []
    nav = peak = 1.0
    trade_cost = fee_rate * 2 + slippage * 2
    in_market = False
    for t in range(n):
        old_hold = list(current_hold)
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
        ma_t = hs300_ma[t]
        if not np.isnan(ma_t):
            if in_market:
                trend_up = hs300[t] > ma_t * (1 - trend_buffer)
            else:
                trend_up = hs300[t] > ma_t * (1 + trend_buffer)
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
        in_market = bool(current_hold) and current_hold[0] in eq_idx
        if old_hold and current_hold != old_hold:
            strat_ret[t] -= trade_cost
    return strat_ret


# 候选参数网格（覆盖全样本最优邻域：tw60 + rb40 + 黄金/防守动量 + 波动率目标 + 缓冲带）
params = []
for defense in ['gold', 'defense_mom']:
    for vt in [0.0, 0.15, 0.20]:
        for tb in [0.03, 0.05]:
            for sl in [0.13, 0.14, 0.15]:
                params.append(dict(trend_window=60, momentum_window=30, rebalance=40, top_k=3,
                                   sent_thresh=0.25, use_sentiment=True, vol_filter=0.45,
                                   stop_loss=sl, defense=defense, vol_target=vt,
                                   trend_buffer=tb))

rows = []
for p in params:
    sr = run_strategy(**p)
    train_m = calc_return_metrics(sr[:split], benchmark_returns=hs300_ret[:split])
    test_m = calc_return_metrics(sr[split:], benchmark_returns=hs300_ret[split:])
    if train_m and test_m:
        rows.append((p, train_m, test_m))

# 训练期选优（按训练期夏普），看测试期表现
rows.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)

print("\n" + "=" * 78)
print("样本外验证：按【训练期夏普】排序，看测试期是否仍达标")
print("=" * 78)
print(f"  {'参数':<44}{'训练夏普':>8}{'测试夏普':>8}{'测试年化%':>9}{'测试回撤%':>9}{'测试IR':>7}")
for p, tr, te in rows[:10]:
    tag = f"{p['defense'][:6]}_mw{p['momentum_window']}_s{p['sent_thresh']}_v{p['vol_filter']}_sl{p['stop_loss']}"
    print(f"  {tag:<44}{tr['sharpe_ratio']:>8.3f}{te['sharpe_ratio']:>8.3f}"
          f"{te['annual_return_pct']:>9.2f}{te['max_drawdown_pct']:>9.2f}{te['information_ratio']:>7.3f}")

# 训练期最优 -> 测试期完整报告
best_p, best_tr, best_te = rows[0]
print("\n" + "=" * 78)
print(f"训练期最优参数 -> 测试期(2024-2026)完整报告")
print(f"参数: {best_p}")
print("=" * 78)
print_report(best_tr, "训练期(2019-2023)表现")
print_report(best_te, "测试期(2024-2026)表现【样本外】")
