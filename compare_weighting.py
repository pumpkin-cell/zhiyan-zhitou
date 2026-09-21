# compare_weighting.py
# =====================================================================
# 对比 top3 持仓的三种权重方案（等权 / 动量加权 / 波动率倒数加权），含手续费
# 在 rotate_backtest_v4 最优参数 + 手续费基础上，回答"按比例加权是否更好"
# =====================================================================
import pandas as pd
import numpy as np
from metrics import calc_return_metrics

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
    prices[c] = prices[c].fillna(prices[c].iloc[:int(len(prices) * 0.7)].mean())

n = len(prices)
hs300 = prices['510300.SH'].values
idx = {c: ALL_CODES.index(c) for c in ALL_CODES}
eq_idx = [idx[c] for c in EQUITY]

FEE, SLIP = 0.0003, 0.001
COST = FEE * 2 + SLIP * 2  # 双边换仓成本


def run(weighting='equal', trend_window=30, momentum_window=30, rebalance=20,
        top_k=3, sent_thresh=0.2, vol_filter=0.45, stop_loss=0.15):
    mom_np = close_df.pct_change(momentum_window).values
    rets_np = daily_rets.values
    hs300_ma = pd.Series(hs300).rolling(trend_window).mean().values
    hs300_vol = pd.Series(hs300_ret).rolling(20).std().values * np.sqrt(252)
    sent_arr = prices['daily_sentiment'].values
    pool = [idx[BOND], idx[CASH], idx[GOLD]]  # defense_mom

    strat_ret = np.zeros(n)
    current_hold = []
    current_w = np.array([])
    nav = peak = 1.0

    for t in range(n):
        old_hold = list(current_hold)
        # 今日收益：加权
        if len(current_hold):
            vals = rets_np[t, current_hold]
            w = current_w.copy()
            mask = ~np.isnan(vals)
            if mask.sum() > 0:
                vals = vals[mask]
                w = w[mask]
                w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
                strat_ret[t] = float(np.dot(vals, w))
        nav *= (1.0 + strat_ret[t])
        peak = max(peak, nav)
        dd = nav / peak - 1.0

        trend_up = hs300[t] > hs300_ma[t] if not np.isnan(hs300_ma[t]) else False
        sent_ok = sent_arr[t] > sent_thresh
        vol_ok = True
        if not np.isnan(hs300_vol[t]):
            vol_ok = hs300_vol[t] <= vol_filter

        if dd <= -stop_loss:
            current_hold, current_w = [idx[CASH]], np.array([1.0])
        elif trend_up and sent_ok and vol_ok:
            if t % rebalance == 0 or not current_hold or current_hold[0] not in eq_idx:
                mom_t = mom_np[t]
                vi = np.array(eq_idx)
                vi = vi[~np.isnan(mom_t[vi])]
                if len(vi) == 0:
                    current_hold, current_w = [idx[CASH]], np.array([1.0])
                else:
                    sel = vi[np.argsort(-mom_t[vi])[:top_k]]
                    current_hold = sel.tolist()
                    current_w = _weights(weighting, mom_t, sel, rets_np)
        else:
            # 防守动量：选债/现金/黄金中动量最强的一个（与 v4 一致，rebalance 才重选）
            if t % rebalance == 0 or not current_hold or current_hold[0] in eq_idx:
                mom_t = mom_np[t]
                pi = np.array(pool)
                pi = pi[~np.isnan(mom_t[pi])]
                current_hold = [pi[np.argmax(mom_t[pi])]] if len(pi) else [idx[CASH]]
                current_w = np.array([1.0])

        if old_hold and current_hold != old_hold:
            strat_ret[t] -= COST

    return strat_ret


def _weights(weighting, mom_t, sel, rets_np):
    k = len(sel)
    if weighting == 'equal':
        return np.ones(k) / k
    if weighting == 'momentum':
        m = mom_t[sel].astype(float)
        m = m - m.min() + 0.01          # min-max 到正数
        return m / m.sum()
    if weighting == 'inv_vol':
        vol = np.nanstd(rets_np[-60:, sel], axis=0)  # 过去60日波动率
        vol = np.where(vol < 1e-6, 1e-6, vol)
        w = 1.0 / vol
        return w / w.sum()
    return np.ones(k) / k


if __name__ == '__main__':
    print("=" * 72)
    print("top3 权重方案对比（含手续费 万3佣金 + 0.1%滑点，双边）")
    print("=" * 72)
    for wname in ['equal', 'momentum', 'inv_vol']:
        sr = run(weighting=wname)
        m = calc_return_metrics(sr)
        print(f"\n  【{wname:9s}】 年化{m['annual_return_pct']:6.2f}%  夏普{m['sharpe_ratio']:6.3f}  "
              f"回撤{m['max_drawdown_pct']:7.2f}%  日胜率{m['daily_win_rate_pct']:5.1f}%")
