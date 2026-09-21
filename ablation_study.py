# ablation_study.py
# =====================================================================
# 消融实验：在「扣手续费+缓冲带」框架下，量化各风控组件的真实贡献
# 回答：回撤止损/缓冲带/波动率目标/黄金防守 各自值多少（夏普、回撤）
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


def run(tw=60, mw=30, rb=40, k=3, defense='gold', sent_thresh=0.25,
        vf=0.45, vt=0.15, sl=0.14, tb=0.05, fee=0.0003, slip=0.0005):
    mom_np = close_df.pct_change(mw).values
    rets_np = daily_rets.values
    hs300_ma = pd.Series(hs300).rolling(tw).mean().values
    hs300_vol = pd.Series(hs300_ret).rolling(20).std().values * np.sqrt(252)
    sent_arr = prices['daily_sentiment'].values
    cash_ret = daily_rets[CASH].values
    pool = {'cash': [idx[CASH]], 'bond': [idx[BOND]], 'gold': [idx[GOLD]],
            'defense_mom': [idx[BOND], idx[CASH], idx[GOLD]]}[defense]
    strat = np.zeros(n)
    cur = []
    nav = peak = 1.0
    cost = fee * 2 + slip * 2
    in_mkt = False
    for t in range(n):
        old = list(cur)
        sw = 1.0
        if vt > 0 and not np.isnan(hs300_vol[t]) and hs300_vol[t] > 0:
            sw = min(vt / hs300_vol[t], 1.0)
        if cur:
            v = rets_np[t, cur]
            v = v[~np.isnan(v)]
            raw = v.mean() if len(v) else 0.0
            if cur[0] in eq_idx and sw < 1.0:
                cr = cash_ret[t] if not np.isnan(cash_ret[t]) else 0.0
                strat[t] = sw * raw + (1 - sw) * cr
            else:
                strat[t] = raw
        nav *= (1 + strat[t])
        peak = max(peak, nav)
        dd = nav / peak - 1
        ma_t = hs300_ma[t]
        if not np.isnan(ma_t):
            trend_up = hs300[t] > ma_t * (1 - tb) if in_mkt else hs300[t] > ma_t * (1 + tb)
        else:
            trend_up = False
        sent_ok = sent_arr[t] > sent_thresh
        vol_ok = hs300_vol[t] <= vf if (vf > 0 and not np.isnan(hs300_vol[t])) else True
        if dd <= -sl:
            cur = [idx[CASH]]
        elif trend_up and sent_ok and vol_ok:
            if t % rb == 0 or not cur or cur[0] not in eq_idx:
                vi = np.array(eq_idx)
                vi = vi[~np.isnan(mom_np[t, vi])]
                cur = vi[np.argsort(-mom_np[t, vi])[:k]].tolist() if len(vi) else [idx[CASH]]
        else:
            if defense == 'defense_mom':
                if t % rb == 0 or not cur or cur[0] in eq_idx:
                    pi = np.array(pool)
                    pi = pi[~np.isnan(mom_np[t, pi])]
                    cur = [pi[np.argmax(mom_np[t, pi])]] if len(pi) else [idx[CASH]]
            else:
                cur = pool
        in_mkt = bool(cur) and cur[0] in eq_idx
        if old and cur != old:
            strat[t] -= cost
    return strat


def show(name, **kw):
    m = calc_return_metrics(run(**kw), benchmark_returns=hs300_ret)
    print(f"  {name:<34} 夏普{m['sharpe_ratio']:6.3f}  年化{m['annual_return_pct']:6.2f}%  "
          f"回撤{m['max_drawdown_pct']:7.2f}%  超额{m['annual_excess_pct']:6.2f}%")


if __name__ == '__main__':
    print("=" * 78)
    print("消融实验（扣手续费 万3佣金+0.05%滑点，tw60 mw30 rb40 k3）")
    print("=" * 78)
    show("① 全部组件（最优）")
    show("② 去缓冲带", tb=0.0)
    show("③ 去波动率目标", vt=0.0)
    show("④ 去回撤止损", sl=0.99)
    show("⑤ 防守改 defense_mom", defense='defense_mom')
    show("⑥ 防守改现金", defense='cash')
    show("⑦ 纯轮动（无任何风控）", vt=0.0, sl=0.99, tb=0.0, defense='cash', sent_thresh=-99)
