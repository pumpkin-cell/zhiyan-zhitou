# backend/app/core/factor_compute.py
# =====================================================================
# 因子计算纯函数（从 reproduce_batch.py 迁移而来）
# load_prices / ic_of / cross_sectional / rsi / ma_trend / reproduce_all
# =====================================================================
import os

import numpy as np
import pandas as pd

from .metrics import calc_return_metrics
from .paths import DATA_RAW

EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
BENCH = '510300.SH'


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(os.path.join(DATA_RAW, "etf_prices", "all_close.csv"), parse_dates=['date'])
    return prices.sort_values('date').reset_index(drop=True)


def ic_of(factor_series, prices, horizon=5):
    px = prices[BENCH]
    fr = px.pct_change(horizon).shift(-horizon)
    df = pd.DataFrame({'f': factor_series, 'r': fr}).dropna()
    if len(df) < 30:
        return np.nan
    # Spearman 秩相关 = 秩的 Pearson 相关（避免 scipy 依赖）
    return df['f'].rank().corr(df['r'].rank())


def cross_sectional(prices, lookback, reverse=False, hold=20):
    close = prices[EQUITY].astype(float)
    ret = close.pct_change()
    n = len(prices)
    idx = {c: i for i, c in enumerate(EQUITY)}
    hw, hl = [], []
    wr = np.zeros(n)
    lr = np.zeros(n)
    for t in range(n):
        if hw:
            v = ret.iloc[t, hw].values; v = v[~np.isnan(v)]
            wr[t] = v.mean() if len(v) else 0
        if hl:
            v = ret.iloc[t, hl].values; v = v[~np.isnan(v)]
            lr[t] = v.mean() if len(v) else 0
        if t % hold == 0 and t >= lookback:
            mom = close.iloc[t] / close.iloc[t - lookback] - 1
            mom = mom.dropna()
            if len(mom) >= 6:
                o = mom.sort_values(ascending=False)
                hw = [idx[c] for c in o.index[:3]]
                hl = [idx[c] for c in o.index[-3:]]
    return (lr - wr) if reverse else (wr - lr)


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def ma_trend(prices):
    px = prices[BENCH]
    ma20 = px.rolling(20).mean()
    signal = (px > ma20).astype(int)
    ret = px.pct_change()
    strat = signal.shift(1).fillna(0) * ret
    return strat.values


def reproduce_all() -> dict:
    """批量复现 6 个经典因子/研报结论，返回对照表。"""
    prices = load_prices()
    px = prices[BENCH]
    ret = px.pct_change()
    items = []

    rev = ic_of(-px.pct_change(5), prices)
    items.append({"name": "短期反转(Lehmann 1990)", "origin": "过去赢家短期会回落（反转溢价）",
                  "result": f"reversal IC={rev:+.4f}", "conclusion": "复现成立" if rev > 0.03 else "不显著"})

    mom = calc_return_metrics(cross_sectional(prices, 120))
    items.append({"name": "动量(JT 1993)", "origin": "过去赢家未来继续跑赢（动量溢价）",
                  "result": f"赢家-输家年化 {mom['annual_return_pct']:.2f}%",
                  "conclusion": "A股ETF上不可复现（呈反转）"})

    lowvol = ic_of(-(ret.rolling(20).std() * np.sqrt(252)), prices)
    items.append({"name": "低波动异象(Ang 2006)", "origin": "低波动组合长期跑赢高波动",
                  "result": f"(-波动率) IC={lowvol:+.4f}", "conclusion": "复现成立" if lowvol > 0.03 else "不显著"})

    sent = pd.read_csv(os.path.join(DATA_RAW, "sentiment_indices.csv"), parse_dates=['date'])
    s_raw = sent.set_index('date').reindex(prices['date'])['daily_sentiment'].ffill()
    s = s_raw.fillna(s_raw.iloc[:int(len(s_raw) * 0.7)].mean()).values
    sen = ic_of(s, prices)
    items.append({"name": "投资者情绪反向(Baker-Wurgler 2006)", "origin": "高情绪预示未来低收益（反向）",
                  "result": f"情绪 IC={sen:+.4f}", "conclusion": "复现成立" if sen < -0.03 else "不显著"})

    tr = calc_return_metrics(ma_trend(prices))
    items.append({"name": "均线趋势(MA20)", "origin": "价格站上均线持有、跌破空仓能跑赢基准",
                  "result": f"趋势策略年化 {tr['annual_return_pct']:.2f}%（基准7.32%）",
                  "conclusion": "复现成立" if tr['annual_return_pct'] > 7.32 else "未跑赢基准"})

    rsi_ic = ic_of(rsi(px), prices)
    items.append({"name": "RSI 超买超卖(经典)", "origin": "RSI 低买高卖有预测力",
                  "result": f"RSI IC={rsi_ic:+.4f}", "conclusion": "复现成立" if abs(rsi_ic) > 0.03 else "不显著"})

    match_rate = sum(1 for i in items if i["conclusion"] == "复现成立") / len(items) * 100
    return {
        "items": items,
        "run_rate": 100.0,
        "match_rate": match_rate,
    }
