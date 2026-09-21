# factor_cards.py
# =====================================================================
# 命题1「面向金融量化投研工作全流程的智能体系统」模块1 + 模块2 原型
#   模块1 研报复现    —— 把研报里的因子用统一公式落成"因子卡片"
#   模块2 因子知识沉淀 —— 对每个因子算 IC / ICIR / 单调性 / 衰减 / 冗余 / 分段，
#                         自动判断"有效/冗余/另类"，再决定是否入因子库
# 用法：python factor_cards.py
# 产出：./data/factor_cards.csv（结构化因子卡片）+ 终端打印卡片
# =====================================================================
import numpy as np
import pandas as pd

RAW = "./data/raw"

EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
DEFENSIVE = {'bond': '511260.SH', 'cash': '511880.SH', 'gold': '518880.SH'}
BENCH = '510300.SH'          # 基准 = 沪深300ETF
FUTURE_H = 5                 # 未来收益窗口（交易日）


def load_data():
    prices = pd.read_csv(f"{RAW}/etf_prices/all_close.csv", parse_dates=['date'])
    prices = prices.sort_values('date').reset_index(drop=True)
    sent = pd.read_csv(f"{RAW}/sentiment_indices.csv", parse_dates=['date'])
    market = pd.read_csv(f"{RAW}/sentiment_indices_market.csv", parse_dates=['date'])
    aligned = pd.read_csv(f"{RAW}/aligned_sentiment_data.csv", parse_dates=['date'])
    # 基准 510300 的 OHLCV（用于日内价格行为衍生因子）
    ohlc = pd.read_csv(f"{RAW}/stock_price_raw.csv")
    ohlc['date'] = pd.to_datetime(ohlc['trade_date'].astype(str), format='%Y%m%d')
    ohlc = ohlc.sort_values('date').reset_index(drop=True)
    return prices, sent, market, aligned, ohlc


def build_factors(prices, sent, market, aligned, ohlc):
    """构建因子库，返回 (factor_df, future_ret)。

    factor_df 每行一天、每列一个因子（时间序列因子，以基准 510300 为锚）。
    future_ret 为基准未来 FUTURE_H 日累计收益（仅用于研究性评估，非交易信号）。
    """
    codes = EQUITY + list(DEFENSIVE.values())
    close = prices[codes].copy()
    ret = close.pct_change()

    f = pd.DataFrame({'date': prices['date']})
    px = prices[BENCH]

    # ---- 价格类因子（基于基准 510300，时序） ----
    f['momentum_5'] = px.pct_change(5)      # 5日动量
    f['momentum_20'] = px.pct_change(20)    # 20日动量
    f['momentum_60'] = px.pct_change(60)    # 60日动量
    f['reversal_5'] = -px.pct_change(5)     # 5日反转（与动量相反）
    f['volatility_20'] = ret[BENCH].rolling(20).std() * np.sqrt(252)  # 年化波动率

    # ---- 成交量/流动性因子（仅基准，来自 aligned 的 volume） ----
    vol = aligned.set_index('date')['volume']
    vol = vol.reindex(prices['date']).ffill()
    f['volume_ratio'] = (vol.rolling(5).mean() / vol.rolling(20).mean()).values  # 量比
    f['liquidity'] = (vol.rolling(20).mean() / vol.rolling(60).mean()).values    # 流动性变化

    # ---- 情绪因子（全市场新闻情绪，另类因子） ----
    s = sent.set_index('date').reindex(prices['date'])
    for c in ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']:
        train_mean = s[c].iloc[:int(len(s) * 0.7)].mean()   # 训练集均值中性化（比 fillna(0) 更严谨）
        f[c] = s[c].ffill().fillna(train_mean).values

    # ---- 大盘/指数聚焦情绪（因子改进版，标的代理情绪） ----
    mkt = market.set_index('date').reindex(prices['date'])
    for c in ['market_sentiment', 'market_ewma_7d', 'market_ewma_14d']:
        train_mean = mkt[c].iloc[:int(len(mkt) * 0.7)].mean()
        f[c] = mkt[c].ffill().fillna(train_mean).values

    # ---- OHLCV 衍生因子（仅基准 510300，日内价格行为） ----
    o = ohlc.set_index('date').reindex(prices['date'])
    hi, lo, op, cl = o['high'], o['low'], o['open'], o['close']
    o['intraday_range'] = (hi - lo) / cl                                  # 日内振幅
    o['upper_shadow'] = (hi - np.maximum(op, cl)) / cl                    # 上影线（上方抛压）
    o['lower_shadow'] = (np.minimum(op, cl) - lo) / cl                    # 下影线（下方支撑）
    o['close_position'] = (cl - lo) / (hi - lo).replace(0, np.nan)        # 收盘在日内区间的位置
    for c in ['intraday_range', 'upper_shadow', 'lower_shadow', 'close_position']:
        f[c] = o[c].values

    # ---- 截面因子：13 只 ETF 截面动量离散度（top-bottom 动量差） ----
    mom20_cs = ret[EQUITY].rolling(20).mean()
    f['cross_momentum_spread'] = mom20_cs.max(axis=1) - mom20_cs.min(axis=1)

    future_ret = px.pct_change(FUTURE_H).shift(-FUTURE_H)
    f = f.dropna(subset=['date']).reset_index(drop=True)
    return f, future_ret


def ic_series(factor, future_ret):
    """逐期滚动 IC（因子值与未来收益的 Spearman 秩相关），返回 IC 序列。"""
    df = pd.DataFrame({'f': factor, 'r': future_ret}).dropna()
    if len(df) < 30:
        return np.array([])
    # Spearman 相关 = 秩的 Pearson 相关；对秩做滚动相关
    rf = df['f'].rank()
    rr = df['r'].rank()
    ic = rf.rolling(60).corr(rr)
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    return ic.values


def eval_factor(name, factor, future_ret):
    """计算单个因子的核心评估指标，返回一个 dict（因子卡片的一行）。"""
    df = pd.DataFrame({'f': factor, 'r': future_ret}).dropna()
    if len(df) < 30:
        return None
    ic = df['f'].corr(df['r'], method='spearman')           # 全样本 IC
    ic_roll = ic_series(factor, future_ret)                 # 滚动 IC 序列
    icir = ic_roll.mean() / ic_roll.std() if len(ic_roll) and ic_roll.std() > 0 else 0.0

    # 单调性：按因子值分 5 组，看各组未来收益均值（第5组-第1组）
    df = df.copy()
    df['grp'] = pd.qcut(df['f'].rank(method='first'), 5, labels=False)
    grp_ret = df.groupby('grp')['r'].mean()
    monotonic = grp_ret.iloc[-1] - grp_ret.iloc[0] if len(grp_ret) == 5 else 0.0
    return {'factor': name, 'ic': ic, 'icir': icir, 'monotonic_ret': monotonic}


def ic_decay(factor, prices):
    """因子在不同持有期（1/5/10/20 日）上的 IC，看收益衰减（口径与主表一致）。"""
    px = prices[BENCH]
    out = {}
    for h in [1, 5, 10, 20]:
        fr = px.pct_change(h).shift(-h)  # t 到 t+h 的 h 日收益，对齐到 t
        df = pd.DataFrame({'f': factor, 'r': fr}).dropna()
        out[h] = df['f'].corr(df['r'], method='spearman') if len(df) >= 30 else np.nan
    return out


def regime_split(prices):
    """三态市场划分：牛 / 震荡 / 熊，并区分大熊与小回调。"""
    px = prices[BENCH]
    ma20 = px.rolling(20).mean()
    ma60 = px.rolling(60).mean()
    regime = pd.Series(index=prices.index, dtype=object)
    for i in prices.index:
        if pd.isna(ma60[i]) or pd.isna(ma20[i]):
            regime[i] = 'unknown'
        elif ma20[i] > ma60[i] and px[i] > ma20[i]:
            regime[i] = 'bull'      # 牛市：均线多头 + 价格站上 MA20
        elif abs(px[i] / ma60[i] - 1) <= 0.05:
            regime[i] = 'range'     # 震荡：价格在 MA60 附近 ±5%
        else:
            regime[i] = 'bear'      # 熊市（含大熊与小回调，视跌幅深度）
    return regime


def main():
    prices, sent, market, aligned, ohlc = load_data()
    f, future_ret = build_factors(prices, sent, market, aligned, ohlc)

    print("=" * 78)
    print("因子库构建 + 因子卡片评估（命题1 模块1&2）")
    print(f"交易日 {len(f)} | 未来收益窗口 {FUTURE_H} 日 | 基准 {BENCH}")
    print("=" * 78)

    factor_cols = ['momentum_5', 'momentum_20', 'momentum_60', 'reversal_5',
                   'volatility_20', 'volume_ratio', 'liquidity',
                   'intraday_range', 'upper_shadow', 'lower_shadow', 'close_position',
                   'daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment',
                   'market_sentiment', 'market_ewma_7d', 'market_ewma_14d',
                   'cross_momentum_spread']
    regime = regime_split(prices).reindex(f.index).values

    rows = []
    print(f"\n{'因子':<22}{'IC':>8}{'ICIR':>8}{'单调性':>10}{'IC@1d':>8}"
          f"{'IC@5d':>8}{'IC@10d':>8}{'IC@20d':>8}{'牛IC':>8}{'熊IC':>8}")
    print("-" * 88)
    for c in factor_cols:
        if c not in f.columns:
            continue
        factor = f[c]
        card = eval_factor(c, factor, future_ret)
        if card is None:
            continue
        decay = ic_decay(factor, prices)
        df = pd.DataFrame({'f': factor, 'r': future_ret, 'regime': regime}).dropna()
        bull = df[df['regime'] == 'bull']
        bear = df[df['regime'] == 'bear']
        bull_ic = bull['f'].corr(bull['r'], method='spearman')
        bear_ic = bear['f'].corr(bear['r'], method='spearman')
        card.update({'ic1': decay[1], 'ic5': decay[5], 'ic10': decay[10],
                     'ic20': decay[20], 'ic_bull': bull_ic, 'ic_bear': bear_ic})
        rows.append(card)
        print(f"{c:<22}{card['ic']:>8.4f}{card['icir']:>8.3f}"
              f"{card['monotonic_ret']*100:>9.2f}%"
              f"{card['ic1']:>8.4f}{card['ic5']:>8.4f}{card['ic10']:>8.4f}{card['ic20']:>8.4f}"
              f"{card['ic_bull']:>8.4f}{card['ic_bear']:>8.4f}")

    # ---- 因子冗余：因子间相关性矩阵 ----
    print("\n" + "=" * 78)
    print("因子冗余识别（因子间 Spearman 相关，|corr|>0.6 视为冗余候选）")
    print("=" * 78)
    corr = f[factor_cols].corr(method='spearman')
    print(corr.round(2).to_string())

    # ---- 自动分类：有效 / 冗余 / 另类 ----
    print("\n" + "=" * 78)
    print("自动筛选结论（入因子库判定）")
    print("=" * 78)
    for card in rows:
        abs_ic = abs(card['ic'])
        if abs_ic >= 0.03 and card['icir'] > 0.1:
            verdict = "有效（入库）"
        elif card['factor'].startswith(('daily_sentiment', 'ewma', 'overnight', 'market')):
            verdict = "另类（风控/过滤型，低阈值并入）"
        else:
            verdict = "弱/冗余（观察）"
        print(f"  {card['factor']:<24} |IC|={abs_ic:.4f}  ICIR={card['icir']:.3f}  → {verdict}")

    # ---- 情绪因子改进对比：全市场 vs 大盘聚焦 ----
    print("\n" + "=" * 78)
    print("情绪因子改进对比（全市场情绪 → 大盘/指数聚焦情绪）")
    print("=" * 78)
    pairs = [('daily_sentiment', 'market_sentiment'),
             ('ewma_7d', 'market_ewma_7d'),
             ('ewma_14d', 'market_ewma_14d')]
    ic_map = {r['factor']: r for r in rows}
    for old, new in pairs:
        if old in ic_map and new in ic_map:
            o, n = ic_map[old], ic_map[new]
            print(f"  {old:<18} IC={o['ic']:+.4f}  →  {new:<18} IC={n['ic']:+.4f}  "
                  f"({'改进' if abs(n['ic']) >= abs(o['ic']) else '弱化'})")

    # ---- 保存结构化因子卡片 ----
    out_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))}
                           for r in rows])
    import os
    os.makedirs("./data", exist_ok=True)
    out_df.to_csv("./data/factor_cards.csv", index=False)
    print("\n✅ 因子卡片已保存到 ./data/factor_cards.csv")


if __name__ == '__main__':
    main()



