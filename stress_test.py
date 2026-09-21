# stress_test.py
# =====================================================================
# 命题1 模块4「多层次回测」原型 —— 生成式重采样压力测试（块自举）
#   对应技术指标：在相同初始状态下生成多条未来轨迹，并完成 ≥3 类压力情景评估，
#   以降低单一历史路径 / 反复调参带来的过拟合风险。
#   诚实说明：当前"生成式"实现为【块自举重采样】，非深度学习生成式模型（如 TimeGAN/扩散），
#   已在商业计划书与答辩中如实标注；进阶方向为参数化 GBM 或 TimeGAN。
#   载体策略：沪深300 趋势 + 黄金防守 + 回撤止损（简化自 rotate_backtest_v4）
# =====================================================================
import numpy as np
import pandas as pd
from metrics import calc_return_metrics

RAW = "./data/raw"
BENCH, GOLD, CASH = '510300.SH', '518880.SH', '511880.SH'


def build_strategy_returns():
    """构建策略日收益（趋势+防守+回撤止损），返回日收益 np.array。"""
    prices = pd.read_csv(f"{RAW}/etf_prices/all_close.csv", parse_dates=['date'])
    prices = prices.sort_values('date').reset_index(drop=True)
    px = prices[BENCH].values
    gold = prices[GOLD].values
    cash = prices[CASH].values
    n = len(prices)

    rets = prices[[BENCH, GOLD, CASH]].pct_change().values
    ma20 = pd.Series(px).rolling(20).mean().values

    strat = np.zeros(n)
    nav = 1.0
    peak = 1.0
    hold = 2  # 0=现金 1=股票 2=黄金
    for t in range(n):
        r = rets[t, hold]
        strat[t] = r if not np.isnan(r) else 0.0
        nav *= (1 + strat[t])
        peak = max(peak, nav)
        dd = nav / peak - 1.0
        trend_up = px[t] > ma20[t] if not np.isnan(ma20[t]) else False
        if dd <= -0.10:
            hold = 0
        elif trend_up:
            hold = 1
        else:
            hold = 2
    return strat


def block_bootstrap(returns, n_paths=200, horizon=252, block=20, seed=0):
    """块自举：从历史日收益中块重采样，生成 n_paths 条 horizon 长轨迹。"""
    rng = np.random.default_rng(seed)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    paths = np.zeros((n_paths, horizon))
    for i in range(n_paths):
        path = []
        while len(path) < horizon:
            start = rng.integers(0, n - block)
            path.extend(returns[start:start + block])
        paths[i] = np.array(path[:horizon])
    return paths


def scenario_shock(paths, mode):
    """对 bootstrap 轨迹施加压力情景冲击。mode: bear / vol / liquidity"""
    rng = np.random.default_rng(1)
    out = paths.copy()
    if mode == 'bear':            # 政策/黑天鹅：随机注入单日 -5% 暴跌
        for i in range(len(out)):
            k = rng.integers(0, out.shape[1])
            out[i, k] -= 0.05
    elif mode == 'vol':           # 波动率冲击：放大 1.5 倍
        out = out * 1.5
    elif mode == 'liquidity':     # 流动性压力：连续 5 日阴跌
        for i in range(len(out)):
            k = rng.integers(0, out.shape[1] - 5)
            out[i, k:k + 5] -= 0.01
    return out


def summarize(paths, name):
    ann = []
    mdd = []
    loss_prob = []
    for p in paths:
        nav = np.cumprod(1 + p)
        ann.append(nav[-1] ** (252 / len(p)) - 1)
        peak = np.maximum.accumulate(nav)
        mdd.append((nav / peak - 1).min())
        loss_prob.append((nav[-1] < 1))
    print(f"\n  【{name}】 {len(paths)} 条未来轨迹")
    print(f"    年化收益: 中位 {np.median(ann)*100:.1f}%  5分位 {np.percentile(ann,5)*100:.1f}%  "
          f"95分位 {np.percentile(ann,95)*100:.1f}%")
    print(f"    最大回撤: 中位 {np.median(mdd)*100:.1f}%  最差(5分位) {np.percentile(mdd,5)*100:.1f}%")
    print(f"    亏损概率: {np.mean(loss_prob)*100:.1f}%")


def main():
    strat = build_strategy_returns()
    m = calc_return_metrics(strat)
    print("=" * 78)
    print("多层次回测 —— 生成式重采样压力测试（块自举，模块4）")
    print("=" * 78)
    print(f"载体策略历史回测：年化 {m['annual_return_pct']:.2f}%  夏普 {m['sharpe_ratio']:.3f}  "
          f"回撤 {m['max_drawdown_pct']:.2f}%")

    base = block_bootstrap(strat, n_paths=200, horizon=252)
    summarize(base, "情景0 基准重采样（历史分布外推）")
    summarize(scenario_shock(base, 'bear'), "情景1 黑天鹅/政策冲击（单日-5%）")
    summarize(scenario_shock(base, 'vol'), "情景2 波动率冲击（×1.5）")
    summarize(scenario_shock(base, 'liquidity'), "情景3 流动性压力（连续阴跌）")

    print("\n" + "=" * 78)
    print("结论：若各情景回撤中位数仍控制在 -15%~-25% 且未出现极端塌陷，")
    print("      说明策略对压力情景具有一定稳健性，非单一历史路径过拟合。")
    print("=" * 78)


if __name__ == '__main__':
    main()
