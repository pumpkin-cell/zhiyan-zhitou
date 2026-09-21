# reproduce_report.py
# =====================================================================
# 命题1 模块1「研报复现」原型
# 复现对象：Jegadeesh & Titman (1993) 动量效应 + A股短期反转（对照）
#   原文结论：美股上，过去3-12个月赢家组合未来继续跑赢输家（动量有效）
#   本复现：  在 13 只 A股 ETF 上，用「过去N日收益排序」构建 赢家/输家组合，
#             验证该结论在 A股 ETF 截面上的可复现性。
# 产出：终端打印「原文结论 vs 本复现」对照表
# =====================================================================
import numpy as np
import pandas as pd
from metrics import calc_return_metrics, print_report

RAW = "./data/raw"
EQUITY = ['159915.SZ', '510300.SH', '510500.SH', '510880.SH', '512000.SH',
          '512010.SH', '512480.SH', '512660.SH', '516160.SH', '588000.SH']
BENCH = '510300.SH'


def load():
    prices = pd.read_csv(f"{RAW}/etf_prices/all_close.csv", parse_dates=['date'])
    prices = prices.sort_values('date').reset_index(drop=True)
    return prices


def portfolio_returns(prices, lookback=20, hold=20, top_k=3, long_short=True):
    """按过去 lookback 日收益排序，构建赢家/输家组合，返回各组日收益序列。"""
    close = prices[EQUITY].astype(float)
    ret = close.pct_change()
    n = len(prices)
    # 组合日收益：未调仓日沿用当前持仓等权收益
    winner_ret = np.zeros(n)
    loser_ret = np.zeros(n)
    current_w = []
    current_l = []

    for t in range(n):
        # 当日收益（用当前持仓）
        if current_w:
            vals = ret.iloc[t, current_w].values
            vals = vals[~np.isnan(vals)]
            winner_ret[t] = vals.mean() if len(vals) else 0.0
        if current_l:
            vals = ret.iloc[t, current_l].values
            vals = vals[~np.isnan(vals)]
            loser_ret[t] = vals.mean() if len(vals) else 0.0

        # 调仓判断（每 hold 日，且信号可用）
        if t % hold == 0 and t >= lookback:
            mom = close.iloc[t] / close.iloc[t - lookback] - 1.0  # 过去 lookback 日累计收益
            mom = mom.dropna()
            if len(mom) >= top_k * 2:
                order = mom.sort_values(ascending=False)
                idx_map = {c: i for i, c in enumerate(EQUITY)}
                current_w = [idx_map[c] for c in order.index[:top_k]]
                current_l = [idx_map[c] for c in order.index[-top_k:]]

    winner_ret = pd.Series(winner_ret, index=prices['date'])
    loser_ret = pd.Series(loser_ret, index=prices['date'])
    if long_short:
        return winner_ret - loser_ret, winner_ret, loser_ret
    return winner_ret, loser_ret


def main():
    prices = load()
    bench_ret = prices[BENCH].pct_change().fillna(0).values
    print("=" * 78)
    print("研报复现：Jegadeesh-Titman (1993) 动量效应 → 13 只 A股 ETF 截面")
    print(f"交易日 {len(prices)} | 排序窗口=过去20日 | 持有=20日 | 赢家/输家各 top3")
    print("=" * 78)

    print_report(calc_return_metrics(bench_ret), "基准：沪深300ETF 买入持有")

    # 多组参数扫描，看动量是否稳定
    print("\n" + "-" * 78)
    print(f"{'lookback':>8}{'hold':>6}{'赢家-输家 年化%':>16}{'夏普':>8}{'回撤%':>8}  → 结论")
    print("-" * 78)
    for lb, hd in [(5, 5), (20, 20), (60, 20), (120, 20)]:
        ls, w, l = portfolio_returns(prices, lookback=lb, hold=hd, top_k=3)
        m = calc_return_metrics(ls.values)
        tag = "动量(原文)" if m['annual_return_pct'] > 2 else "反转(反向)"
        print(f"{lb:>8}{hd:>6}{m['annual_return_pct']:>15.2f}%{m['sharpe_ratio']:>8.3f}"
              f"{m['max_drawdown_pct']:>8.2f}%  → {tag}")

    # 固定经典参数（20/20）详细对照
    ls, w, l = portfolio_returns(prices, lookback=20, hold=20, top_k=3)
    print("\n" + "=" * 78)
    print("复现对照表（原文 vs 本复现）")
    print("=" * 78)
    print("  原文(JT 1993, 美股): 过去3-12月赢家组合 未来显著跑赢输家组合（动量溢价为正）")
    print(f"  本复现(A股ETF, 20日): 赢家-输家 年化 {calc_return_metrics(ls.values)['annual_return_pct']:.2f}%")
    print(f"  结论: {'✅ 复现成功（动量有效）' if calc_return_metrics(ls.values)['annual_return_pct'] > 2 else '⚠️ 不可复现（A股ETF短期呈反转，动量反向）'}")
    print("\n  说明: A股 ETF 日频/周频呈短期反转（与 factor_cards.py 中 reversal_5 正IC、")
    print("        momentum 负IC 的结论一致），JT 动量效应在美股中期成立，在 A股")
    print("        短期截面不成立——这正是『研报复现』的价值：检验结论的迁移性。")
    print("\n  赢家组合:")
    print_report(calc_return_metrics(w.values), "赢家组合(top3)")
    print_report(calc_return_metrics(l.values), "输家组合(bottom3)")


if __name__ == '__main__':
    main()
