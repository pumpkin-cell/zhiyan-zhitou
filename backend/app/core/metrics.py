# metrics.py
# 完整量化指标计算模块（年化、夏普、IR、回撤、胜率、盈亏比、超额收益）
import numpy as np


def calc_return_metrics(returns, benchmark_returns=None, periods_per_year=252, rf=0.02):
    """
    计算策略日收益率序列的完整指标。

    参数:
        returns: 策略日收益率（np.array，可含 NaN，会被剔除）
        benchmark_returns: 基准日收益率（用于计算超额收益与 IR）
        periods_per_year: 年化周期数（日频=252）
        rf: 无风险利率（默认 2%）
    返回:
        dict: 各指标
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return None

    n = len(r)
    nav = np.cumprod(1.0 + r)

    total_return = nav[-1] - 1.0
    annual_return = nav[-1] ** (periods_per_year / n) - 1.0
    annual_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0.0

    # 最大回撤
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    max_dd = float(dd.min())

    # 日级别胜率（收益 > 0 的天数占比）
    daily_win_rate = float((r > 0).mean())

    # 盈亏比（日级别）：平均盈利 / 平均亏损
    gains = r[r > 0]
    losses = r[r < 0]
    avg_gain = float(gains.mean()) if len(gains) > 0 else 0.0
    avg_loss = float(-losses.mean()) if len(losses) > 0 else 0.0
    daily_profit_loss = avg_gain / avg_loss if avg_loss > 0 else 0.0

    # 超额收益与信息比率 IR（标准定义：超额年化 = 策略年化 - 基准年化）
    annual_excess = 0.0
    ir = 0.0
    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float)
        b = b[~np.isnan(b)]
        m = min(len(r), len(b))
        excess = r[:m] - b[:m]
        if len(excess) > 0:
            b_nav = np.cumprod(1.0 + b[:m])
            annual_return_bench = b_nav[-1] ** (periods_per_year / m) - 1.0
            annual_excess = annual_return - annual_return_bench
            tracking_error = excess.std(ddof=1) * np.sqrt(periods_per_year)
            ir = annual_excess / tracking_error if tracking_error > 0 else 0.0

    return {
        'total_return_pct': total_return * 100,
        'annual_return_pct': annual_return * 100,
        'annual_vol_pct': annual_vol * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'daily_win_rate_pct': daily_win_rate * 100,
        'daily_profit_loss_ratio': daily_profit_loss,
        'annual_excess_pct': annual_excess * 100,
        'information_ratio': ir,
        'nav': nav,
        'daily_returns': r,
        'n_days': n,
    }


def calc_trade_metrics(trade_returns, periods_per_year=252, rf=0.02):
    """
    计算交易级别指标（每笔完整交易记一次收益）。

    参数:
        trade_returns: 每笔交易的收益率列表（np.array）
    返回:
        dict: 胜率、盈亏比、交易次数等
    """
    tr = np.asarray(trade_returns, dtype=float)
    tr = tr[~np.isnan(tr)]
    if len(tr) == 0:
        return {'num_trades': 0, 'win_rate_pct': 0.0, 'profit_loss_ratio': 0.0,
                'avg_win_pct': 0.0, 'avg_loss_pct': 0.0}

    wins = tr[tr > 0]
    losses = tr[tr < 0]
    win_rate = len(wins) / len(tr)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(-losses.mean()) if len(losses) > 0 else 0.0
    plr = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        'num_trades': int(len(tr)),
        'win_rate_pct': win_rate * 100,
        'profit_loss_ratio': plr,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
    }


def print_report(metrics, name="策略"):
    """打印完整指标报告。"""
    if metrics is None:
        print(f"  {name}: 无有效数据")
        return
    print(f"\n  📊 {name} 指标报告:")
    print(f"     年化收益: {metrics['annual_return_pct']:.2f}%")
    print(f"     年化波动: {metrics['annual_vol_pct']:.2f}%")
    print(f"     夏普比率: {metrics['sharpe_ratio']:.4f}")
    print(f"     最大回撤: {metrics['max_drawdown_pct']:.2f}%")
    print(f"     日胜率:   {metrics['daily_win_rate_pct']:.2f}%")
    print(f"     日盈亏比: {metrics['daily_profit_loss_ratio']:.4f}")
    if metrics['information_ratio'] != 0:
        print(f"     年化超额: {metrics['annual_excess_pct']:.2f}%")
        print(f"     信息比率IR: {metrics['information_ratio']:.4f}")
