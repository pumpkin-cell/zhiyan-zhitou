# backtest.py
# DEPRECATED：历史版本，已被 rotate_backtest_v4.py 取代（本文件止损止盈逻辑未实际实现，勿用于主流程）
import numpy as np
import pandas as pd


def backtest(predictions, actual_returns, close_prices=None,
             initial_capital=100000,
             fee_rate=0.0003, slippage=0.001,
             limit_up=0.10, limit_down=-0.10,
             ma_window=20, signal_confirm_days=3,
             enable_trend_filter=True, enable_signal_smooth=True,
             ):
    """
    完整回测函数（含趋势过滤 + 信号平滑 + 止损止盈）

    参数:
        predictions: 预测的收益率序列
        actual_returns: 实际的收益率序列（label）
        close_prices: 收盘价序列（用于计算均线，趋势过滤需要）
        initial_capital: 初始资金
        fee_rate: 手续费率（万三=0.0003）
        slippage: 滑点（0.1%）
        limit_up: 涨停阈值（10%，ETF 场内基金）
        limit_down: 跌停阈值（-10%，ETF 场内基金）
        ma_window: 均线窗口（默认20日）
        signal_confirm_days: 信号确认天数（连续N天信号一致才执行）
        enable_trend_filter: 是否启用趋势过滤
        enable_signal_smooth: 是否启用信号平滑
        stop_loss: 止损阈值（默认 -5%）
        take_profit: 止盈阈值（默认 +15%）

    返回:
        dict: 包含所有评价指标和净值曲线
    """
    if len(predictions) == 0 or len(actual_returns) == 0:
        return None

    # 截断到相同长度
    min_len = min(len(predictions), len(actual_returns))
    signals = predictions[:min_len]
    rets = actual_returns[:min_len]

    # 收盘价（用于计算均线）
    if close_prices is not None:
        close_prices = close_prices[:min_len]
    else:
        # 如果没有提供收盘价，用收益率模拟（不准确，但不会报错）
        close_prices = np.cumprod(1 + rets)

    capital = initial_capital
    position = 0
    cash = capital
    equity_curve = [capital]
    trade_count = 0
    trade_records = []
    buy_price = None  # 🔥 记录买入价格，用于止损止盈

    # ========== 信号平滑缓冲区 ==========
    signal_buffer = []

    # ========== 计算均线（修复只读错误） ==========
    if enable_trend_filter:
        ma = pd.Series(close_prices).rolling(window=ma_window).mean().values
        ma = ma.copy()  # 创建可写副本
        # 前 ma_window-1 天没有均线，用收盘价代替（不触发趋势过滤）
        ma[:ma_window - 1] = close_prices[:ma_window - 1]

    # ========== 主循环 ==========
    for i in range(1, len(signals)):
        # ---- 1. 当前信号 ----
        raw_signal = signals[i - 1]  # 前一天收盘后生成的信号
        ret = rets[i]  # 当天的实际收益率
        close = close_prices[i] if close_prices is not None else 1
        ma_today = ma[i] if enable_trend_filter else close

        # ---- 2. 信号平滑 ----
        if enable_signal_smooth:
            signal_buffer.append(raw_signal)
            if len(signal_buffer) > signal_confirm_days:
                signal_buffer.pop(0)
            # 只有连续 N 天信号一致才执行
            if len(signal_buffer) >= signal_confirm_days:
                if all(s > 0 for s in signal_buffer):
                    smooth_signal = 1
                elif all(s <= 0 for s in signal_buffer):
                    smooth_signal = 0
                else:
                    smooth_signal = -1  # 信号不一致，不操作
            else:
                smooth_signal = -1  # 缓冲区未满，不操作
        else:
            smooth_signal = 1 if raw_signal > 0 else 0

        # ---- 3. 趋势过滤（只在上升趋势中买入） ----
        if enable_trend_filter and smooth_signal == 1:
            if close <= ma_today:
                smooth_signal = -1  # 不在上升趋势中，取消买入


        # ---- 4. 涨跌停过滤 ----
        if ret >= limit_up or ret <= limit_down:
            equity_curve.append(cash + position)
            continue

        # ---- 5. 执行交易 ----
        if smooth_signal == 1 and cash > 0 :
            # 买入
            price = 1 * (1 + slippage)
            max_shares = int((cash * (1 - fee_rate)) / price)
            if max_shares > 0:
                cost = max_shares * price
                fee = cost * fee_rate
                cash -= (cost + fee)
                position += max_shares
                buy_price = price  # 🔥 记录买入价格
                trade_count += 1
                trade_records.append({
                    'date': i,
                    'action': 'BUY',
                    'shares': max_shares,
                    'price': price,
                    'fee': fee
                })
        elif smooth_signal == 0 and position > 0:
            # 卖出（清仓）
            price = 1 * (1 - slippage)
            shares_sold = position          # 🔥 先记录卖出前的持仓，再清零
            revenue = shares_sold * price
            fee = revenue * fee_rate
            cash += (revenue - fee)
            position = 0
            buy_price = None  # 🔥 清空买入价格
            trade_count += 1
            trade_records.append({
                'date': i,
                'action': 'SELL',
                'shares': shares_sold,
                'price': price,
                'fee': fee
            })
        # smooth_signal == -1 时，不操作

        # ---- 6. 记录当日净值 ----
        equity = cash + position * (1 + ret)
        equity_curve.append(equity)

    # ========== 计算评价指标 ==========
    equity_curve = np.array(equity_curve)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0])

    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    annual_return = (equity_curve[-1] / initial_capital) ** (252 / len(equity_curve)) - 1 if len(
        equity_curve) > 0 else 0
    annual_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
    sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
    peak = np.maximum.accumulate(equity_curve)
    max_dd = (equity_curve - peak).min() / peak.max() if peak.max() > 0 else 0
    win_rate = np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0

    # 🔥 统计止损止盈次数
    stop_loss_count = sum(1 for r in trade_records if r.get('action') == 'STOP_LOSS')
    take_profit_count = sum(1 for r in trade_records if r.get('action') == 'TAKE_PROFIT')

    return {
        'total_return_pct': total_return,
        'annual_return_pct': annual_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'win_rate_pct': win_rate * 100,
        'num_trades': trade_count,
        'final_value': equity_curve[-1],
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'trade_records': trade_records,
    }


def print_backtest_report(result, name="模型"):
    """打印回测报告（含止损止盈统计）"""
    if result is None:
        print(f"  {name}: 回测失败")
        return

    print(f"\n  📊 {name} 回测结果:")
    print(f"     累计收益: {result['total_return_pct']:.2f}%")
    print(f"     年化收益: {result['annual_return_pct']:.2f}%")
    print(f"     夏普比率: {result['sharpe_ratio']:.4f}")
    print(f"     最大回撤: {result['max_drawdown_pct']:.2f}%")
    print(f"     胜率: {result['win_rate_pct']:.2f}%")
    print(f"     交易次数: {result['num_trades']}")
    print(f"     最终净值: {result['final_value']:.2f}")
