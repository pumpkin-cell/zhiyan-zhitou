# strategy_backtest.py
# 经典量化择时策略回测 + 参数扫描（严格无未来函数：t 日收盘信号 -> t+1 日执行）
import pandas as pd
import numpy as np
from metrics import calc_return_metrics, calc_trade_metrics, print_report

# ========== 加载数据 ==========
df = pd.read_csv("./data/raw/aligned_sentiment_data.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

close = df['close'].astype(float).values
ret = np.zeros(len(close))
ret[1:] = np.diff(close) / close[:-1]
n = len(ret)

# 情绪因子列（已在 pipeline 中按日期对齐，t 日可用）
sent_cols = [c for c in ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment'] if c in df.columns]
sent = {c: df[c].astype(float).values for c in sent_cols}


# ========== 工具函数 ==========
def rolling_mean(x, window):
    s = pd.Series(x)
    return s.rolling(window).mean().values


def backtest(signal, ret_, fee_rate=0.0003):
    """
    signal[t]：t 日收盘时给出的信号（0/1，只用 <=t 信息）
    实际 t+1 日持有，策略日收益 strat_ret[t+1] = signal[t] * ret[t+1]
    """
    signal = np.asarray(signal)
    position = np.concatenate([[0], signal[:-1]])  # position[t] 表示 t 日是否持仓
    strat_ret = position * ret_

    # 交易成本（双边，换手日扣除）
    turnover = np.abs(np.diff(np.concatenate([[0], position])))
    costs = turnover * fee_rate
    strat_ret_net = strat_ret - costs

    # 提取交易（连续持仓段）
    trades = []
    in_pos = False
    cum = 1.0
    for t in range(len(position)):
        if position[t] == 1:
            if not in_pos:
                in_pos = True
                cum = 1.0 + ret_[t]
            else:
                cum *= (1.0 + ret_[t])
        else:
            if in_pos:
                in_pos = False
                trades.append(cum - 1.0)
    if in_pos:
        trades.append(cum - 1.0)

    return strat_ret_net, position, np.array(trades)


def evaluate(signal, ret_, benchmark, fee_rate=0.0003, name="策略"):
    sr, pos, trades = backtest(signal, ret_, fee_rate)
    m = calc_return_metrics(sr, benchmark_returns=benchmark)
    t = calc_trade_metrics(trades)
    if m is None:
        return None
    m.update(t)
    m['name'] = name
    return m


# ========== 策略信号生成（全部只用 <=t 信息，无未来函数） ==========
def sig_trend_ma(window):
    ma = rolling_mean(close, window)
    return (close > ma).astype(int)


def sig_momentum(lookback):
    mom = np.zeros(n)
    mom[lookback:] = close[lookback:] / close[:-lookback] - 1.0
    return (mom > 0).astype(int)


def sig_dual_ma(short, long):
    ma_s = rolling_mean(close, short)
    ma_l = rolling_mean(close, long)
    return (ma_s > ma_l).astype(int)


def sig_sentiment(col, thresh):
    s = sent[col]
    return (s > thresh).astype(int)


# ========== 参数扫描（全样本，看理论上限） ==========
print("=" * 70)
print("经典策略全样本扫描（无未来函数；信号 t 日 -> t+1 执行；手续费万三）")
print("=" * 70)

results = []


def scan(name, fn, params):
    global results
    best = None
    for p in params:
        try:
            sig = fn(p) if not isinstance(p, tuple) else fn(*p)
            m = evaluate(sig, ret, ret, name=f"{name}{p}")
            if m is None:
                continue
            results.append(m)
            if best is None or m['sharpe_ratio'] > best['sharpe_ratio']:
                best = m
        except Exception as e:
            print(f"   [{name}{p}] 跳过: {e}")
    if best:
        print_report(best, f"★ {name} 最优")


# 趋势跟踪：MA 窗口
scan("趋势MA", sig_trend_ma, [5, 10, 20, 30, 60, 120])

# 动量
scan("动量", sig_momentum, [5, 10, 20, 40, 60, 120])

# 双均线
scan("双均线", sig_dual_ma, [(5, 20), (10, 30), (10, 60), (20, 60), (20, 120), (50, 200)])

# 情绪择时（阈值）
for c in sent_cols:
    s = sent[c]
    lo, hi = np.nanpercentile(s, 10), np.nanpercentile(s, 90)
    ths = list(np.linspace(lo, hi, 8))
    scan(f"情绪[{c}]", sig_sentiment, [(c, th) for th in ths])

# ========== 汇总：全样本最优（按夏普排序） ==========
print("\n" + "=" * 70)
print("全样本各策略 Top10（按夏普排序）")
print("=" * 70)
valid = [r for r in results if r is not None]
valid.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
print(f"  {'策略':<22}{'夏普':>8}{'年化%':>9}{'回撤%':>9}{'超额%':>9}{'IR':>7}{'交易':>6}{'胜率%':>7}{'盈亏比':>7}")
for r in valid[:10]:
    print(f"  {r['name']:<22}{r['sharpe_ratio']:>8.3f}{r['annual_return_pct']:>9.2f}"
          f"{r['max_drawdown_pct']:>9.2f}{r['annual_excess_pct']:>9.2f}{r['information_ratio']:>7.3f}"
          f"{r['num_trades']:>6}{r['win_rate_pct']:>7.2f}{r['profit_loss_ratio']:>7.3f}")
