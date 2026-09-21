# baseline.py
# 计算 510300 沪深300ETF 买入持有基准，以及分年度表现
import pandas as pd
import numpy as np
from metrics import calc_return_metrics, print_report

df = pd.read_csv("./data/raw/aligned_sentiment_data.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

close = df['close'].astype(float).values
ret = np.zeros(len(close))
ret[1:] = np.diff(close) / close[:-1]
df['ret'] = ret

print("=" * 60)
print("510300 沪深300ETF 买入持有基准")
print("=" * 60)
print(f"数据区间: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"交易日数: {len(df)}")

# 整体基准
m = calc_return_metrics(ret)
print_report(m, "买入持有（全程）")

# 分年度
print("\n  📅 分年度表现:")
print(f"  {'年份':<6}{'收益%':>10}{'年化波动%':>12}{'最大回撤%':>12}{'胜率%':>8}")
for year, g in df.groupby(df['date'].dt.year):
    r = g['ret'].values
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    mdd = (nav / peak - 1).min() * 100
    win = (r > 0).mean() * 100
    total = (nav[-1] - 1) * 100
    print(f"  {year:<6}{total:>10.2f}{r.std(ddof=1)*np.sqrt(252)*100:>12.2f}{mdd:>12.2f}{win:>8.2f}")

# 按训练/验证/测试分段（3-fold 对应的测试区间）
print("\n  📅 各 Fold 测试区间基准（用于对照）:")
folds = [(0.6, 0.8, 1.0), (0.7, 0.9, 1.0), (0.8, 0.95, 1.0)]
for i, (tr, va, te) in enumerate(folds, 1):
    n = len(df)
    test_df = df.iloc[int(n * va):int(n * te)]
    r = test_df['ret'].values
    mm = calc_return_metrics(r)
    if mm:
        print(f"  Fold{i} 测试期 {test_df['date'].min().date()} ~ {test_df['date'].max().date()}: "
              f"年化 {mm['annual_return_pct']:.2f}%, 夏普 {mm['sharpe_ratio']:.3f}, "
              f"回撤 {mm['max_drawdown_pct']:.2f}%")
