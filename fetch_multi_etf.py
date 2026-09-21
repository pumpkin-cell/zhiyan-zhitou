# fetch_multi_etf.py
# 批量拉取多个 ETF（宽基+行业+风格）日线数据，用于多标的轮动策略
import pandas as pd
from pathlib import Path
from fetch_ETF_price import fetch_all_with_fallback, normalize_df

API = "fund_daily"
START_DATE = "20190101"
END_DATE = "20260630"
OUT_DIR = Path("D:/01_project/ML/国创赛/data/raw/etf_prices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 标的池：宽基 + 行业 + 风格（常见场内 ETF）
ETF_CODES = [
    "510300.SH",  # 沪深300
    "510500.SH",  # 中证500
    "588000.SH",  # 科创50
    "159915.SZ",  # 创业板
    "512480.SH",  # 半导体
    "512660.SH",  # 军工
    "512000.SH",  # 券商
    "512010.SH",  # 医药
    "510880.SH",  # 红利
    "516160.SH",  # 新能源
]

all_close = None
for code in ETF_CODES:
    fixed_params = {"ts_code": code, "start_date": START_DATE, "end_date": END_DATE}
    try:
        df = fetch_all_with_fallback(API, fixed_params, channel="basic", page_limit=5000)
    except Exception as e:
        print(f"{code} 拉取失败: {e}")
        continue
    df = normalize_df(df)
    if df.empty:
        print(f"{code} 无数据，跳过")
        continue

    df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df[['date', 'close', 'vol']].rename(columns={'vol': 'volume'})
    df = df.sort_values('date').drop_duplicates(subset=['date'])

    df.to_csv(OUT_DIR / f"{code.replace('.', '_')}.csv", index=False)
    print(f"{code}: {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}")

    tmp = df[['date', 'close']].rename(columns={'close': code})
    all_close = tmp if all_close is None else all_close.merge(tmp, on='date', how='outer')

if all_close is not None:
    all_close = all_close.sort_values('date').reset_index(drop=True)
    all_close.to_csv(OUT_DIR / "all_close.csv", index=False)
    print(f"\n合并 close 表: {all_close.shape}")
    print(f"日期范围: {all_close['date'].min().date()} ~ {all_close['date'].max().date()}")
else:
    print("没有任何数据被拉取。")
