# fetch_missing_etf.py
# 补拉缺失的 ETF 标的，并重新生成 all_close.csv
import pandas as pd
from pathlib import Path
from fetch_ETF_price import fetch_all_with_fallback, normalize_df

API = "fund_daily"
START_DATE = "20190101"
END_DATE = "20260630"
OUT_DIR = Path("D:/01_project/ML/国创赛/data/raw/etf_prices")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ETF_CODES = ["510300.SH", "510500.SH", "512480.SH", "512010.SH"]

for code in ETF_CODES:
    fixed_params = {"ts_code": code, "start_date": START_DATE, "end_date": END_DATE}
    try:
        df = fetch_all_with_fallback(API, fixed_params, channel="basic", page_limit=5000)
    except Exception as e:
        print(f"{code} 失败: {e}", flush=True)
        continue
    df = normalize_df(df)
    if df.empty:
        print(f"{code} 无数据，跳过", flush=True)
        continue
    df['date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    df = df[['date', 'close', 'vol']].rename(columns={'vol': 'volume'})
    df = df.sort_values('date').drop_duplicates(subset=['date'])
    df.to_csv(OUT_DIR / f"{code.replace('.', '_')}.csv", index=False)
    print(f"{code}: {len(df)} 条, {df['date'].min().date()} ~ {df['date'].max().date()}", flush=True)

# 重新生成 all_close.csv（合并所有已拉到的标的）
codes = sorted(f.stem.replace('_', '.') for f in OUT_DIR.glob('*.csv') if f.name != 'all_close.csv')
all_close = None
for code in codes:
    p = OUT_DIR / f"{code.replace('.', '_')}.csv"
    tmp = pd.read_csv(p, parse_dates=['date'])[['date', 'close']].rename(columns={'close': code})
    all_close = tmp if all_close is None else all_close.merge(tmp, on='date', how='outer')
if all_close is not None:
    all_close = all_close.sort_values('date').reset_index(drop=True)
    all_close.to_csv(OUT_DIR / "all_close.csv", index=False)
    print(f"all_close.csv 生成完成: {all_close.shape}, 列: {list(all_close.columns)}", flush=True)
else:
    print("没有任何数据", flush=True)
