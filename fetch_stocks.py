# -*- coding: utf-8 -*-
"""
获取 10 支行业龙头个股日线数据（个股初步验证用）
接口：daily（股票日线，区别于 fund_daily 场内基金）
区间：20190101 -> 20260630
输出：D:/01_project/ML/国创赛/data/raw/stocks_10.csv
"""
import os
import time
import uuid
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

CHANNEL = "basic"
BASIC_BASE = "http://datahubco.com/app-api/openapi/v1/tushare"
BASIC_KEY = os.getenv(
    "DATAHUBCO_API_KEY",
    "dba548a206a453c197f9175189b757374fa6db9554bb29e69efea127"
)
PROMAX_BASE = "https://pcd.mobcvb.cn/tushare/pro"
PROMAX_KEY = os.getenv(
    "PROMAX_API_KEY",
    "tsr_1FjRkziz3M7m0aLcTk0ZgnK03__xO3EYq0ZdwQqdwSE"
)

OUT_DIR = Path(r"D:/01_project/ML/国创赛/data/raw")
API = "daily"
START_DATE = "20190101"
END_DATE = "20260630"

# 10 支行业龙头（2019 前上市，退市风险极低，行业分散）
STOCK_CODES = [
    "600519.SH",  # 贵州茅台 白酒
    "600036.SH",  # 招商银行 银行
    "601318.SH",  # 中国平安 保险
    "600276.SH",  # 恒瑞医药 医药
    "000333.SZ",  # 美的集团 家电
    "002594.SZ",  # 比亚迪 汽车
    "600887.SH",  # 伊利股份 消费
    "002415.SZ",  # 海康威视 科技/安防
    "600900.SH",  # 长江电力 公用事业
    "000002.SZ",  # 万科A 地产
]

PAGE_LIMIT = 5000


def channel_config(channel):
    if channel == "basic":
        return BASIC_BASE, BASIC_KEY, True
    elif channel == "promax":
        return PROMAX_BASE, PROMAX_KEY, False
    raise ValueError(f"未知通道: {channel}")


def request_page(api, params, channel):
    base, key, verify = channel_config(channel)
    url = f"{base}/{api}"
    headers = {
        "X-API-Key": key,
        "X-Request-Id": f"stocks10-{api}-{uuid.uuid4().hex[:12]}",
    }
    r = requests.get(url, headers=headers, params=params, timeout=30, verify=verify)
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"接口错误: code={body.get('code')}, msg={body.get('msg')}")
    data = body.get("data") or {}
    fields = data.get("fields") or body.get("fields") or []
    items = data.get("items") or body.get("items") or []
    return fields, items


def fetch_all(api, fixed_params, channel="basic", page_limit=5000):
    all_items, final_fields, offset = [], None, 0
    while True:
        params = dict(fixed_params)
        params["limit"] = page_limit
        params["offset"] = offset
        fields_resp, items = request_page(api, params, channel)
        if final_fields is None:
            final_fields = fields_resp
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_limit:
            break
        offset += page_limit
        time.sleep(0.2)
    df = pd.DataFrame(all_items, columns=final_fields if final_fields else None)
    return df


def fetch_all_with_fallback(api, fixed_params, channel="basic", page_limit=5000):
    try:
        return fetch_all(api, fixed_params, channel=channel, page_limit=page_limit)
    except Exception as e:
        if channel == "basic":
            print(f"  基础通道失败，切 promax。原因：{e}")
            return fetch_all(api, fixed_params, channel="promax", page_limit=page_limit)
        raise


def normalize_df(df):
    if df.empty:
        return df
    num_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ts_code" in df.columns and "trade_date" in df.columns:
        df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    return df.reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"拉取 10 支个股日线：{API}，{START_DATE} -> {END_DATE}")
    all_dfs = []
    for code in STOCK_CODES:
        fixed = {"ts_code": code, "start_date": START_DATE, "end_date": END_DATE}
        print(f"\n开始拉取: {code}")
        try:
            df = fetch_all_with_fallback(API, fixed, channel=CHANNEL, page_limit=PAGE_LIMIT)
            df = normalize_df(df)
        except Exception as e:
            print(f"  {code} 失败: {e}")
            continue
        if df.empty:
            print(f"  {code} 无数据")
            continue
        print(f"  {code}: {len(df)} 条, {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        all_dfs.append(df)

    if not all_dfs:
        print("❌ 未拉取到任何个股数据")
        return

    merged = normalize_df(pd.concat(all_dfs, ignore_index=True))
    out = OUT_DIR / "stocks_10.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存: {out}，共 {len(merged)} 条，{merged['ts_code'].nunique()} 支股票")


if __name__ == "__main__":
    main()
