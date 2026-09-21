# -*- coding: utf-8 -*-
"""
获取沪深300ETF每日数据并保存到本地

默认标的：华泰柏瑞沪深300ETF 510300.SH
接口：fund_daily 场内基金日线行情
区间：20190101 -> 20260630
输出：D:/01_project/ML/国创赛/data/raw
"""

import os
import time
import uuid
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ================== 配置 ==================
# 通道：basic = 普通基础功能；promax = 聚合接口
# basic 失败会自动切 promax
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

API = "fund_daily"          # 场内基金日线行情
START_DATE = "20190101"
END_DATE = "20260630"

# 沪深300ETF代码：
# 510300.SH 华泰柏瑞沪深300ETF
# 510330.SH 华夏沪深300ETF
# 159919.SZ 嘉实沪深300ETF
# 510310.SH 易方达沪深300ETF
ETF_CODES = ["510300.SH"]

# 如果想一次拉多个，把上面改成：
# ETF_CODES = ["510300.SH", "510330.SH", "159919.SZ", "510310.SH"]

# None 表示让接口返回默认全部字段
FIELDS = None
PAGE_LIMIT = 5000


# ================== 工具函数 ==================
def channel_config(channel):
    if channel == "basic":
        return BASIC_BASE, BASIC_KEY, True
    elif channel == "promax":
        return PROMAX_BASE, PROMAX_KEY, False
    else:
        raise ValueError(f"未知通道: {channel}")


def request_page(api, params, channel):
    base, key, verify = channel_config(channel)
    url = f"{base}/{api}"

    headers = {
        "X-API-Key": key,
        "X-Request-Id": f"hs300-etf-{api}-{uuid.uuid4().hex[:12]}",
    }

    r = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
        verify=verify
    )
    r.raise_for_status()

    body = r.json()
    code = body.get("code")

    if code != 0:
        raise RuntimeError(
            f"接口返回错误: code={code}, msg={body.get('msg')}, url={r.url}"
        )

    data = body.get("data") or {}
    fields = data.get("fields") or body.get("fields") or []
    items = data.get("items") or body.get("items") or []

    req_id = r.headers.get("X-Request-Id") or headers["X-Request-Id"]
    return fields, items, req_id


def fetch_all(api, fixed_params, fields=None, channel="basic", page_limit=5000):
    """
    自动分页拉取全部数据。
    分页时只改变 offset，业务参数和 fields 保持不变。
    """
    all_items = []
    final_fields = None
    offset = 0

    while True:
        params = dict(fixed_params)
        params["limit"] = page_limit
        params["offset"] = offset

        if fields:
            params["fields"] = fields

        fields_resp, items, req_id = request_page(api, params, channel)

        if final_fields is None:
            final_fields = fields_resp

        print(
            f"[{channel}] {api} offset={offset} "
            f"本页={len(items)} 条, request_id={req_id}"
        )

        if not items:
            break

        all_items.extend(items)

        if len(items) < page_limit:
            break

        offset += page_limit
        time.sleep(0.2)

    if final_fields is None:
        final_fields = []

    df = pd.DataFrame(
        all_items,
        columns=final_fields if final_fields else None
    )
    return df


def fetch_all_with_fallback(api, fixed_params, fields=None, channel="basic", page_limit=5000):
    try:
        return fetch_all(
            api,
            fixed_params,
            fields=fields,
            channel=channel,
            page_limit=page_limit
        )
    except Exception as e:
        if channel == "basic":
            print(f"基础通道失败，自动切 promax 重试。原因：{e}")
            return fetch_all(
                api,
                fixed_params,
                fields=fields,
                channel="promax",
                page_limit=page_limit
            )
        raise


def normalize_df(df):
    if df.empty:
        return df

    # 保留 trade_date 原始 YYYYMMDD 字符串，同时用它排序
    if "trade_date" in df.columns:
        df["_dt"] = pd.to_datetime(
            df["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )
        df = df.sort_values("_dt").drop(columns=["_dt"])

    # 数值列转数值
    num_cols = [
        "open", "high", "low", "close", "pre_close",
        "change", "pct_chg", "vol", "amount"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 按代码和日期去重
    if "ts_code" in df.columns and "trade_date" in df.columns:
        df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    return df.reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("输出目录:", OUT_DIR)
    print("接口:", API)
    print("日期:", START_DATE, "->", END_DATE)
    print("ETF代码:", ETF_CODES)

    all_dfs = []

    for code in ETF_CODES:
        fixed_params = {
            "ts_code": code,
            "start_date": START_DATE,
            "end_date": END_DATE,
        }

        print(f"\n开始拉取: {code}")

        df = fetch_all_with_fallback(
            API,
            fixed_params,
            fields=FIELDS,
            channel=CHANNEL,
            page_limit=PAGE_LIMIT
        )
        df = normalize_df(df)

        if df.empty:
            print(f"{code} 没有返回数据。请检查代码、日期、权限。")
            continue

        out_file = OUT_DIR / (
            f"hs300_etf_{code.replace('.', '_')}_"
            f"fund_daily_{START_DATE}_{END_DATE}.csv"
        )
        df.to_csv(out_file, index=False, encoding="utf-8-sig")

        print(f"已保存: {out_file}")
        print(f"数据形状: {df.shape}")

        all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged = normalize_df(merged)

        merged_file = OUT_DIR / (
            f"hs300_etf_all_fund_daily_{START_DATE}_{END_DATE}.csv"
        )
        merged.to_csv(merged_file, index=False, encoding="utf-8-sig")

        print(f"\n合并文件已保存: {merged_file}")
        print(f"合并数据形状: {merged.shape}")
    else:
        print("\n没有任何数据被保存。")


if __name__ == "__main__":
    main()