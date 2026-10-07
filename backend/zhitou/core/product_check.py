# backend/app/core/product_check.py
# =====================================================================
# C 端「产品体检报告」计算逻辑（面向普通投资者的大白话结论）
# 输入 ETF 代码 + 时间段 → 输出风险指标 + 净值曲线 + 大白话结论
# =====================================================================
import os

import numpy as np
import pandas as pd

from .factor_compute import load_prices
from .metrics import calc_return_metrics
from .paths import DATA_RAW

ETF_NAMES = {
    '510300.SH': '沪深300ETF', '159915.SZ': '创业板ETF', '510500.SH': '中证500ETF',
    '510880.SH': '红利ETF', '512000.SH': '券商ETF', '512010.SH': '医药ETF',
    '512480.SH': '半导体ETF', '512660.SH': '军工ETF', '516160.SH': '新能源ETF',
    '588000.SH': '科创50ETF', '511260.SH': '国债ETF', '511880.SH': '货币ETF',
    '518880.SH': '黄金ETF',
}

# 时间段 → 交易日数（近似 252 天/年）
PERIODS = {"1y": 252, "3y": 756, "5y": 1260, "all": None}


def _risk_level(vol: float, mdd: float) -> str:
    v, d = abs(vol), abs(mdd)
    if v < 10 and d < 10:
        return "低风险"
    if v < 20 and d < 20:
        return "中低风险"
    if v < 30 and d < 30:
        return "中风险"
    return "高风险"


def _market_temp(prices) -> str:
    sent_path = os.path.join(DATA_RAW, "sentiment_indices.csv")
    if not os.path.exists(sent_path):
        return "中性"
    sent = pd.read_csv(sent_path, parse_dates=['date'])
    s = sent.set_index('date').reindex(prices['date'])['daily_sentiment'].ffill().dropna()
    if not len(s):
        return "中性"
    v = float(s.iloc[-1])
    return "偏热" if v > 0.15 else ("偏冷" if v < -0.15 else "中性")


def _credibility(ret) -> tuple:
    """业绩可信度：前半段 vs 后半段收益对比（样本外验证的简化）。"""
    n = len(ret)
    if n < 60:
        return "数据不足", "样本太短，暂无法判断"
    half = n // 2
    first = (1 + ret.iloc[:half]).prod() - 1
    second = (1 + ret.iloc[half:]).prod() - 1
    diff = second - first
    if diff > 0.2:
        return "需警惕", f"近段收益（{second*100:.0f}%）远好于前段（{first*100:.0f}%），业绩可能靠近期行情"
    if diff < -0.2:
        return "需警惕", f"近段收益（{second*100:.0f}%）明显弱于前段（{first*100:.0f}%），业绩在走弱"
    return "可信", f"前段（{first*100:.0f}%）与近段（{second*100:.0f}%）收益接近，业绩较稳定"


def check_product(code: str, period: str = "3y") -> dict:
    prices = load_prices()
    if code not in prices.columns:
        supported = "、".join(list(ETF_NAMES)[:6])
        return {"error": f"未找到代码 {code}，当前支持 {supported} 等 13 只 ETF"}

    # 按时间段截取
    days = PERIODS.get(period, 756)
    if days is not None:
        prices = prices.tail(days).reset_index(drop=True)

    name = ETF_NAMES.get(code, code)
    px = prices[code].astype(float)
    ret = px.pct_change().dropna()
    m = calc_return_metrics(ret.values)

    # 净值曲线（前端画图用）：对齐 ret（少一个首日）
    nav = (1 + ret).cumprod()
    nav_dates = pd.to_datetime(prices['date']).iloc[1:].dt.strftime('%Y-%m-%d').tolist()

    risk = _risk_level(m['annual_vol_pct'], m['max_drawdown_pct'])
    worst = abs(m['max_drawdown_pct'])
    temp = _market_temp(prices)
    credibility, credibility_hint = _credibility(ret)
    temp_hint = ("当前市场情绪指标处于偏热区间" if temp == "偏热"
                 else "当前市场情绪指标处于偏冷区间" if temp == "偏冷"
                 else "当前市场情绪指标处于中性区间")

    conclusions = [
        f"{name}（{code}）近一年年化收益约 {m['annual_return_pct']:.1f}%，历史最大回撤 {m['max_drawdown_pct']:.1f}%。",
        f"最坏情况：极端行情下可能回撤约 {worst:.0f}%，投 10 万最多可能亏约 {worst * 1000:.0f} 元。",
        f"风险等级「{risk}」，对应能承受 {worst:.0f}% 左右波动的投资者。{temp_hint}。",
    ]

    return {
        "code": code,
        "name": name,
        "period": period,
        "annual_return_pct": round(m['annual_return_pct'], 2),
        "max_drawdown_pct": round(m['max_drawdown_pct'], 2),
        "annual_vol_pct": round(m['annual_vol_pct'], 2),
        "sharpe_ratio": round(m['sharpe_ratio'], 3),
        "risk_level": risk,
        "worst_case_pct": round(worst, 2),
        "market_temp": temp,
        "credibility": credibility,
        "credibility_hint": credibility_hint,
        "conclusions": conclusions,
        "nav": [round(float(x), 4) for x in nav.tolist()],
        "nav_dates": nav_dates,
    }


def list_overview(period: str = "3y") -> list:
    """所有产品的概览（供前端首页列表展示）。"""
    prices = load_prices()
    days = PERIODS.get(period, 756)
    if days is not None:
        prices = prices.tail(days).reset_index(drop=True)

    rows = []
    for code, name in ETF_NAMES.items():
        if code not in prices.columns:
            continue
        px = prices[code].astype(float)
        ret = px.pct_change().dropna()
        m = calc_return_metrics(ret.values)
        risk = _risk_level(m['annual_vol_pct'], m['max_drawdown_pct'])
        credibility, _ = _credibility(ret)
        rows.append({
            "code": code, "name": name, "risk_level": risk,
            "annual_return_pct": round(m['annual_return_pct'], 2),
            "max_drawdown_pct": round(m['max_drawdown_pct'], 2),
            "annual_vol_pct": round(m['annual_vol_pct'], 2),
            "credibility": credibility,
        })
    return rows
