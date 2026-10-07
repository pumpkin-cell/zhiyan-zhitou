# frontend/app.py
# =====================================================================
# 智研智投 · 工行杯版（面向普通大众）
# 产品列表 + 产品体检（净值曲线/指标/结论/适不适合你）+ 金融术语
# 运行：streamlit run frontend/app.py
# =====================================================================
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import matplotlib.font_manager as _fm
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_font_path = os.path.join(_PROJECT_ROOT, "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(_font_path):
    _fm.fontManager.addfont(_font_path)
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_client import check_product, list_overview, list_products

st.set_page_config(page_title="智研智投 · 财富翻译官", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1080px;}
    h1 {font-size: 1.6rem;} h3 {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

st.title("智研智投 · 财富翻译官")
st.divider()

page = st.sidebar.radio("导航", ["产品体检", "金融术语"])

# =====================================================================
# 页面一：产品体检
# =====================================================================
if page == "产品体检":
    # 时间段选择
    period_label = st.radio("时间段", ["近 1 年", "近 3 年", "近 5 年", "全部"],
                            index=1, horizontal=True)
    period_map = {"近 1 年": "1y", "近 3 年": "3y", "近 5 年": "5y", "全部": "all"}

    # 产品概览列表
    try:
        overview = list_overview(period_map[period_label])
    except Exception as e:
        st.error(f"读取产品列表失败：{e}")
        st.stop()

    if overview:
        df = pd.DataFrame(overview).rename(columns={
            "name": "产品", "code": "代码", "risk_level": "风险等级",
            "annual_return_pct": "年化收益%", "max_drawdown_pct": "最大回撤%",
            "annual_vol_pct": "年化波动%", "credibility": "业绩可信度",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # 详情
    options = {f"{p['name']}（{p['code']}）": p['code'] for p in overview}
    col1, col2 = st.columns([2, 1])
    selected = col1.selectbox("选择产品查看体检报告", list(options.keys()))
    run = col2.button("生成体检报告", type="primary", use_container_width=True)

    report = None
    if run:
        with st.spinner("正在生成体检报告..."):
            try:
                report = check_product(options[selected], period_map[period_label])
            except Exception as e:
                st.error(f"生成失败：{e}")
                st.stop()
        if "error" in report:
            st.error(report["error"])
            report = None

    if report:
        st.divider()
        # 净值曲线
        if report.get("nav") and report.get("nav_dates"):
            fig, ax = plt.subplots(figsize=(10, 3.4))
            dates = report["nav_dates"]
            nav = report["nav"]
            ax.plot(dates, nav, color="#2563eb", linewidth=1.6)
            ax.fill_between(dates, 1, nav, alpha=0.06, color="#2563eb")
            ax.axhline(y=1, color="#cbd5e1", linewidth=0.8, linestyle="--")
            ax.set_title(f"{report['name']} 净值走势（归一化）", fontsize=11, loc="left")
            ax.grid(alpha=0.25)
            step = max(1, len(dates) // 6)
            ax.set_xticks(range(0, len(dates), step))
            ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], fontsize=8)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            st.pyplot(fig)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("风险等级", report["risk_level"])
        c2.metric("业绩可信度", report["credibility"])
        c3.metric("年化收益", f"{report['annual_return_pct']:.1f}%")
        c4.metric("最大回撤", f"{report['max_drawdown_pct']:.1f}%")
        c5.metric("年化波动", f"{report['annual_vol_pct']:.1f}%")
        c6.metric("市场温度", report["market_temp"])
        st.caption(report["credibility_hint"])

        st.divider()
        st.subheader("结论")
        for i, c in enumerate(report["conclusions"], 1):
            st.write(f"{i}. {c}")

        # 适不适合你
        st.divider()
        st.subheader("适不适合你？")
        q1 = st.radio("你的年龄", ["30 岁以下", "30-50 岁", "50 岁以上"], horizontal=True)
        q2 = st.radio("这笔钱多久不用", ["1 年以内", "1-3 年", "3 年以上"], horizontal=True)
        q3 = st.radio("能接受的最大亏损", ["10% 以内", "10%-30%", "30% 以上"], horizontal=True)
        q4 = st.radio("投资经验", ["几乎没有", "有一些", "比较丰富"], horizontal=True)
        q5 = st.radio("投资目标", ["保本优先", "稳健增值", "追求高收益"], horizontal=True)

        score = ({"30 岁以下": 3, "30-50 岁": 2, "50 岁以上": 1}[q1]
                 + {"1 年以内": 0, "1-3 年": 1, "3 年以上": 2}[q2]
                 + {"10% 以内": 0, "10%-30%": 2, "30% 以上": 3}[q3]
                 + {"几乎没有": 0, "有一些": 1, "比较丰富": 2}[q4]
                 + {"保本优先": 0, "稳健增值": 1, "追求高收益": 2}[q5])
        pref = "保守" if score <= 3 else ("稳健" if score <= 6 else "进取")
        st.info(f"你的风险偏好：**{pref}**（评分 {score}/12）")

        fit_map = {
            ("低风险", "保守"): "适合", ("低风险", "稳健"): "适合", ("低风险", "进取"): "偏保守",
            ("中低风险", "保守"): "需谨慎", ("中低风险", "稳健"): "适合", ("中低风险", "进取"): "适合",
            ("中风险", "保守"): "不太适合", ("中风险", "稳健"): "需谨慎", ("中风险", "进取"): "适合",
            ("高风险", "保守"): "不太适合", ("高风险", "稳健"): "需谨慎", ("高风险", "进取"): "适合",
        }
        verdict = fit_map.get((report["risk_level"], pref), "需谨慎")
        st.success(f"结论：这只产品（{report['risk_level']}）对你（{pref}）来说 —— **{verdict}**")

    st.divider()
    st.caption("以上内容为产品公开信息与历史数据的客观呈现，不构成投资建议，不承诺任何收益。")

# =====================================================================
# 页面二：金融术语
# =====================================================================
else:
    st.subheader("金融术语，说人话")
    terms = [
        ("年化收益", "把一段时间的收益折算成「一年能赚多少%」，方便不同产品横向比较。"),
        ("最大回撤", "从最高点跌到最低点的最大跌幅。回撤 20% 就是投 10 万最多浮亏 2 万。"),
        ("年化波动", "价格上下波动的剧烈程度，越大越「颠簸」，越容易拿不住。"),
        ("风险等级", "低/中低/中/高，越高代表亏钱的可能性越大。"),
        ("业绩可信度", "历史收益是否稳定：前段和近段收益接近 = 可信；差异大 = 需警惕。"),
        ("市场温度", "市场情绪冷热。过热提示谨慎追高，过冷可能是布局机会。"),
        ("夏普比率", "每承担 1 单位风险能多赚多少，一般 >1 算不错。"),
        ("净值", "基金/ETF 每一份的价值，1 元起，随行情涨跌。"),
    ]
    df = pd.DataFrame(terms, columns=["术语", "大白话解释"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("把专业名词翻译成人话，正是「财富翻译官」在做的事。")
