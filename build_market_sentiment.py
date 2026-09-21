# build_market_sentiment.py
# =====================================================================
# 因子改进版：把「全市场新闻情绪」升级为「大盘/指数聚焦情绪」
# 背景：news_with_sentiment.csv 无股票标签，经扫描「沪深300」直接相关新闻仅 0.12%
#      （过稀疏，无法成稳定日频因子），故用「大盘/指数」关键词构造标的代理情绪。
# 产出：./data/raw/sentiment_indices_market.csv（大盘情绪日度指数）
# =====================================================================
import pandas as pd
import numpy as np

PATH = "data/raw/news_with_sentiment.csv"
# 大盘/指数关键词（沪深300 直接相关 + 宽基指数语境）
KW = "沪深300|沪深 300|300ETF|510300|300指数|沪指|上证指数|上证综指|大盘|沪深两市|A股市场|蓝筹|白马股|权重股|指数"

frames = []
hit = 0
total = 0
chunk_iter = pd.read_csv(PATH, chunksize=500000, usecols=["datetime", "title", "content", "sentiment_score"],
                         dtype=str, on_bad_lines="skip")
for ch in chunk_iter:
    total += len(ch)
    txt = (ch["title"].fillna("") + " " + ch["content"].fillna(""))
    m = txt.str.contains(KW, regex=True, na=False)
    hit += int(m.sum())
    if m.any():
        sub = ch[m].copy()
        sub["datetime"] = pd.to_datetime(sub["datetime"], errors="coerce")
        sub = sub[sub["datetime"].dt.hour < 15]  # 仅 15:00 前的新闻（盘中，杜绝未来函数）
        sub["date"] = sub["datetime"].dt.date
        sub["sentiment_score"] = pd.to_numeric(sub["sentiment_score"], errors="coerce")
        frames.append(sub[["date", "sentiment_score"]].dropna())

print(f"全量 {total} 条，命中大盘关键词 {hit} 条 ({hit/max(total,1)*100:.2f}%)")

if frames:
    df = pd.concat(frames, ignore_index=True)
    daily = df.groupby("date")["sentiment_score"].agg(["mean", "count"]).reset_index()
    daily.columns = ["date", "market_sentiment", "news_count"]

    def ewma(s, alpha=0.3):
        out = np.zeros_like(s, dtype=float)
        for i in range(len(s)):
            out[i] = s[i] if i == 0 else alpha * s[i] + (1 - alpha) * out[i - 1]
        return out

    daily["market_ewma_7d"] = ewma(daily["market_sentiment"].values, alpha=0.3)
    daily["market_ewma_14d"] = ewma(daily["market_sentiment"].values, alpha=0.2)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_csv("data/raw/sentiment_indices_market.csv", index=False)
    print(f"✅ 大盘情绪指数已保存：{len(daily)} 个交易日，"
          f"日期 {daily['date'].min()} ~ {daily['date'].max()}，日均新闻 {daily['news_count'].mean():.0f} 条")
else:
    print("⚠️ 未命中任何大盘关键词，请检查关键词")

