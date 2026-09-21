# step5_build_sentiment_indices.py
import pandas as pd
import numpy as np

print("📂 加载带情感的新闻数据（全市场）...")
# 确保这个文件是全市场新闻，如果里面有 stock_name 列，不要做任何过滤
news = pd.read_csv("./data/raw/news_with_sentiment.csv", parse_dates=['datetime'])
news['date'] = news['datetime'].dt.date

# 盘中情绪指数：仅 15:00 前的新闻（严格避免用盘后信息，杜绝未来函数）
print("📊 计算盘中情绪指数（15:00 前新闻）...")
news['hour'] = news['datetime'].dt.hour
intraday = news[news['hour'] < 15]
daily_sentiment = intraday.groupby('date')['sentiment_score'].mean().reset_index()
daily_sentiment.columns = ['date', 'daily_sentiment']

# 盘后情绪指数：15:00 后的新闻归入次日交易日（盘后信息属于次日）
print("🌙 计算盘后情绪指数（15:00 后新闻归入次日）...")
after_close = news[news['hour'] >= 15].copy()
after_close['target_datetime'] = after_close['datetime'] + pd.Timedelta(days=1)
after_close['target_datetime'] = after_close['target_datetime'] + pd.offsets.BDay(0)  # 周末顺延到周一
after_close['target_date'] = after_close['target_datetime'].dt.date

overnight_sentiment = after_close.groupby('target_date')['sentiment_score'].mean().reset_index()
overnight_sentiment.columns = ['date', 'overnight_sentiment']

# EWMA 累积情绪指数
print("📈 计算 EWMA 累积情绪指数...")
def calc_ewma(series, alpha=0.3):
    result = np.zeros_like(series, dtype=float)
    for i in range(len(series)):
        if i == 0:
            result[i] = series[i]
        else:
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    return result

daily_sentiment['ewma_7d'] = calc_ewma(daily_sentiment['daily_sentiment'].values, alpha=0.3)
daily_sentiment['ewma_14d'] = calc_ewma(daily_sentiment['daily_sentiment'].values, alpha=0.2)

# 合并（以日度情绪表为主，把隔夜情绪拼上去）
merged = daily_sentiment.merge(overnight_sentiment, on='date', how='left')
merged.to_csv("./data/raw/sentiment_indices.csv", index=False)
print(f"✅ 全市场情绪指数已保存，共 {len(merged)} 个交易日")