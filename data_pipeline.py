# data_pipeline.py
import os
import pandas as pd
import numpy as np
from config_ETF import RAW_DATA_DIR, OUTPUT_DIR, FOLD_CONFIGS, START_DATE, END_DATE

print("=" * 50)
print("🔧 阶段2：数据对齐、特征工程与3-Fold切分（含情绪因子合并）")
print("=" * 50)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- 1. 读取原始数据 -------------------
print("\n📂 读取原始数据...")

# 读取 510300 ETF 价格（兼容 tushare fund_daily 格式：trade_date/vol）
stock = pd.read_csv(os.path.join(RAW_DATA_DIR, "stock_price_raw.csv"))
if 'trade_date' in stock.columns and 'date' not in stock.columns:
    stock['date'] = pd.to_datetime(stock['trade_date'].astype(str), format='%Y%m%d')
if 'vol' in stock.columns and 'volume' not in stock.columns:
    stock['volume'] = stock['vol']
stock = stock[['date', 'close', 'volume']].copy()
stock['date'] = pd.to_datetime(stock['date'])

# 读取新闻（沪深300 ETF 采用全市场新闻；news_raw.csv 不存在时 news_text 置空）
news_path = os.path.join(RAW_DATA_DIR, "news_raw.csv")
if os.path.exists(news_path):
    news_df = pd.read_csv(news_path, parse_dates=['date'])
    news_df['date'] = pd.to_datetime(news_df['date'])
else:
    print("   ⚠️ 未找到 news_raw.csv，news_text 置空（模型仅使用情绪因子列）")
    news_df = pd.DataFrame(columns=['date', 'title'])
    news_df['date'] = pd.to_datetime(news_df['date'])

print(f"   股价: {len(stock)} 条")
print(f"   新闻: {len(news_df)} 条")

# ------------------- 2. 过滤日期范围 -------------------
print(f"\n📅 过滤日期范围: {START_DATE} ~ {END_DATE}")
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)

stock = stock[(stock['date'] >= start_dt) & (stock['date'] <= end_dt)]
news_df = news_df[(news_df['date'] >= start_dt) & (news_df['date'] <= end_dt)]

print(f"   过滤后股价: {len(stock)} 条")
print(f"   过滤后新闻: {len(news_df)} 条")

# ------------------- 3. T-1 新闻对齐 -------------------
print("\n🔗 执行 T-1 新闻与 T 日股价对齐...")

stock = stock.sort_values('date').reset_index(drop=True)

if not news_df.empty:
    news_grouped = news_df.groupby('date')['title'].apply(lambda x: ' [SEP] '.join(x)).reset_index()
    news_grouped.columns = ['date', 'news_text']
    news_grouped = news_grouped.sort_values('date')

    merged = pd.merge_asof(
        stock,
        news_grouped,
        left_on='date',
        right_on='date',
        direction='backward'
    )
    merged['news_text'] = merged['news_text'].fillna('')
else:
    # 无新闻文本：直接生成空 news_text 列（不影响模型训练）
    merged = stock.copy()
    merged['news_text'] = ''

merged = merged.dropna(subset=['close', 'volume'])

print(f"✅ 对齐完成: {len(merged)} 个交易日")

# ------------------- 4. 特征工程 -------------------
print("\n📊 计算收益率和标签...")

merged = merged.sort_values('date').reset_index(drop=True)

# 避免除零：成交量为0时替换为1
merged['volume'] = merged['volume'].replace(0, 1)

# 计算收益率和成交量变化率
merged['return'] = merged['close'].pct_change()
merged['volume_change'] = merged['volume'].pct_change()
# 目标标签：下一交易日收益率（5日收益率可选，这里保持1日）
merged['label'] = merged['return'].shift(-1)

# 过滤掉所有 inf 和 nan（基础特征）
merged = merged.replace([np.inf, -np.inf], np.nan)
merged = merged.dropna(subset=['return', 'volume_change', 'label'])

# ------------------- 5. 计算技术指标 -------------------
print("\n📊 计算技术指标（MA5, MA20, RSI, MACD）...")

# 1. 移动平均线 MA5 / MA20
merged['ma5'] = merged['close'].rolling(window=5).mean()
merged['ma20'] = merged['close'].rolling(window=20).mean()

# 2. RSI（14天）
def calc_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

merged['rsi'] = calc_rsi(merged['close'], window=14)

# 3. MACD（12, 26, 9）
def calc_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line

merged['macd'] = calc_macd(merged['close'])

# 填充技术指标 NaN（前20天）
tech_cols = ['ma5', 'ma20', 'rsi', 'macd']
merged[tech_cols] = merged[tech_cols].ffill().fillna(0)

print(f"✅ 技术指标计算完成，共 {len(merged)} 个交易日")
print(f"✅ 处理后共 {len(merged)} 个有效交易日")

# ------------------- 6. 合并情绪因子 -------------------
print("\n📊 合并情绪因子...")
sentiment_path = os.path.join(RAW_DATA_DIR, "sentiment_indices.csv")
if os.path.exists(sentiment_path):
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=['date'])
    print(f"   情绪因子: {len(sentiment_df)} 条")
    # 合并到 merged（左连接，保留所有 merged 日期）
    merged = merged.merge(sentiment_df, on='date', how='left')
    # 填充缺失的情绪因子（2019年之前的数据可能没有情绪）
    sentiment_cols = ['daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']
    for col in sentiment_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().fillna(0)
        else:
            merged[col] = 0
    print(f"✅ 情绪因子合并完成，新增列: {[c for c in sentiment_cols if c in merged.columns]}")
else:
    print("   ⚠️ 未找到 sentiment_indices.csv，将创建全0情绪因子列")
    merged['daily_sentiment'] = 0
    merged['ewma_7d'] = 0
    merged['ewma_14d'] = 0
    merged['overnight_sentiment'] = 0

# ------------------- 7. 保存完整数据集 -------------------
output_full_path = os.path.join(RAW_DATA_DIR, "aligned_sentiment_data.csv")
merged.to_csv(output_full_path, index=False)
print(f"\n💾 完整数据已保存到: {output_full_path}")
print(f"   包含列: {merged.columns.tolist()}")

# ------------------- 8. 3-Fold 滚动窗口切分 -------------------
print("\n✂️ 生成 3-Fold 滚动窗口数据集...")

total_len = len(merged)

for i, (train_ratio, val_ratio, test_ratio) in enumerate(FOLD_CONFIGS, 1):
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * val_ratio)
    test_end = int(total_len * test_ratio)

    fold_dir = os.path.join(OUTPUT_DIR, f"fold{i}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = merged.iloc[:train_end].copy()
    val_df = merged.iloc[train_end:val_end].copy()
    test_df = merged.iloc[val_end:test_end].copy()

    train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

    print(f"\n   ✅ Fold {i}:")
    print(f"      训练: {train_df['date'].min()} ~ {train_df['date'].max()} ({len(train_df)}条)")
    print(f"      验证: {val_df['date'].min()} ~ {val_df['date'].max()} ({len(val_df)}条)")
    print(f"      测试: {test_df['date'].min()} ~ {test_df['date'].max()} ({len(test_df)}条)")

print("\n🎉 特征工程完成！")
print("👉 下一步：重新运行 train_sentiment_model.py")