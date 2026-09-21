# score_all_fast.py
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import os
import sys

print("=" * 50)
print("步骤4：高速批量打分（batch_size=32，每5000条保存）")
print("=" * 50)

model_dir = "./finbert_finetuned_large/model"
tokenizer_dir = "./finbert_finetuned_large/tokenizer"

print("加载模型...")
tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"使用设备: {device}")

def get_batch_scores(texts, batch_size=32):
    """批量推理"""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        batch_scores = (-probs[:, 0] + probs[:, 2]).cpu().numpy()
        scores.extend(batch_scores)
    return np.array(scores)

# 读取全量新闻
print("\n读取全量新闻...")
news = pd.read_csv("./data/raw/news_all_full.csv", parse_dates=['datetime'])
total = len(news)
print(f"总新闻数: {total:,} 条")

# 断点续传
output_path = "./data/raw/news_with_sentiment.csv"
if os.path.exists(output_path):
    existing = pd.read_csv(output_path)
    if 'sentiment_score' in existing.columns:
        start_idx = len(existing)
        print(f"✅ 发现已有进度 {start_idx} 条，从第 {start_idx} 条继续")
        existing_scores = existing['sentiment_score'].tolist()
        # 填充已有分数到 news
        if start_idx > 0:
            news.loc[:start_idx-1, 'sentiment_score'] = existing_scores
    else:
        print("⚠️ 文件缺少 sentiment_score 列，从头开始")
        start_idx = 0
        existing_scores = []
else:
    start_idx = 0
    existing_scores = []

if 'sentiment_score' not in news.columns:
    news['sentiment_score'] = 0.0

CHUNK_SIZE = 5000  # 🔥 每 5000 条保存一次，减少 I/O 开销
all_scores = existing_scores.copy()
print(f"\n开始从第 {start_idx} 条处理（每批 32 条）...")

try:
    for chunk_start in range(start_idx, total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total)
        chunk_texts = news['content'].iloc[chunk_start:chunk_end].fillna('').astype(str).tolist()
        chunk_scores = get_batch_scores(chunk_texts, batch_size=32)
        all_scores.extend(chunk_scores)
        news.loc[chunk_start:chunk_end-1, 'sentiment_score'] = chunk_scores
        news.iloc[:chunk_end].to_csv(output_path, index=False)
        print(f"  已处理 {chunk_end:,}/{total:,} 条，进度 {chunk_end/total*100:.2f}%")

    print(f"\n✅ 全部完成！共 {len(news):,} 条，已保存到 {output_path}")

except KeyboardInterrupt:
    print("\n⚠️ 用户中断，保存进度...")
    if len(all_scores) > 0:
        news.iloc[:len(all_scores)].to_csv(output_path, index=False)
        print(f"💾 已保存 {len(all_scores)} 条")
    sys.exit(0)