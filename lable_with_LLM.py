# step2_label_with_llm.py
import pandas as pd
import json
import time
from dashscope import Generation

import os

# API Key 从环境变量读取（阿里云百炼平台获取），勿硬编码提交
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("未设置环境变量 DASHSCOPE_API_KEY，请先设置后再运行")

def label_sentiment(text):
    """调用 Qwen 给单条新闻打标签（正面/中性/负面）"""
    prompt = f"""请判断以下财经新闻的情感倾向，输出格式为 JSON：{{"sentiment": "positive|neutral|negative", "confidence": 0.0-1.0, "reason": "简短理由"}}

新闻：{text[:300]}
"""

    messages = [{"role": "user", "content": prompt}]
    try:
        response = Generation.call(
            model='qwen-turbo',
            messages=messages,
            api_key=DASHSCOPE_API_KEY,
            result_format='message'
        )
        result = response.output.choices[0].message.content
        # 解析 JSON
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('sentiment', 'neutral'), data.get('confidence', 0.5)
        return 'neutral', 0.5
    except Exception as e:
        print(f"   ⚠️ 标注失败: {e}")
        return 'neutral', 0.5

print("📂 加载样本数据...")
df = pd.read_csv("./data/raw/sample_for_labeling.csv")

print("🏷️ 开始标注（每条约 1-2 秒，共 15000 条，预计 4-5 小时）...")
# ⚠️ 可先跑 100 条测试，确认无误后再跑全量
sentiments = []
for i, row in df.iterrows():
    if i % 100 == 0 and i > 0:
        print(f"  进度: {i}/{len(df)}")
    sentiment, confidence = label_sentiment(row['content'])
    sentiments.append({'sentiment': sentiment, 'confidence': confidence})
    time.sleep(0.2)  # 防止限流

df['sentiment'] = [s['sentiment'] for s in sentiments]
df['confidence'] = [s['confidence'] for s in sentiments]

# 只保留高置信度的样本
df = df[df['confidence'] > 0.7]
df.to_csv("./data/raw/labeled_data.csv", index=False)
print(f"✅ 标注完成，有效样本 {len(df)} 条")