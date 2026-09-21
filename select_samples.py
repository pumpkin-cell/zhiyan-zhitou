# step1_select_samples.py
import pandas as pd
import numpy as np
import os

print("="*50)
print("步骤1：从全量新闻抽取约1.5万条标注样本")
print("="*50)

# 读取全量新闻（354万条）
news = pd.read_csv("./data/raw/news_all_full.csv", parse_dates=['datetime'])
print(f"总新闻数: {len(news)}")

positive_keywords = ['利好', '大涨', '突破', '超预期', '增持', '看好', '涨停', '利润大增']
negative_keywords = ['利空', '大跌', '跌破', '不及预期', '减持', '看空', '跌停', '亏损']

news['has_pos'] = news['content'].str.contains('|'.join(positive_keywords), na=False, case=False)
news['has_neg'] = news['content'].str.contains('|'.join(negative_keywords), na=False, case=False)

POS_TARGET = 300
NEG_TARGET = 300
NEU_TARGET = 900

sampled = []
for year in range(2019, 2027):
    year_data = news[news['datetime'].dt.year == year]
    if len(year_data) == 0:
        continue
    pos_data = year_data[year_data['has_pos']]
    pos_sample = pos_data.sample(n=min(POS_TARGET, len(pos_data)), random_state=42) if len(pos_data) > 0 else pd.DataFrame()
    neg_data = year_data[year_data['has_neg']]
    neg_sample = neg_data.sample(n=min(NEG_TARGET, len(neg_data)), random_state=42) if len(neg_data) > 0 else pd.DataFrame()
    neutral_data = year_data[~(year_data['has_pos'] | year_data['has_neg'])]
    neutral_sample = neutral_data.sample(n=min(NEU_TARGET, len(neutral_data)), random_state=42) if len(neutral_data) > 0 else pd.DataFrame()
    sampled.append(pd.concat([pos_sample, neg_sample, neutral_sample]))
    print(f"   {year}年: 正{len(pos_sample)} 负{len(neg_sample)} 中{len(neutral_sample)}")

final_sample = pd.concat(sampled).drop_duplicates(subset=['content'])
if len(final_sample) > 15000:
    final_sample = final_sample.sample(n=15000, random_state=42)
final_sample = final_sample[['datetime', 'content']]
final_sample.to_csv("./data/raw/sample_for_labeling.csv", index=False)
print(f"✅ 已选 {len(final_sample)} 条样本，保存到 sample_for_labeling.csv")