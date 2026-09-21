# step3_finetune_finbert.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 🔥 强制离线模式，禁止联网检查

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

print("="*50)
print("步骤3：微调 FinBERT2-Large（离线模式）")
print("="*50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 读取标注数据
df = pd.read_csv("./data/raw/labeled_data.csv")
print(f"总标注样本: {len(df)}")
print("标签分布:")
print(df['sentiment'].value_counts())

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(label_map)
df = df.dropna(subset=['label'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['content'].tolist(),
    df['label'].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df['label']
)

print(f"训练集: {len(train_texts)} 条")
print(f"验证集: {len(val_texts)} 条")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 🔥 手动指定本地缓存目录（根据你的实际路径调整）
CACHE_DIR = "C:/Users/baylor/.cache/huggingface/hub/models--valuesimplex-ai-lab--FinBERT2-large/snapshots"
# 但更好的做法是直接使用模型名 + local_files_only=True
model_name = "valuesimplex-ai-lab/FinBERT2-large"

print(f"加载模型: {model_name} (离线模式)")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=3)
except Exception as e:
    print(f"❌ 从缓存加载失败，尝试指定具体路径...")
    # 如果上面失败，尝试直接指定缓存目录下的具体模型文件夹
    # 你需要找到实际的 snapshot 文件夹名，或者直接指定模型所在目录
    # 这里我们尝试使用缓存父目录
    cache_parent = "C:/Users/baylor/.cache/huggingface/hub/models--valuesimplex-ai-lab--FinBERT2-large"
    # 找到 snapshots 下的具体子目录（如果有多个，选最新的）
    import glob
    snapshots = glob.glob(os.path.join(cache_parent, "snapshots", "*"))
    if snapshots:
        model_path = snapshots[0]  # 取第一个
        print(f"使用缓存路径: {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True, num_labels=3)
    else:
        raise RuntimeError("无法找到本地模型缓存，请检查路径或重新下载。")

model.to(device)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

training_args = TrainingArguments(
    output_dir="./finbert_finetuned_large",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=400,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("开始微调 FinBERT2-Large...")
trainer.train()

model.save_pretrained("./finbert_finetuned_large/model")
tokenizer.save_pretrained("./finbert_finetuned_large/tokenizer")
print("✅ 微调完成，模型保存在 ./finbert_finetuned_large/")