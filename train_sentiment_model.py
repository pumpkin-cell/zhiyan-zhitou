# train_sentiment_model.py
# 实验2：技术面 + 情绪因子（单任务 LSTM）
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config_ETF import OUTPUT_DIR
from backtest import backtest, print_backtest_report
import warnings
warnings.filterwarnings('ignore')

# ========== 随机种子 ==========
def set_seed(seed=85):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(85)

print("=" * 60)
print("🔬 实验2：技术面 + 情绪因子 LSTM（单任务）")
print("   特征: return + volume_change + ma5 + ma20 + rsi + macd")
print("        + daily_sentiment + ewma_7d + ewma_14d + overnight_sentiment")
print("   目的: 验证情绪因子在技术面基础上的增量贡献")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  设备: {device}")

# ========== 超参数 ==========
SEQ_LEN = 10
HIDDEN_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 30

# ========== 特征列 ==========
FEATURE_COLS = [
    'return', 'volume_change',
    'ma5', 'ma20', 'rsi', 'macd',
    'daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment',
]

# ========== 模型定义 ==========
class SentimentLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.gru = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

# ========== 数据准备 ==========
def create_sequences(df, seq_len=10):
    df = df.sort_values('date').reset_index(drop=True)
    # 确保所有特征列都存在
    available = [col for col in FEATURE_COLS if col in df.columns]
    if len(available) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available)
        print(f"   ⚠️ 缺失特征: {missing}, 将用 0 填充")
        for col in missing:
            df[col] = 0
    features = np.column_stack([df[col].values.astype(np.float32) for col in FEATURE_COLS])
    labels = df['label'].values.astype(np.float32)

    def normalize(x):
        if np.std(x) < 1e-10:
            return np.zeros_like(x)
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
    features_norm = normalize(features)

    X_seq, y = [], []
    for i in range(seq_len, len(features_norm)):
        seq_feat = features_norm[i-seq_len:i]
        if np.isnan(seq_feat).any() or np.isinf(seq_feat).any():
            continue
        if np.isnan(labels[i]) or np.isinf(labels[i]):
            continue
        X_seq.append(seq_feat)
        y.append(labels[i])
    return np.array(X_seq, dtype=np.float32), np.array(y, dtype=np.float32)

# ========== 加载数据 ==========
print("\n📂 加载数据...")
df = pd.read_csv("./data/raw/aligned_sentiment_data.csv", parse_dates=['date'])
df = df[df['date'] >= pd.Timestamp('2019-01-01')]
df = df.dropna(subset=['return', 'label'] + FEATURE_COLS)
print(f"总样本数: {len(df)}")

total_len = len(df)
fold_configs = [(0.6, 0.8, 1.0), (0.7, 0.9, 1.0), (0.8, 0.95, 1.0)]

# ========== 训练函数 ==========
def train_fold(fold_id, train_end_ratio, val_end_ratio, test_end_ratio):
    train_end = int(total_len * train_end_ratio)
    val_end = int(total_len * val_end_ratio)
    test_end = int(total_len * test_end_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:test_end].copy()

    X_train, y_train = create_sequences(train_df, SEQ_LEN)
    X_val, y_val = create_sequences(val_df, SEQ_LEN)
    X_test, y_test = create_sequences(test_df, SEQ_LEN)

    if len(y_train) == 0 or len(y_val) ==0 or len(y_test) == 0:
        print(f' 测试样本为0 跳过此fold')
        return None

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                              batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentLSTM(input_size=len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_path = os.path.join(OUTPUT_DIR, f"sentiment_fold{fold_id}.pth")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, label)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if train_loss == 0:
            continue

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, label in val_loader:
                seq, label = seq.to(device), label.to(device)
                pred = model(seq)
                loss = criterion(pred, label)
                if torch.isnan(loss):
                    continue
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ⏹️  Early stop at epoch {epoch+1}")
                break

    # 加载最佳模型
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
    else:
        print("  ⚠️ 未保存最佳模型，使用当前模型")


    # 预测
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, label in test_loader:
            seq = seq.to(device)
            pred = model(seq)
            # 🔥 修复：处理标量情况
            pred_np = pred.cpu().numpy()
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np])
            preds.extend(pred_np.tolist())

            label_np = label.cpu().numpy()
            if label_np.ndim == 0:
                label_np = np.array([label_np])
            trues.extend(label_np.tolist())

    if len(preds) == 0:
        print("  ⚠️ 预测结果为空，跳过此 Fold")
        return None

    preds = np.array(preds)
    trues = np.array(trues)

    if len(preds) == 0:
        return None


    # 回测
    result = backtest(preds, trues)
    print_backtest_report(result, f"Fold {fold_id} (实验2)")
    return result

# ========== 主程序 ==========
print("\n🚀 开始训练...")
all_results = []
for i, config in enumerate(fold_configs, 1):
    print(f"\n{'='*40}")
    print(f"Fold {i}")
    print('='*40)
    result = train_fold(i, config[0], config[1], config[2])
    if result:
        all_results.append(result)

if all_results:
    print(f"\n{'='*50}")
    print("📊 实验2 最终汇总（技术面 + 情绪因子）")
    print(f"平均夏普比率: {np.mean([r['sharpe_ratio'] for r in all_results]):.4f}")
    print(f"平均累计收益: {np.mean([r['total_return_pct'] for r in all_results]):.2f}%")
    print(f"平均胜率: {np.mean([r['win_rate_pct'] for r in all_results]):.2f}%")
    print(f"平均最大回撤: {np.mean([r['max_drawdown_pct'] for r in all_results]):.2f}%")
else:
    print("❌ 所有 Fold 训练失败")