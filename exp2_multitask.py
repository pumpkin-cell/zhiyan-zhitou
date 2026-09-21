# exp2_multitask.py
# 实验3：技术面 + 情绪因子 + 多任务学习
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
print("🔬 实验3：技术面 + 情绪因子 + 多任务学习 LSTM")
print("   特征: 同实验2（10个特征）")
print("   模型: 回归 + 分类 多任务共享编码器")
print("   目的: 验证多任务学习能否进一步提升预测能力")
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

FEATURE_COLS = [
    'return', 'volume_change',
    'ma5', 'ma20', 'rsi', 'macd',
    'daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment'
]

# ========== 多任务模型 ==========
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        # 回归头：预测收益率数值
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        # 分类头：预测涨跌方向（二分类）
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        reg_out = self.reg_head(h).squeeze()
        cls_out = self.cls_head(h)
        return reg_out, cls_out

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

    X_seq, y_reg, y_cls = [], [], []
    for i in range(seq_len, len(features_norm)):
        seq_feat = features_norm[i-seq_len:i]
        if np.isnan(seq_feat).any() or np.isinf(seq_feat).any():
            continue
        if np.isnan(labels[i]) or np.isinf(labels[i]):
            continue
        X_seq.append(seq_feat)
        y_reg.append(labels[i])
        y_cls.append(1 if labels[i] > 0 else 0)
    return (np.array(X_seq, dtype=np.float32),
            np.array(y_reg, dtype=np.float32),
            np.array(y_cls, dtype=np.int64))  # 🔥 关键：改为 np.int64

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

    X_train, y_reg_train, y_cls_train = create_sequences(train_df, SEQ_LEN)
    X_val, y_reg_val, y_cls_val = create_sequences(val_df, SEQ_LEN)
    X_test, y_reg_test, y_cls_test = create_sequences(test_df, SEQ_LEN)

    if len(y_reg_train) == 0 or len(y_reg_val) == 0:
        return None

    # 创建 DataLoader，确保 y_cls 是 long 类型
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_reg_train),
            torch.tensor(y_cls_train, dtype=torch.long)  # 🔥 显式 long
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_reg_val),
            torch.tensor(y_cls_val, dtype=torch.long)   # 🔥 显式 long
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test),
            torch.tensor(y_reg_test),
            torch.tensor(y_cls_test, dtype=torch.long)  # 🔥 显式 long
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = MultiTaskLSTM(input_size=len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_path = os.path.join(OUTPUT_DIR, f"exp3_fold{fold_id}.pth")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for seq, reg_label, cls_label in train_loader:
            seq, reg_label, cls_label = seq.to(device), reg_label.to(device), cls_label.to(device)
            optimizer.zero_grad()
            reg_pred, cls_pred = model(seq)
            reg_weight=100
            loss = reg_weight*reg_criterion(reg_pred, reg_label) + cls_criterion(cls_pred, cls_label)
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
            for seq, reg_label, cls_label in val_loader:
                seq, reg_label, cls_label = seq.to(device), reg_label.to(device), cls_label.to(device)
                reg_pred, cls_pred = model(seq)
                loss = reg_criterion(reg_pred, reg_label) + cls_criterion(cls_pred, cls_label)
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

    # 预测（只使用回归输出）
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, reg_label, _ in test_loader:
            seq = seq.to(device)
            reg_pred, _ = model(seq)
            # 处理最后一个 batch 只有 1 个样本时的标量情况
            pred_np = reg_pred.cpu().numpy()
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np])
            preds.extend(pred_np.tolist())
            label_np = reg_label.cpu().numpy()
            if label_np.ndim == 0:
                label_np = np.array([label_np])
            trues.extend(label_np.tolist())

    if len(preds) == 0:
        return None

    # 回测
    result = backtest(np.array(preds), np.array(trues))
    print_backtest_report(result, f"Fold {fold_id} (实验3)")
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
    print("📊 实验3 最终汇总（技术面 + 情绪因子 + 多任务）")
    print(f"平均夏普比率: {np.mean([r['sharpe_ratio'] for r in all_results]):.4f}")
    print(f"平均累计收益: {np.mean([r['total_return_pct'] for r in all_results]):.2f}%")
    print(f"平均胜率: {np.mean([r['win_rate_pct'] for r in all_results]):.2f}%")
    print(f"平均最大回撤: {np.mean([r['max_drawdown_pct'] for r in all_results]):.2f}%")
else:
    print("❌ 所有 Fold 训练失败")