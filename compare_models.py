# compare_models.py
# =====================================================================
# 命题1 支撑模块：多算法对比（同一特征集，预测未来1日收益方向）
#   模型：逻辑回归 / 随机森林 / 梯度提升树(XGBoost优先,sklearn兜底) / 神经网络MLP
#   输出：AUC / 准确率 / IC / 特征重要性（反哺因子卡片）
# 说明：A股日频信噪比极低，方向预测 AUC 接近 0.5 属正常，重点看特征重要性排序。
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


FEATURES = ['return', 'volume_change', 'ma5', 'ma20', 'rsi', 'macd',
            'daily_sentiment', 'ewma_7d', 'ewma_14d', 'overnight_sentiment']


def load():
    df = pd.read_csv("./data/raw/aligned_sentiment_data.csv", parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def build_xy(df):
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_dir = (df['label'] > 0).astype(int)          # 分类标签：未来1日涨/跌
    y_ret = df['label'].values                     # 回归标签：未来1日收益（算IC用）
    return X, y_dir, y_ret


def ic(y_true, y_pred):
    """预测值与实际收益的 Spearman 秩相关（IC）。"""
    return pd.Series(y_pred).corr(pd.Series(y_true), method='spearman')


def main():
    df = load()
    X, y_dir, y_ret = build_xy(df)
    n = len(df)
    split = int(n * 0.7)                 # 前70%训练，后30%测试（时序切分，无未来函数）

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y_dir[:split], y_dir[split:]
    yret_te = y_ret[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = [
        ("逻辑回归 LogisticRegression", LogisticRegression(max_iter=1000), X_tr_s, X_te_s),
        ("随机森林 RandomForest", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1), X_tr, X_te),
    ]
    if HAS_XGB:
        models.append(("XGBoost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1), X_tr, X_te))
    else:
        models.append(("梯度提升树 GradientBoosting", GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42), X_tr, X_te))
    models.append(("神经网络 MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42), X_tr_s, X_te_s))

    print("=" * 78)
    print("多算法对比：预测沪深300ETF 未来1日涨跌方向（时序切分 70/30）")
    print(f"样本 {n} | 训练 {split} | 测试 {n - split} | 特征 {len(FEATURES)} 个")
    print("=" * 78)
    print(f"{'模型':<36}{'AUC':>8}{'准确率%':>9}{'IC':>8}")
    print("-" * 78)

    results = {}
    for name, model, Xtr, Xte in models:
        model.fit(Xtr, y_tr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba > 0.5).astype(int)
        auc = roc_auc_score(y_te, proba)
        acc = accuracy_score(y_te, pred)
        ic_val = ic(yret_te, proba)
        results[name] = (auc, acc, ic_val, model)
        print(f"{name:<36}{auc:>8.4f}{acc*100:>8.2f}%{ic_val:>8.4f}")

    # 基准：始终预测上涨
    base_acc = (y_te == 1).mean()
    print(f"\n  基准(恒多)准确率: {base_acc*100:.2f}%   → AUC 越接近 0.5 越说明日频方向难以预测")

    # ---- 特征重要性（树模型） ----
    print("\n" + "=" * 78)
    print("特征重要性（树模型 feature importance，反哺因子卡片）")
    print("=" * 78)
    for name in results:
        model = results[name][3]
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
            print(f"\n  [{name}]")
            for k, v in imp.items():
                print(f"    {k:<20} {v:.4f}")
            break

    print("\n说明：若 AUC≈0.5、IC≈0，说明单一价量+情绪特征在日频上无稳定预测力，")
    print("      这正印证命题1的痛点——需靠『研报复现+因子机制+多层次验证』而非暴力挖因子。")


if __name__ == '__main__':
    main()
