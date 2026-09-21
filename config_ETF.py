# config_ETF.py
STOCK_CODE = "510300"
STOCK_NAME = "沪深300ETF"
START_DATE = "2019-01-01"   # 根据实际新闻起始年份（快讯从2019年开始）
END_DATE = "2026-06-30"     # 截止到2026年6月30日
RAW_DATA_DIR = "./data/raw"
OUTPUT_DIR = "./data/splits"
FOLD_CONFIGS = [
    (0.6, 0.8, 1.0),   # Fold 1: 训练60%, 验证20%, 测试20%
    (0.7, 0.9, 1.0),   # Fold 2: 训练70%, 验证20%, 测试10%
    (0.8, 0.95, 1.0)   # Fold 3: 训练80%, 验证15%, 测试5%
]

USE_TECHNICAL_INDICATORS = True  # True=加入技术指标, False=纯价量+情绪