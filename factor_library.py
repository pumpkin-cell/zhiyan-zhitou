# factor_library.py
# =====================================================================
# 因子库（完整版）：可增长、可追溯、可避坑的因子知识库
#   - 注册/更新：同「因子名」去重，更新指标与参数，版本 +1
#   - 来源追踪：经典文献 / factor_cards 自动评估 / 用户指令复现
#   - 状态判定：有效 / 已证伪 / 待验证
#   - 避坑查询：复现前命中「已证伪」因子时提示，避免重复探索
# 持久化：./output/factor_library.json（首次运行自动从 factor_cards.csv 种子化）
# =====================================================================
import os
import json
import math
from datetime import datetime

LIBRARY_PATH = "./output/factor_library.json"

# 可复现因子的元信息（定义 + 经典来源），供种子化与复现入库使用
FACTOR_META = {
    "动量":   {"type": "截面", "sign": 1, "definition": "过去N日累计收益排序，做多赢家、做空输家",
               "source": "Jegadeesh-Titman 1993"},
    "反转":   {"type": "截面", "sign": 1, "definition": "短期输家反弹、赢家回落（与动量相反）",
               "source": "Lehmann 1990"},
    "波动率": {"type": "时序", "sign": -1, "definition": "低波动组合长期跑赢高波动（低波动异象）",
               "source": "Ang et al. 2006"},
    "情绪":   {"type": "另类", "sign": -1, "definition": "投资者情绪过热→未来收益回落（反向指标，风控/过滤型）",
               "source": "Baker-Wurgler 2006"},
    "均线":   {"type": "时序", "sign": 1, "definition": "价格站上N日均线持有、跌破空仓（趋势择时）",
               "source": "经典技术分析"},
    "RSI":    {"type": "时序", "sign": 0, "definition": "RSI 超卖买入、超买卖出（均值回归）",
               "source": "经典技术分析"},
}


def _to_float(v):
    """转 python float，NaN/None → None，保证可 JSON 序列化。"""
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


class FactorLibrary:
    """轻量因子库（JSON 持久化，无外部依赖）。"""

    def __init__(self, path=LIBRARY_PATH):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding='utf-8') as f:
                d = json.load(f)
                d.setdefault("factors", [])
                return d
        data = {"factors": seed_library()}
        self._save_data(data)
        return data

    def _save_data(self, data=None):
        data = data if data is not None else self.data
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save(self):
        self._save_data()

    def register(self, name, params, metrics, source="", definition="", ftype=""):
        """注册/更新因子：同「因子名」去重，更新指标与参数，版本 +1。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        params = params or {}
        metrics = {k: _to_float(v) for k, v in (metrics or {}).items()}
        status = self._judge_status(name, metrics)

        existing = None
        for fac in self.data["factors"]:
            if fac["name"] == name:
                existing = fac
                break

        if existing:
            existing["params"] = params
            existing.setdefault("metrics", {}).update(metrics)
            existing["status"] = status
            existing["version"] = existing.get("version", 1) + 1
            existing["updated"] = now
            if source:
                existing["source"] = source
            self.save()
            return existing

        meta = FACTOR_META.get(name, {})
        new = {
            "name": name,
            "type": ftype or meta.get("type", "未知"),
            "definition": definition or meta.get("definition", ""),
            "params": params,
            "source": source or meta.get("source", ""),
            "metrics": metrics,
            "status": status,
            "version": 1,
            "created": now,
            "updated": now,
        }
        self.data["factors"].append(new)
        self.save()
        return new

    @staticmethod
    def _judge_status(name, metrics):
        """根据指标判定状态：有效 / 已证伪 / 待验证。
        可复现因子按「预期方向」判定（sign：+1 预期正 IC / -1 预期负 IC / 0 双向）；
        csv 评估因子按 |IC| 判定预测力。"""
        ic = metrics.get("ic")
        sharpe = metrics.get("sharpe")
        annual = metrics.get("annual")
        sign = FACTOR_META.get(name, {}).get("sign", 0)
        if ic is not None:
            if sign == 0:
                return "有效" if abs(ic) >= 0.03 else "已证伪"
            if sign * ic >= 0.03:
                return "有效"
            if sign * ic <= -0.03:
                return "已证伪"
            return "待验证"
        if sharpe is not None:
            return "有效" if sharpe > 0.5 else "已证伪"
        if annual is not None:
            return "有效" if annual > 2 else "已证伪"
        return "待验证"

    def get_all(self):
        return self.data["factors"]

    def find(self, name):
        return [f for f in self.data["factors"] if f["name"] == name]

    def warn_if_failed(self, name):
        """避坑：命中同名且「已证伪」的因子（提示无需重复探索）。"""
        return [f for f in self.data["factors"]
                if f["name"] == name and f.get("status") == "已证伪"]


def seed_library():
    """首次建库：从 factor_cards.csv 导入评估因子 + 注册 6 个可复现因子元信息。"""
    import csv as _csv
    factors = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    csv_path = "./data/factor_cards.csv"
    if os.path.exists(csv_path):
        try:
            with open(csv_path, encoding='utf-8-sig') as f:
                for row in _csv.DictReader(f):
                    name = str(row.get("factor", "")).strip()
                    if not name or name == "nan":
                        continue
                    ic = _to_float(row.get("ic"))
                    ftype = "另类" if any(k in name for k in
                                          ["sentiment", "ewma", "overnight", "market"]) else "时序"
                    factors.append({
                        "name": name,
                        "type": ftype,
                        "definition": "",
                        "params": {},
                        "source": "factor_cards.csv 自动评估",
                        "metrics": {
                            "ic": ic,
                            "icir": _to_float(row.get("icir")),
                            "monotonic_ret": _to_float(row.get("monotonic_ret")),
                        },
                        "status": "有效" if (ic is not None and abs(ic) >= 0.03) else "已证伪",
                        "version": 1,
                        "created": now,
                        "updated": now,
                    })
        except Exception:
            pass

    for name, meta in FACTOR_META.items():
        factors.append({
            "name": name,
            "type": meta["type"],
            "definition": meta["definition"],
            "params": {},
            "source": meta["source"],
            "metrics": {},
            "status": "待验证",
            "version": 1,
            "created": now,
            "updated": now,
        })
    return factors


if __name__ == '__main__':
    lib = FactorLibrary()
    print(f"因子库已初始化，共 {len(lib.get_all())} 个因子，路径 {lib.path}")
    for f in lib.get_all():
        print(f"  · {f['name']:<20} 类型={f['type']:<4} 状态={f['status']:<4} "
              f"版本={f['version']} 来源={f['source']}")

