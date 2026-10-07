# backend/app/services/factor_service.py
# =====================================================================
# 因子库业务逻辑（SQLite 持久化，替代原 factor_library.py 的 JSON 持久化）
# 保留原有语义：同因子名去重、版本+1、状态判定、避坑查询、从 CSV 种子化
# =====================================================================
import csv
import math
from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Factor

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


def _judge_status(name: str, metrics: dict) -> str:
    """根据指标判定状态：有效 / 已证伪 / 待验证。"""
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


def seed_if_empty(db: Session, csv_path: str = "./data/factor_cards.csv") -> None:
    """首次建库：从 factor_cards.csv 导入评估因子 + 注册 6 个可复现因子元信息。"""
    if db.query(Factor).count() > 0:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    import os
    if os.path.exists(csv_path):
        try:
            with open(csv_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    name = str(row.get("factor", "")).strip()
                    if not name or name == "nan":
                        continue
                    ic = _to_float(row.get("ic"))
                    ftype = "另类" if any(k in name for k in
                                          ["sentiment", "ewma", "overnight", "market"]) else "时序"
                    db.add(Factor(
                        name=name, type=ftype, definition="", params={},
                        source="factor_cards.csv 自动评估",
                        metrics={"ic": ic, "icir": _to_float(row.get("icir")),
                                 "monotonic_ret": _to_float(row.get("monotonic_ret"))},
                        status="有效" if (ic is not None and abs(ic) >= 0.03) else "已证伪",
                        version=1, created=now, updated=now,
                    ))
        except Exception:
            pass

    for name, meta in FACTOR_META.items():
        db.add(Factor(
            name=name, type=meta["type"], definition=meta["definition"], params={},
            source=meta["source"], metrics={}, status="待验证",
            version=1, created=now, updated=now,
        ))
    db.commit()


def register(db: Session, name: str, params: dict, metrics: dict,
             source: str = "", definition: str = "", ftype: str = "") -> Factor:
    """注册/更新因子：同「因子名」去重，更新指标与参数，版本 +1。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = params or {}
    metrics = {k: _to_float(v) for k, v in (metrics or {}).items()}
    status = _judge_status(name, metrics)

    existing = db.query(Factor).filter(Factor.name == name).first()
    if existing:
        existing.params = params
        merged = dict(existing.metrics or {})
        merged.update(metrics)
        existing.metrics = merged
        existing.status = status
        existing.version = (existing.version or 1) + 1
        existing.updated = now
        if source:
            existing.source = source
        db.commit()
        db.refresh(existing)
        return existing

    meta = FACTOR_META.get(name, {})
    new = Factor(
        name=name, type=ftype or meta.get("type", "未知"),
        definition=definition or meta.get("definition", ""),
        params=params, source=source or meta.get("source", ""),
        metrics=metrics, status=status, version=1, created=now, updated=now,
    )
    db.add(new)
    db.commit()
    db.refresh(new)
    return new


def get_all(db: Session):
    return db.query(Factor).all()


def find(db: Session, name: str):
    return db.query(Factor).filter(Factor.name == name).all()


def warn_if_failed(db: Session, name: str):
    """避坑：命中同名且「已证伪」的因子（提示无需重复探索）。"""
    return db.query(Factor).filter(Factor.name == name, Factor.status == "已证伪").all()


def to_dict(f: Factor) -> dict:
    return {
        "id": f.id, "name": f.name, "type": f.type, "definition": f.definition,
        "params": f.params, "source": f.source, "metrics": f.metrics,
        "status": f.status, "version": f.version, "created": f.created, "updated": f.updated,
    }
