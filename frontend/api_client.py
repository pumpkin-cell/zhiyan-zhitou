# frontend/api_client.py
# =====================================================================
# 前端唯一调用后端的地方：封装所有接口
# 双模式：
#   1) HTTP 模式（本地开发）：BACKEND_URL=http://localhost:8000
#   2) 直连模式（云端单进程兜底）：后端未启动时，直接 import backend
# =====================================================================
import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _try_http(path: str, method: str = "GET", payload: dict = None):
    try:
        if method == "POST":
            resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=60)
        else:
            resp = requests.get(f"{BACKEND_URL}{path}", timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _add_paths():
    import sys
    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _backend = os.path.join(_proj, "backend")
    for _p in (_proj, _backend):
        if _p not in sys.path:
            sys.path.insert(0, _p)


def _direct(method_name: str, *args):
    """直连兜底：优先走纯计算（不依赖数据库），必要时才 import database。"""
    _add_paths()

    # ---- 产品体检（纯计算，只读 CSV，不依赖数据库）----
    if method_name in ("list_products", "list_overview", "check_product"):
        from zhitou.core import product_check
        if method_name == "list_products":
            return [{"code": k, "name": v} for k, v in product_check.ETF_NAMES.items()]
        if method_name == "check_product":
            return product_check.check_product(args[0], args[1] if len(args) > 1 else "3y")
        return product_check.list_overview(args[0] if args else "3y")

    # ---- 因子复现（纯计算）----
    if method_name == "reproduce_batch":
        from zhitou.core import factor_compute
        return factor_compute.reproduce_all()

    # ---- 以下需要数据库（sqlalchemy）----
    from zhitou.db.database import SessionLocal
    db = SessionLocal()
    try:
        if method_name == "get_factors":
            from zhitou.services import factor_service
            factor_service.seed_if_empty(db)
            return [factor_service.to_dict(f) for f in factor_service.get_all(db)]
        if method_name == "register_factor":
            from zhitou.services import factor_service
            p = args[0]
            f = factor_service.register(db, p.get("name", ""), p.get("params"),
                                        p.get("metrics"), p.get("source", ""),
                                        p.get("definition", ""), p.get("ftype", ""))
            return factor_service.to_dict(f)
        if method_name == "get_codes":
            from zhitou.services import code_service
            return [code_service.to_dict(c) for c in code_service.get_all(db)]
        if method_name == "get_reports":
            from zhitou.services import memory_service
            return [_report(r) for r in memory_service.get_reports(db)]
        if method_name == "get_experiments":
            from zhitou.services import memory_service
            return [_exp(e) for e in memory_service.get_experiments(db)]
    finally:
        db.close()
    return None


def _report(r):
    return {"id": r.id, "time": r.time, "title": r.title, "core_view": r.core_view,
            "key_points": r.key_points, "topics": r.topics, "region": r.region,
            "sentiment": r.sentiment, "sentiment_score": r.sentiment_score,
            "has_quant_factor": r.has_quant_factor, "source_file": r.source_file,
            "md5": r.md5, "publisher": r.publisher, "publish_date": r.publish_date}


def _exp(e):
    return {"id": e.id, "time": e.time, "hypothesis": e.hypothesis, "params": e.params,
            "result": e.result, "conclusion": e.conclusion, "tags": e.tags}


# ---------- 产品体检（工行杯版核心） ----------
def list_products():
    r = _try_http("/api/product/list")
    return r if r is not None else _direct("list_products")


def list_overview(period: str = "3y"):
    r = _try_http(f"/api/product/overview?period={period}")
    return r if r is not None else _direct("list_overview", period)


def check_product(code: str, period: str = "3y"):
    r = _try_http(f"/api/product/check?code={code}&period={period}")
    return r if r is not None else _direct("check_product", code, period)


# ---------- 因子复现 ----------
def reproduce_batch():
    r = _try_http("/api/reproduce/batch")
    return r if r is not None else _direct("reproduce_batch")


# ---------- 因子库（国创赛量化功能） ----------
def get_factors():
    r = _try_http("/api/factors")
    return r if r is not None else _direct("get_factors")


def register_factor(payload: dict):
    r = _try_http("/api/factors", "POST", payload)
    return r if r is not None else _direct("register_factor", payload)


# ---------- 代码库 ----------
def get_codes():
    r = _try_http("/api/codes")
    return r if r is not None else _direct("get_codes")


# ---------- 研究记忆库 ----------
def get_reports():
    r = _try_http("/api/memory/reports")
    return r if r is not None else _direct("get_reports")


def get_experiments():
    r = _try_http("/api/memory/experiments")
    return r if r is not None else _direct("get_experiments")
