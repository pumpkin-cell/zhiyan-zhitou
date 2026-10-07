# backend/app/services/memory_service.py
# =====================================================================
# 研究记忆库业务逻辑（SQLite 持久化，替代原 research_memory.py 的 JSON 持久化）
# 含研报素材（Report）与实验结论（Experiment）两类
# =====================================================================
import re
from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Experiment, Report


# ---------- 研报素材 ----------
def record_report(db: Session, summary: dict) -> Report:
    rep = Report(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        title=summary.get("title", "未知"),
        core_view=summary.get("core_view", ""),
        key_points=summary.get("key_points", []),
        topics=summary.get("topics", []),
        region=summary.get("region", "全球"),
        sentiment=summary.get("sentiment", "中性"),
        sentiment_score=summary.get("sentiment_score", 0.0),
        has_quant_factor=summary.get("has_quant_factor", False),
        source_file=summary.get("source_file", ""),
        md5=summary.get("md5", ""),
        publisher=summary.get("publisher", ""),
        publish_date=summary.get("publish_date", ""),
    )
    db.add(rep)
    db.commit()
    db.refresh(rep)
    return rep


def get_reports(db: Session):
    return db.query(Report).all()


def find_report_by_title(db: Session, title: str, prefix_len: int = 10):
    key = (title or "").strip()[:prefix_len]
    if not key:
        return []
    rows = db.query(Report).all()
    return [r for r in rows if (r.title or "").strip()[:prefix_len] == key]


def find_report_by_file(db: Session, filename: str, md5: str = None):
    """上传前查重：优先 MD5 内容指纹，其次文件名归一化。"""
    if md5:
        hits = db.query(Report).filter(Report.md5 == md5).all()
        if hits:
            return hits
    if filename:
        def _norm(s):
            s = (s or "").lower()
            s = re.sub(r'\.pdf$', '', s)
            s = re.sub(r'[\s_\-—·、，,。.（）()【】\[\]《》]', '', s)
            return s
        n = _norm(filename)
        if n:
            rows = db.query(Report).all()
            out = []
            for r in rows:
                src = _norm(r.source_file)
                if src and (n in src or src in n):
                    out.append(r)
            return out
    return []


# ---------- 实验结论 ----------
def record(db: Session, hypothesis: str, params: dict, result: dict,
           conclusion: str, tags: list) -> Experiment:
    exp = Experiment(
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        hypothesis=hypothesis, params=params, result=result,
        conclusion=conclusion, tags=tags,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


def get_experiments(db: Session):
    return db.query(Experiment).all()


def find_similar(db: Session, keyword: str, top_k: int = 3):
    kw = keyword.lower()
    rows = db.query(Experiment).all()
    hits = []
    for e in rows:
        text = " ".join([str(e.hypothesis or ""), str(e.conclusion or ""),
                         " ".join(e.tags or [])]).lower()
        if kw in text:
            hits.append(e)
    return hits[:top_k]


def warn_if_explored(db: Session, keyword: str):
    hits = find_similar(db, keyword)
    neg_words = ["不可复现", "无效", "失效", "反向", "失败", "未跑赢", "不显著"]
    return [e for e in hits if any(k in str(e.conclusion or "") for k in neg_words)]
