# backend/app/db/init_db.py
# =====================================================================
# 建表 + 从旧 JSON 文件一次性迁移预制演示数据到 SQLite
# 运行：python -m app.db.init_db   （在 backend/ 目录下）
# =====================================================================
import json
import os

from .database import Base, SessionLocal, engine
from ..models import Code, Experiment, Factor, Report
from ..services import factor_service

# backend/ 目录与项目根目录
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)


def _load_json(rel_path):
    path = os.path.join(_PROJECT_ROOT, rel_path)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def migrate():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1) 因子库：优先迁移 factor_library.json，否则用 seed_if_empty 种子化
        if db.query(Factor).count() == 0:
            data = _load_json("output/factor_library.json")
            if data:
                for fac in data.get("factors", []):
                    db.add(Factor(
                        name=fac.get("name", ""), type=fac.get("type", "未知"),
                        definition=fac.get("definition", ""), params=fac.get("params", {}),
                        source=fac.get("source", ""), metrics=fac.get("metrics", {}),
                        status=fac.get("status", "待验证"), version=fac.get("version", 1),
                        created=fac.get("created", ""), updated=fac.get("updated", ""),
                    ))
                db.commit()
            else:
                factor_service.seed_if_empty(db)

        # 2) 代码库
        if db.query(Code).count() == 0:
            data = _load_json("output/code_library.json")
            if data:
                for c in data.get("codes", []):
                    db.add(Code(
                        file=c.get("file", ""), md5=c.get("md5", ""),
                        content=c.get("content", ""), summary=c.get("summary", {}),
                        time=c.get("time", ""), source=c.get("source", "页面手动上传"),
                    ))
                db.commit()

        # 3) 研报 + 实验记忆
        if db.query(Report).count() == 0:
            data = _load_json("output/research_memory.json")
            if data:
                for r in data.get("reports", []):
                    db.add(Report(
                        time=r.get("time", ""), title=r.get("title", "未知"),
                        core_view=r.get("core_view", ""), key_points=r.get("key_points", []),
                        topics=r.get("topics", []), region=r.get("region", "全球"),
                        sentiment=r.get("sentiment", "中性"),
                        sentiment_score=r.get("sentiment_score", 0.0),
                        has_quant_factor=r.get("has_quant_factor", False),
                        source_file=r.get("source_file", ""), md5=r.get("md5", ""),
                        publisher=r.get("publisher", ""), publish_date=r.get("publish_date", ""),
                    ))
                for e in data.get("experiments", []):
                    db.add(Experiment(
                        time=e.get("time", ""), hypothesis=e.get("hypothesis", ""),
                        params=e.get("params", {}), result=e.get("result", {}),
                        conclusion=e.get("conclusion", ""), tags=e.get("tags", []),
                    ))
                db.commit()

        print("迁移完成：")
        print(f"  factors     = {db.query(Factor).count()}")
        print(f"  codes       = {db.query(Code).count()}")
        print(f"  reports     = {db.query(Report).count()}")
        print(f"  experiments = {db.query(Experiment).count()}")
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
