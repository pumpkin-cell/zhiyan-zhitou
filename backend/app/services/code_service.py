# backend/app/services/code_service.py
# =====================================================================
# 策略代码库业务逻辑（SQLite 持久化，替代原 code_library.py 的 JSON 持久化）
# =====================================================================
import hashlib
from datetime import datetime

from sqlalchemy.orm import Session

from ..models import Code


def register(db: Session, filename: str, content: str, summary: dict) -> Code:
    """入库一份代码（调用前建议先用 find_by_md5 去重）。"""
    md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
    rec = Code(
        file=filename, md5=md5, content=content, summary=summary,
        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source="页面手动上传",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_all(db: Session):
    return db.query(Code).all()


def find_by_md5(db: Session, md5: str):
    return db.query(Code).filter(Code.md5 == md5).all()


def to_dict(c: Code) -> dict:
    return {
        "id": c.id, "file": c.file, "md5": c.md5, "content": c.content,
        "summary": c.summary, "time": c.time, "source": c.source,
    }
