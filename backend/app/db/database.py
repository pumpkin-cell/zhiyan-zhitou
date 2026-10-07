# backend/app/db/database.py
# =====================================================================
# SQLite + SQLAlchemy 数据库连接与会话管理
# 单文件 SQLite，放在 backend/app.db；后续可平滑迁移到 PostgreSQL
# =====================================================================
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# backend/ 目录（main.py 所在层级的上一级即 backend/）
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_BACKEND_DIR, "app.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 在多线程（FastAPI）下需要
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 依赖：每个请求一个数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
