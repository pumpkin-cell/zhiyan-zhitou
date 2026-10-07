# backend/main.py
# =====================================================================
# 智研智投 · 后端 FastAPI 入口
# 运行：uvicorn main:app --reload --port 8000   （在 backend/ 目录下）
# =====================================================================
from fastapi import FastAPI

from app.db.database import Base, engine
from app import models  # noqa: F401  确保模型注册到 Base.metadata
from app.api.health import router as health_router
from app.api import codes, factors, memory, product, reproduce

# 建表（首次启动自动建表；数据迁移见 app/db/init_db.py）
Base.metadata.create_all(bind=engine)

app = FastAPI(title="智研智投 API", version="1.0.0")

app.include_router(health_router, tags=["health"])
app.include_router(factors.router)
app.include_router(codes.router)
app.include_router(memory.router)
app.include_router(reproduce.router)
app.include_router(product.router)


@app.get("/")
def root():
    return {"service": "zhiyan-zhitou", "docs": "/docs"}
