# backend/app/api/factors.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..services import factor_service

router = APIRouter(prefix="/api/factors", tags=["factors"])


@router.get("")
def list_factors(db: Session = Depends(get_db)):
    factor_service.seed_if_empty(db)
    return [factor_service.to_dict(f) for f in factor_service.get_all(db)]


@router.post("")
def register_factor(payload: dict, db: Session = Depends(get_db)):
    f = factor_service.register(
        db,
        payload.get("name", ""),
        payload.get("params"),
        payload.get("metrics"),
        payload.get("source", ""),
        payload.get("definition", ""),
        payload.get("ftype", ""),
    )
    return factor_service.to_dict(f)


@router.get("/warn/{name}")
def warn_failed(name: str, db: Session = Depends(get_db)):
    return [factor_service.to_dict(f) for f in factor_service.warn_if_failed(db, name)]
