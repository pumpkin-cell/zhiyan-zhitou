# backend/app/api/codes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..services import code_service

router = APIRouter(prefix="/api/codes", tags=["codes"])


@router.get("")
def list_codes(db: Session = Depends(get_db)):
    return [code_service.to_dict(c) for c in code_service.get_all(db)]


@router.post("")
def register_code(payload: dict, db: Session = Depends(get_db)):
    c = code_service.register(
        db,
        payload.get("file", ""),
        payload.get("content", ""),
        payload.get("summary", {}),
    )
    return code_service.to_dict(c)
