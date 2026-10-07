# backend/app/api/reproduce.py
from fastapi import APIRouter

from ..core import factor_compute

router = APIRouter(prefix="/api/reproduce", tags=["reproduce"])


@router.get("/batch")
def reproduce_batch():
    """批量复现 6 个经典因子/研报结论。"""
    return factor_compute.reproduce_all()
