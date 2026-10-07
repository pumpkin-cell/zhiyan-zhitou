# backend/app/api/product.py
from fastapi import APIRouter

from ..core import product_check

router = APIRouter(prefix="/api/product", tags=["product"])


@router.get("/check")
def check(code: str, period: str = "3y"):
    return product_check.check_product(code, period)


@router.get("/list")
def list_products():
    return [{"code": k, "name": v} for k, v in product_check.ETF_NAMES.items()]


@router.get("/overview")
def overview(period: str = "3y"):
    return product_check.list_overview(period)
