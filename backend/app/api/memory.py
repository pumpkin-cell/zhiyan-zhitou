# backend/app/api/memory.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..services import memory_service

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _report_to_dict(r) -> dict:
    return {
        "id": r.id, "time": r.time, "title": r.title, "core_view": r.core_view,
        "key_points": r.key_points, "topics": r.topics, "region": r.region,
        "sentiment": r.sentiment, "sentiment_score": r.sentiment_score,
        "has_quant_factor": r.has_quant_factor, "source_file": r.source_file,
        "md5": r.md5, "publisher": r.publisher, "publish_date": r.publish_date,
    }


def _experiment_to_dict(e) -> dict:
    return {
        "id": e.id, "time": e.time, "hypothesis": e.hypothesis, "params": e.params,
        "result": e.result, "conclusion": e.conclusion, "tags": e.tags,
    }


@router.get("/reports")
def list_reports(db: Session = Depends(get_db)):
    return [_report_to_dict(r) for r in memory_service.get_reports(db)]


@router.get("/experiments")
def list_experiments(db: Session = Depends(get_db)):
    return [_experiment_to_dict(e) for e in memory_service.get_experiments(db)]
