from fastapi import APIRouter, Query

from app.src.prediction.predictor import model_info
from app.src.registry import model_registry as registry

router = APIRouter(
    prefix="/models",
    tags=["Models"],
)

# Read-only on purpose: promotion and rollback change what production serves,
# so they stay in the registry CLI rather than on an unauthenticated endpoint.


@router.get("/")
def production_models():
    """
    The model versions currently serving predictions, with their metrics.
    """

    return model_info()


@router.get("/versions")
def model_versions(
    name: str = None,
    limit: int = Query(20, ge=1, le=200),
):
    """
    Version history from the model registry, newest first.

    Pass `name` (e.g. `aqi_xgboost_day3`) to follow a single horizon's
    metrics over time.
    """

    documents = registry.list_versions(name=name, limit=limit)

    return {
        "count": len(documents),
        "versions": [registry.summarise(doc) for doc in documents],
    }


@router.get("/names")
def model_names():
    """Every model name known to the registry."""

    return {"names": registry.list_names()}
