from fastapi import APIRouter

from app.src.features.feature_store import get_collection

router = APIRouter(
    tags=["Meta"],
)

collection = get_collection()


@router.get("/cities")
def list_cities():
    """Return every city currently present in the feature store."""

    cities = collection.distinct("city")

    return {"cities": sorted(cities)}
