from fastapi import APIRouter, Query

from app.src.features.feature_store import get_collection

router = APIRouter(
    prefix="/history",
    tags=["History"],
)

collection = get_collection()


@router.get("/{city}")
def get_history(city: str, hours: int = Query(168, ge=1, le=8760)):
    """
    Return the most recent `hours` feature-store readings for a city,
    oldest first, for charting historical trends.
    """

    cursor = (
        collection
        .find({"city": city.lower()}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(hours)
    )

    records = list(cursor)

    if not records:
        return {"error": f"No data found for city '{city}'"}

    records.reverse()

    return {
        "city": city,
        "count": len(records),
        "readings": records,
    }
