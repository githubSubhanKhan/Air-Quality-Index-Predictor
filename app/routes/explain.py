import pandas as pd
from fastapi import APIRouter, Query

from app.src.explain.explainer import DEFAULT_TOP_N, explain_prediction
from app.src.features.feature_store import get_collection
from app.src.prediction.build_prediction_features import (
    build_prediction_features,
)
from app.src.prediction.predictor import HORIZONS, model_info

router = APIRouter(
    prefix="/explain",
    tags=["Explainability"],
)

collection = get_collection()


# Declared before /{city} so the literal path wins the match.
@router.get("/global")
def global_explanation():
    """
    The global SHAP view — mean |SHAP| per feature — recorded for the model
    versions currently in production.
    """

    info = model_info()

    return {
        "source": info["source"],
        "horizons": {
            horizon: entry.get("explanations", {})
            for horizon, entry in info["horizons"].items()
        },
    }


@router.get("/{city}")
def explain_city(
    city: str,
    horizon: str = Query(
        None,
        description="Limit to one horizon: day1, day2 or day3",
    ),
    top: int = Query(DEFAULT_TOP_N, ge=1, le=30),
):
    """
    SHAP explanation of the current forecast for a city.

    Every contribution is in AQI points: `base_value` plus the sum of all
    contributions reconstructs the prediction, so the numbers can be checked
    against `/predict/{city}`.
    """

    if horizon is not None and horizon not in HORIZONS:
        return {
            "error": f"Unknown horizon '{horizon}'",
            "expected": list(HORIZONS),
        }

    records = list(
        collection
        .find({"city": city.lower()})
        .sort("timestamp", 1)
    )

    if not records:
        return {"error": f"No data found for city '{city}'"}

    features = build_prediction_features(pd.DataFrame(records))

    explanation = explain_prediction(
        features,
        horizons=[horizon] if horizon else None,
        top_n=top,
    )

    return {"city": city, **explanation}
