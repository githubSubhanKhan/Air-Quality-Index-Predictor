"""
The current 3-day forecast for a city, in one call.

Loading a city's rows, engineering the serving row, predicting and collecting
the model provenance is the same four steps everywhere it is needed. This is
that sequence in one place, so a second caller (the alert CLI) does not
re-implement it and drift.

``reading_time`` is returned deliberately. The forecast is anchored on the most
recent *complete* feature row, which is not necessarily the most recent
reading: the lag features are not gap-tolerant, so a missed run in the hourly
pipeline pushes the usable row backwards. Anything presenting a forecast should
be able to say what it was computed from.
"""

import pandas as pd

from app.src.features.feature_store import get_collection
from app.src.prediction.build_prediction_features import build_prediction_features
from app.src.prediction.predictor import model_info, predict


def forecast_for_city(city: str, include_models: bool = True) -> dict:
    """
    ``{city, forecast, reading_time, features, models}`` for one city.

    Raises ``LookupError`` when the feature store has nothing usable, which is
    a different problem from a model or a mail failure and worth telling apart.
    """

    key = city.strip().lower()

    records = list(
        get_collection()
        .find({"city": key}, {"_id": 0})
        .sort("timestamp", 1)
    )

    if not records:
        raise LookupError(f"No readings in the feature store for '{city}'.")

    features = build_prediction_features(pd.DataFrame(records))

    if features.empty:
        raise LookupError(
            f"No complete feature row for '{city}' — the recent history has "
            f"gaps the engineered features cannot be built from."
        )

    forecast = predict(features)

    reading_time = pd.to_datetime(
        features["timestamp"].iloc[0], utc=True
    ).to_pydatetime()

    models = None

    if include_models:
        try:
            models = model_info()["horizons"]

        except Exception as exc:
            # Provenance is a nice-to-have on a forecast, not a reason to
            # withhold one.
            print(f"Model provenance unavailable: {exc}")

    return {
        "city": key,
        "forecast": forecast,
        "reading_time": reading_time,
        "features": features,
        "models": models,
    }
