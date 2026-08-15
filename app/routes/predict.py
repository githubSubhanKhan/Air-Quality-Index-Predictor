from fastapi import APIRouter
from pymongo import MongoClient
import pandas as pd
import os

from app.src.prediction.predictor import predict
from app.src.prediction.build_prediction_features import (
    build_prediction_features,
)
from app.src.features.feature_store import get_collection

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)

collection = get_collection()

print("MONGODB_URI =", os.getenv("MONGODB_URI"))
print("MONGODB_DB_NAME =", os.getenv("MONGODB_DB_NAME"))

@router.get("/{city}")
def predict_city(city: str):

    cursor = (
        collection
        .find({"city": city.lower()})
        .sort("timestamp", 1)
    )

    records = list(cursor)

    if not records:
        return {
            "error": f"No data found for city '{city}'"
        }

    df = pd.DataFrame(records)

    features = build_prediction_features(df)

    forecast = predict(features)

    return {
        "city": city,
        "forecast": forecast,
    }