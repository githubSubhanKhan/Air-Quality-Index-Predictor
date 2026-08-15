from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

day1_model = joblib.load(MODEL_DIR / "xgboost_day1.pkl")
day2_model = joblib.load(MODEL_DIR / "xgboost_day2.pkl")
day3_model = joblib.load(MODEL_DIR / "xgboost_day3.pkl")

FEATURES = joblib.load(MODEL_DIR / "feature_columns.pkl")


def predict(features_df):
    """
    Predict AQI for next 3 days.
    """

    X = features_df[FEATURES]

    day1 = float(day1_model.predict(X)[0])
    day2 = float(day2_model.predict(X)[0])
    day3 = float(day3_model.predict(X)[0])

    return {
        "current_aqi": round(float(features_df["aqi"].iloc[0]), 2),
        "day_1": round(day1, 2),
        "day_2": round(day2, 2),
        "day_3": round(day3, 2),
    }