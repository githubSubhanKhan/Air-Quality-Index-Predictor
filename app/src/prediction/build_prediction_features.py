import pandas as pd

from app.src.features.feature_engineering import build_training_features


def build_prediction_features(df: pd.DataFrame):
    """
    Convert historical AQI records into
    a single prediction-ready feature row.
    """

    df = build_training_features(df)

    df = df.dropna()

    latest_row = df.iloc[-1:]

    return latest_row