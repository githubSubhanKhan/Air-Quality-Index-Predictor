import pandas as pd

from app.src.features.feature_engineering import (
    build_training_features,
    is_history_column,
)


def build_prediction_features(df: pd.DataFrame):
    """
    Convert historical AQI records into
    a single prediction-ready feature row.
    """

    df = build_training_features(df)

    # The raw history block is deliberately excluded from the completeness
    # check. Its columns go back a week hour by hour and are not gap-tolerant,
    # so an outage longer than the ffill limit would leave NaNs in them and a
    # blanket dropna() would discard the newest row — quietly serving a stale
    # forecast. The series models handle their own missing hours; every other
    # column still has to be present, including `city`, whose absence is how a
    # row invented by the hourly grid is told apart from a real reading.
    required = [column for column in df.columns if not is_history_column(column)]

    df = df.dropna(subset=required)

    latest_row = df.iloc[-1:]

    return latest_row
