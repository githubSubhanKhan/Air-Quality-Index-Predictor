import argparse
import os

import pandas as pd

from app.src.features.build_features import build_feature_row
from app.src.features.fetch_waqi import fetch_city_data

# Temporary local store until the feature store (Hopsworks) is wired up in Phase 3
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "features.csv")


def run(city: str) -> dict:
    raw = fetch_city_data(city)
    history_path = os.path.abspath(HISTORY_PATH)
    row = build_feature_row(city, raw, history_path)

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    pd.DataFrame([row]).to_csv(
        history_path,
        mode="a",
        header=not os.path.exists(history_path),
        index=False,
    )

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and build AQI features for a city")
    parser.add_argument("--city", required=True, help="City name as recognized by WAQI, e.g. 'karachi' or 'beijing'")
    args = parser.parse_args()

    print(run(args.city))
