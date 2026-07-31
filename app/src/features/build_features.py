import os
from datetime import datetime

import pandas as pd

# WAQI reports individual pollutant/weather readings under "iaqi", keyed like this
IAQI_FIELDS = {
    "pm25": "pm25",
    "pm10": "pm10",
    "o3": "o3",
    "no2": "no2",
    "so2": "so2",
    "co": "co",
    "temperature": "t",
    "humidity": "h",
    "pressure": "p",
    "wind_speed": "w",
}


def _iaqi_value(iaqi: dict, key: str):
    entry = iaqi.get(key)
    return entry["v"] if entry else None


def _previous_aqi(city: str, history_path: str):
    if not os.path.exists(history_path):
        return None

    history = pd.read_csv(history_path)
    city_history = history[history["city"] == city]
    if city_history.empty:
        return None

    return city_history.iloc[-1]["aqi"].item()


def build_feature_row(city: str, raw: dict, history_path: str) -> dict:
    """Turn one raw WAQI reading into a model-ready feature row (inputs + target)."""
    timestamp = raw.get("time", {}).get("iso")
    dt = datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()

    iaqi = raw.get("iaqi", {})
    aqi = raw.get("aqi")
    previous_aqi = _previous_aqi(city, history_path)

    row = {
        "city": city,
        "timestamp": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "aqi": aqi,
        "aqi_change_rate": (aqi - previous_aqi) if previous_aqi is not None and aqi is not None else 0.0,
    }

    for feature_name, iaqi_key in IAQI_FIELDS.items():
        row[feature_name] = _iaqi_value(iaqi, iaqi_key)

    return row
