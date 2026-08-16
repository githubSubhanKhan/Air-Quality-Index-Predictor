from datetime import datetime, timezone
import os
import pandas as pd

from app.src.features.aqi import calculate_aqi_from_pm25


def _previous_aqi(city: str, history_path: str):
    if not os.path.exists(history_path):
        return None

    history = pd.read_csv(history_path)

    city_history = history[history["city"] == city]

    if city_history.empty:
        return None

    return city_history.iloc[-1]["aqi"].item()


def build_openweather_feature_row(
    city: str,
    raw: dict,
    history_path: str,
):
    dt = datetime.fromtimestamp(
        raw["dt"],
        tz=timezone.utc,
    )

    components = raw["components"]

    pm25 = components.get("pm2_5")

    aqi = (
        calculate_aqi_from_pm25(pm25)
        if pm25 is not None
        else None
    )

    previous_aqi = _previous_aqi(
        city,
        history_path,
    )

    return {
        "city": city,
        "timestamp": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "aqi": aqi,
        "aqi_change_rate": (
            aqi - previous_aqi
            if previous_aqi is not None and aqi is not None
            else 0.0
        ),
        "pm25": pm25,
        "pm10": components.get("pm10"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        "temperature": None,
        "humidity": None,
        "pressure": None,
        "wind_speed": None,
    }