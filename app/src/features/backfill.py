import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.src.features.aqi import calculate_aqi_from_pm25
from app.src.features.feature_store import insert_feature_row
from app.src.features.fetch_openweather import fetch_historical_pollution, geocode_city
from app.src.features.pipeline import HISTORY_PATH


def _build_row(city: str, record: dict, previous_aqi) -> dict:
    dt = datetime.fromtimestamp(record["dt"], tz=timezone.utc)
    components = record.get("components", {})
    pm25 = components.get("pm2_5")
    aqi = calculate_aqi_from_pm25(pm25) if pm25 is not None else None

    return {
        "city": city,
        "timestamp": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "day_of_week": dt.weekday(),
        "aqi": aqi,
        "aqi_change_rate": (aqi - previous_aqi) if aqi is not None and previous_aqi is not None else 0.0,
        "pm25": pm25,
        "pm10": components.get("pm10"),
        "o3": components.get("o3"),
        "no2": components.get("no2"),
        "so2": components.get("so2"),
        "co": components.get("co"),
        # OpenWeather's historical air pollution endpoint has no weather data (free tier
        # has no historical weather API); these stay unset for backfilled rows.
        "temperature": None,
        "humidity": None,
        "pressure": None,
        "wind_speed": None,
    }


def run(city: str, days: int) -> int:
    lat, lon = geocode_city(city)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    records = fetch_historical_pollution(lat, lon, int(start.timestamp()), int(end.timestamp()))
    records.sort(key=lambda r: r["dt"])

    previous_aqi = None
    rows = []
    for record in records:
        row = _build_row(city, record, previous_aqi)
        previous_aqi = row["aqi"]
        rows.append(row)
        insert_feature_row(row)

    if rows:
        history_path = os.path.abspath(HISTORY_PATH)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(
            history_path,
            mode="a",
            header=not os.path.exists(history_path),
            index=False,
        )

    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical AQI features for a city")
    parser.add_argument("--city", required=True, help="City name, e.g. 'karachi'")
    parser.add_argument("--days", type=int, default=30, help="Number of past days to backfill (default: 30)")
    args = parser.parse_args()

    count = run(args.city, args.days)
    print(f"Inserted {count} historical feature rows for '{args.city}'")
