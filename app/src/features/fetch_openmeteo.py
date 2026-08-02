import requests

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
):
    """
    Fetch hourly historical weather from Open-Meteo.
    Returns a dictionary keyed by UNIX timestamp.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
        ],
        "timezone": "UTC",
    }

    response = requests.get(
        OPENMETEO_ARCHIVE_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    weather = {}

    for i, time in enumerate(hourly["time"]):

        from datetime import datetime, timezone

        timestamp = int(
            datetime.fromisoformat(time)
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

        weather[timestamp] = {
            "temperature": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "pressure": hourly["surface_pressure"][i],
            "wind_speed": hourly["wind_speed_10m"][i],
        }

    return weather