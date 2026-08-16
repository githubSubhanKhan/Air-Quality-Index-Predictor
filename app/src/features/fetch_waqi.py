import os

import requests
from dotenv import load_dotenv

load_dotenv()

WAQI_BASE_URL = "https://api.waqi.info/feed"


def fetch_city_data(city: str) -> dict:
    """Fetch raw current AQI/pollutant/weather data for a city from WAQI."""
    token = os.getenv("WAQI_API_KEY")
    if not token:
        raise RuntimeError("WAQI_API_KEY is not set in the environment")

    response = requests.get(
        f"{WAQI_BASE_URL}/{city}/",
        params={"token": token},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "ok":
        raise RuntimeError(f"WAQI API error for city '{city}': {payload}")

    data = payload["data"]

    print("WAQI Timestamp:", data["time"]["iso"])
    print("WAQI AQI:", data["aqi"])

    return payload["data"]
