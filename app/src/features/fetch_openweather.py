import os

import requests
from dotenv import load_dotenv

load_dotenv()

GEOCODING_URL = "https://api.openweathermap.org/geo/1.0/direct"
AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

CURRENT_AIR_POLLUTION_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution"
)


def _api_key() -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set in the environment")
    return api_key


def geocode_city(city: str) -> tuple[float, float]:
    """Resolve a city name to (lat, lon) using OpenWeather's geocoding API."""
    response = requests.get(
        GEOCODING_URL,
        params={"q": city, "limit": 1, "appid": _api_key()},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()

    if not results:
        raise RuntimeError(f"Could not geocode city '{city}'")

    return results[0]["lat"], results[0]["lon"]


def fetch_historical_pollution(lat: float, lon: float, start: int, end: int) -> list[dict]:
    """Fetch hourly historical pollutant readings between unix timestamps start and end."""
    response = requests.get(
        AIR_POLLUTION_HISTORY_URL,
        params={"lat": lat, "lon": lon, "start": start, "end": end, "appid": _api_key()},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()

    return payload.get("list", [])

def fetch_current_pollution(city: str) -> dict:
    """
    Fetch current pollution data for a city.
    """

    lat, lon = geocode_city(city)

    response = requests.get(
        CURRENT_AIR_POLLUTION_URL,
        params={
            "lat": lat,
            "lon": lon,
            "appid": _api_key(),
        },
        timeout=10,
    )

    response.raise_for_status()

    payload = response.json()

    return payload["list"][0]
