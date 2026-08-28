"""
EPA AQI arithmetic and the category scale.

The category scale lives here rather than in the dashboard because more than
one thing needs it now: the dashboard colours its badges by it, and the email
alerts decide their subject line, severity and health advice from it. Two
copies of a public health scale drifting apart is exactly the kind of bug
nobody notices until the advice is wrong.
"""

from typing import NamedTuple

# US EPA breakpoints for PM2.5 (24-hr avg, ug/m3) -> AQI (0-500)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def calculate_aqi_from_pm25(pm25: float) -> int:
    """Convert a PM2.5 concentration (ug/m3) to a standard EPA AQI value (0-500)."""
    pm25 = max(0.0, pm25)

    for conc_low, conc_high, aqi_low, aqi_high in PM25_BREAKPOINTS:
        if pm25 <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (pm25 - conc_low) + aqi_low
            return round(aqi)

    # Above the top breakpoint: clamp to the max AQI
    return 500


class Category(NamedTuple):
    """One band of the six-category EPA AQI scale."""

    index: int
    low: int
    high: int
    label: str
    headline: str
    advice: str


# The standard six EPA categories, with the guidance each one carries.
AQI_CATEGORIES = (
    Category(
        0, 0, 50, "Good",
        "Air quality is satisfactory.",
        "No precautions needed — a good few days to be outdoors.",
    ),
    Category(
        1, 51, 100, "Moderate",
        "Air quality is acceptable.",
        "Unusually sensitive people may want to limit long periods of "
        "strenuous activity outdoors.",
    ),
    Category(
        2, 101, 150, "Unhealthy for Sensitive Groups",
        "Sensitive groups may feel effects.",
        "Children, older adults and anyone with asthma or a heart or lung "
        "condition should cut back on prolonged exertion outdoors.",
    ),
    Category(
        3, 151, 200, "Unhealthy",
        "Everyone may begin to feel effects.",
        "Avoid prolonged exertion outdoors. Sensitive groups should stay "
        "indoors where they can and keep windows shut.",
    ),
    Category(
        4, 201, 300, "Very Unhealthy",
        "Health alert — serious effects for everyone.",
        "Avoid all outdoor exertion. Stay indoors, keep windows closed and "
        "run an air purifier if you have one. Wear an N95 outdoors.",
    ),
    Category(
        5, 301, 500, "Hazardous",
        "Health emergency — the whole population is at risk.",
        "Stay indoors with windows and doors shut. Avoid going out at all; "
        "if you must, wear a properly fitted N95.",
    ),
)

# The default point at which a forecast is treated as worth alerting about.
# "Unhealthy for Sensitive Groups" rather than "Hazardous": waiting for 301
# would mean the alerts never fire in a city that sits in the 50-150 band, and
# 101 is where the EPA's own guidance first asks anyone to change behaviour.
DEFAULT_ALERT_THRESHOLD = 101


def categorise(aqi) -> Category:
    """The EPA category an AQI value falls in. Values above 500 clamp."""

    if aqi is None:
        raise ValueError("Cannot categorise a missing AQI value")

    value = float(aqi)

    for category in AQI_CATEGORIES:
        if value <= category.high:
            return category

    return AQI_CATEGORIES[-1]


def category_label(aqi) -> str:
    return categorise(aqi).label


def category_by_label(label: str) -> Category:
    for category in AQI_CATEGORIES:
        if category.label == label:
            return category

    raise KeyError(f"Unknown AQI category '{label}'")


def threshold_categories() -> tuple:
    """
    The categories a user may reasonably choose as an alert threshold.

    "Good" is excluded: an alert on every forecast is not an alert.
    """

    return AQI_CATEGORIES[1:]
