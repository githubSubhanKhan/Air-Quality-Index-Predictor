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
