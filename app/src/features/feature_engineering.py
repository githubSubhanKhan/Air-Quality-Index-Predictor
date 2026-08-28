"""
Feature engineering shared by training and serving.

Two things matter here beyond building columns:

1. **The hourly grid.** Backfilled rows sit exactly on the hour while live
   pipeline rows land at whatever minute the reading was taken, and there are
   gaps in the history. ``normalise_hourly`` floors every timestamp and
   reindexes onto a complete hourly grid, so ``shift(24)`` really means "24
   hours ago" instead of "24 rows ago".

2. **Stationary features.** Karachi's pollutant levels drift a long way
   across a year (o3 fell 54% between the training and evaluation windows of
   the 2025-26 data). Absolute levels put the far horizons outside anything
   the trees saw in training, so the model set is built from *relative*
   features — deviations from trailing means — which stay in range.

3. **The raw history block.** The classical forecasters in
   ``app/src/training/statistical.py`` model the AQI *series*, not a
   feature-to-target mapping, so they need the recent hourly readings
   themselves rather than summaries of them. ``create_history_features``
   attaches them as ``aqi_hist_0`` (now) through ``aqi_hist_167`` (a week
   ago), which lets a series model be an ordinary estimator with its own
   feature list — the registry already stores one per horizon, so nothing in
   serving had to learn about a new kind of model.

Legacy columns (``day``, ``month``, raw pollutant levels, the original lag
and rolling set) are still produced, because model versions registered before
this change reference them and must keep serving after a rollback.
"""

import numpy as np
import pandas as pd

POLLUTANTS = ["pm25", "pm10", "o3", "no2", "so2", "co"]

WEATHER = ["temperature", "humidity", "pressure", "wind_speed"]

# Gaps up to this many hours are carried forward. Forward only — a later
# reading must never fill an earlier hour, or features would see the future.
FFILL_LIMIT = 6

# Hours of raw hourly AQI handed to the series models. One week: seven daily
# cycles is enough to estimate a 24-hour seasonal profile and a local trend,
# and it is short enough that the block adds no NaN rows the curated feature
# set did not already have (``aqi_roll_mean_168`` needs a comparable run-up),
# so the usable-row count — and therefore every model's metrics — is
# unaffected by turning it on.
HISTORY_WINDOW = 168

HISTORY_PREFIX = "aqi_hist_"


def history_columns(window: int = HISTORY_WINDOW) -> list:
    """
    The history block's column names, newest first.

    ``aqi_hist_0`` is the reading at the row's own timestamp, so a row's
    window in chronological order is this list reversed.
    """

    return [f"{HISTORY_PREFIX}{lag}" for lag in range(window)]


def is_history_column(name: str) -> bool:
    return str(name).startswith(HISTORY_PREFIX)


def _rolling(series: pd.Series, window: int, how: str = "mean") -> pd.Series:
    """
    Rolling statistic that tolerates the gaps left by a missing hour.

    Requiring a completely full window would throw away every row for a day
    after each outage; 70% of the window keeps them at negligible cost.
    """

    rolled = series.rolling(window, min_periods=max(2, int(window * 0.7)))

    return getattr(rolled, how)()


def normalise_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Put the readings on a complete, regular hourly grid."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")

    df = df.drop_duplicates(subset="timestamp", keep="last")

    df = df.set_index("timestamp").sort_index()

    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")

    df = df.reindex(grid)

    carried = ["aqi", *POLLUTANTS, *WEATHER]

    df[carried] = df[carried].ffill(limit=FFILL_LIMIT)

    # `city` is deliberately left unfilled: it stays NaN on rows the grid
    # invented, which is how serving tells a real reading from a filler row.

    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar features derived from the grid itself.

    The cyclical encodings are what the models use: raw ``day`` and ``month``
    are period identifiers rather than signals — with a single year of
    history, month 7 never appeared in training at all, so every split on it
    sent July down an untrained branch.
    """

    df = df.copy()

    index = df.index

    df["hour"] = index.hour
    df["day"] = index.day
    df["month"] = index.month
    df["day_of_week"] = index.dayofweek

    day_of_year = index.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in (1, 3, 6, 12, 24):
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

    for lag in (1, 6, 24):
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for window in (6, 12, 24, 48, 72, 168):
        df[f"aqi_roll_mean_{window}"] = _rolling(df["aqi"], window)

    df["aqi_roll_std_24"] = _rolling(df["aqi"], 24, "std")
    df["aqi_roll_std_72"] = _rolling(df["aqi"], 72, "std")

    return df


def create_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deviations and trends, which stay in range as levels drift."""

    df = df.copy()

    df["aqi_diff_6"] = df["aqi"] - df["aqi_lag_6"]
    df["aqi_diff_24"] = df["aqi"] - df["aqi_lag_24"]

    df["aqi_rel_24"] = df["aqi"] - df["aqi_roll_mean_24"]
    df["aqi_rel_72"] = df["aqi"] - df["aqi_roll_mean_72"]
    df["aqi_rel_168"] = df["aqi"] - df["aqi_roll_mean_168"]

    df["aqi_trend_24_72"] = df["aqi_roll_mean_24"] - df["aqi_roll_mean_72"]
    df["aqi_trend_24_168"] = df["aqi_roll_mean_24"] - df["aqi_roll_mean_168"]

    for column in POLLUTANTS + WEATHER:
        df[f"{column}_rel_24"] = df[column] - _rolling(df[column], 24)
        df[f"{column}_rel_168"] = df[column] - _rolling(df[column], 168)

    return df


def create_history_features(
    df: pd.DataFrame,
    window: int = HISTORY_WINDOW,
) -> pd.DataFrame:
    """
    The last ``window`` hours of AQI, one column per hour of lag.

    Built in one ``concat`` rather than ``window`` separate assignments, which
    would fragment the frame and warn about it. Gaps are left as NaN on
    purpose: how to bridge a missing hour is a modelling choice, so the
    forecaster makes it (see ``statistical._clean_window``) rather than having
    it silently baked into the feature.
    """

    shifted = [
        df["aqi"].shift(lag).rename(f"{HISTORY_PREFIX}{lag}")
        for lag in range(window)
    ]

    return pd.concat([df, *shifted], axis=1)


def build_training_features(
    df: pd.DataFrame,
    include_history: bool = True,
) -> pd.DataFrame:
    """
    The full feature frame, used by both training and serving.

    Returns a frame with ``timestamp`` back as a column, matching what the
    rest of the project expects. ``include_history=False`` skips the raw
    history block, for callers that only want the curated feature set.
    """

    df = normalise_hourly(df)

    df = create_calendar_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_stationary_features(df)

    if include_history:
        df = create_history_features(df)

    return df.reset_index(names="timestamp")
