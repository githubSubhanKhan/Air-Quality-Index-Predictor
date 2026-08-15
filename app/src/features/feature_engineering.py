import pandas as pd


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # AQI lag features
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_lag_6"] = df["aqi"].shift(6)
    df["aqi_lag_12"] = df["aqi"].shift(12)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # PM2.5 lag features
    df["pm25_lag_1"] = df["pm25"].shift(1)
    df["pm25_lag_6"] = df["pm25"].shift(6)
    df["pm25_lag_24"] = df["pm25"].shift(24)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # AQI rolling mean
    df["aqi_roll_mean_6"] = df["aqi"].rolling(6).mean()
    df["aqi_roll_mean_12"] = df["aqi"].rolling(12).mean()
    df["aqi_roll_mean_24"] = df["aqi"].rolling(24).mean()

    # AQI rolling std
    df["aqi_roll_std_24"] = df["aqi"].rolling(24).std()

    return df


def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = create_lag_features(df)
    df = create_rolling_features(df)

    return df