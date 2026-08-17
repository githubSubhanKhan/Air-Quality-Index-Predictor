import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit Cloud secrets live in st.secrets, but feature_store.py (and dotenv)
# read from os.environ — bridge them before any DB-touching import runs.
# Locally there's no secrets.toml at all, which makes st.secrets raise, so
# this is a no-op there and feature_store.py's own load_dotenv() takes over.
try:
    for _key in ("MONGODB_URI", "MONGODB_DB_NAME"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from app.src.features.feature_store import get_collection
from app.src.prediction.build_prediction_features import build_prediction_features
from app.src.prediction.predictor import predict as predict_aqi

# "Very Unhealthy"/"Hazardous" extend the palette's 4-step status scale to
# match the 6-category EPA AQI standard users expect.
AQI_SCALE = [
    (0, 50, "Good", "#0ca30c"),
    (51, 100, "Moderate", "#fab219"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ec835a"),
    (151, 200, "Unhealthy", "#d03b3b"),
    (201, 300, "Very Unhealthy", "#4a3aa7"),
    (301, 500, "Hazardous", "#6b1414"),
]

THEME = dict(
    surface="#fcfcfb", card_bg="#f2f1ed",
    text_primary="#0b0b0b", text_secondary="#52514e", muted="#898781",
    gridline="#e1e0d9", border="rgba(11,11,11,0.10)", axis="#c3c2b7",
    line="#2a78d6", band_opacity=0.08,
)

POLLUTANT_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "o3": "Ozone (O3)",
    "no2": "Nitrogen Dioxide (NO2)",
    "so2": "Sulfur Dioxide (SO2)",
    "co": "Carbon Monoxide (CO)",
}
WEATHER_LABELS = {
    "temperature": ("Temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "pressure": ("Pressure", "hPa"),
    "wind_speed": ("Wind Speed", "m/s"),
}


def get_aqi_category(aqi):
    if aqi is None:
        return "Unknown", "#898781"
    for low, high, label, color in AQI_SCALE:
        if aqi <= high:
            return label, color
    return AQI_SCALE[-1][2], AQI_SCALE[-1][3]


@st.cache_data(ttl=120, show_spinner=False)
def fetch_cities():
    try:
        return sorted(get_collection().distinct("city"))
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_prediction(city):
    records = list(
        get_collection().find({"city": city.lower()}).sort("timestamp", 1)
    )

    if not records:
        return {"error": f"No data found for city '{city}'"}

    df = pd.DataFrame(records)
    features = build_prediction_features(df)
    forecast = predict_aqi(features)

    return {"city": city, "forecast": forecast}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(city, hours):
    records = list(
        get_collection()
        .find({"city": city.lower()}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(hours)
    )

    if not records:
        return {"error": f"No data found for city '{city}'"}

    records.reverse()
    return {"city": city, "count": len(records), "readings": records}


def inject_css(t):
    st.markdown(
        f"""
        <style>
        :root {{
            --surface: {t['surface']};
            --card-bg: {t['card_bg']};
            --text-primary: {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --muted: {t['muted']};
            --gridline: {t['gridline']};
            --border: {t['border']};
            --axis: {t['axis']};
        }}

        .block-container {{ padding-top: 2rem; max-width: 1200px; }}

        .aqi-hero {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem 2rem;
        }}
        .aqi-hero-value {{ font-size: 4rem; font-weight: 700; line-height: 1; color: var(--text-primary); }}
        .aqi-hero-label {{ font-size: 0.95rem; color: var(--text-secondary); margin-bottom: 0.35rem; }}
        .aqi-badge {{
            display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
            font-weight: 600; font-size: 0.9rem; color: #fcfcfb; margin-top: 0.6rem;
        }}

        .meter-track {{
            position: relative; height: 14px; border-radius: 7px; overflow: visible;
            margin-top: 1.25rem;
            background: linear-gradient(to right,
                #0ca30c 0% 10%, #fab219 10% 20%, #ec835a 20% 30%,
                #d03b3b 30% 40%, #4a3aa7 40% 60%, #6b1414 60% 100%);
        }}
        .meter-marker {{
            position: absolute; top: -5px; width: 4px; height: 24px;
            background: var(--text-primary); border-radius: 2px;
            box-shadow: 0 0 0 2px var(--surface);
        }}
        .meter-scale {{ display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); margin-top: 0.3rem; }}

        .forecast-card {{
            background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
            padding: 1.1rem 1rem; text-align: center;
        }}
        .forecast-day {{ font-size: 0.85rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }}
        .forecast-value {{ font-size: 2.2rem; font-weight: 700; margin: 0.3rem 0; color: var(--text-primary); }}

        .legend-row {{ display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.5rem; }}
        .legend-chip {{
            display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; color: var(--text-secondary);
            background: var(--card-bg); border-radius: 999px; padding: 0.25rem 0.7rem;
        }}
        .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(current_aqi):
    label, color = get_aqi_category(current_aqi)
    marker_pct = min(max((current_aqi or 0) / 500 * 100, 0), 100)

    st.markdown(
        f"""
        <div class="aqi-hero">
            <div class="aqi-hero-label">Current Air Quality Index</div>
            <div class="aqi-hero-value">{current_aqi:.0f}</div>
            <span class="aqi-badge" style="background:{color};">{label}</span>
            <div class="meter-track">
                <div class="meter-marker" style="left:{marker_pct}%;"></div>
            </div>
            <div class="meter-scale"><span>0</span><span>100</span><span>200</span><span>300</span><span>400</span><span>500</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_legend():
    chips = "".join(
        f'<div class="legend-chip"><span class="legend-dot" style="background:{color};"></span>{label} ({low}-{high})</div>'
        for low, high, label, color in AQI_SCALE
    )
    st.markdown(f'<div class="legend-row">{chips}</div>', unsafe_allow_html=True)


def render_forecast_cards(forecast):
    days = [("Day 1", forecast["day_1"]), ("Day 2", forecast["day_2"]), ("Day 3", forecast["day_3"])]
    cols = st.columns(3)
    for col, (day_label, value) in zip(cols, days):
        label, color = get_aqi_category(value)
        with col:
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-day">{day_label}</div>
                    <div class="forecast-value">{value:.0f}</div>
                    <span class="aqi-badge" style="background:{color};">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def add_aqi_bands(fig, y_max, t):
    for low, high, label, color in AQI_SCALE:
        if low > y_max:
            continue
        fig.add_hrect(
            y0=low, y1=min(high, y_max),
            fillcolor=color, opacity=t["band_opacity"], line_width=0,
            layer="below",
        )


def style_axes(fig, t):
    fig.update_layout(
        plot_bgcolor=t["surface"],
        paper_bgcolor=t["surface"],
        font=dict(color=t["text_primary"], family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], font_color=t["text_primary"], bordercolor=t["axis"]),
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"])
    fig.update_yaxes(showgrid=True, gridcolor=t["gridline"], zeroline=False)
    return fig


def render_forecast_chart(current_aqi, forecast, t):
    x = ["Now", "Day 1", "Day 2", "Day 3"]
    y = [current_aqi, forecast["day_1"], forecast["day_2"], forecast["day_3"]]
    y_max = max(y + [100]) * 1.15

    fig = go.Figure()
    add_aqi_bands(fig, y_max, t)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers+text",
        line=dict(color=t["line"], width=2),
        marker=dict(size=9, color=t["line"]),
        text=[f"{v:.0f}" for v in y],
        textposition="top center",
        name="AQI forecast",
        hovertemplate="%{x}: %{y:.0f} AQI<extra></extra>",
    ))
    fig.update_layout(title="3-Day AQI Forecast", yaxis_range=[0, y_max], showlegend=False, height=340)
    style_axes(fig, t)
    st.plotly_chart(fig, use_container_width=True)


def render_history_chart(readings, t):
    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["aqi"])

    if df.empty:
        st.info("No historical AQI readings available yet for this city.")
        return

    y_max = max(df["aqi"].max(), 100) * 1.1

    fig = go.Figure()
    add_aqi_bands(fig, y_max, t)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"], mode="lines",
        line=dict(color=t["line"], width=2),
        name="AQI",
        hovertemplate="%{x|%b %d, %H:%M}: %{y:.0f} AQI<extra></extra>",
    ))
    fig.update_layout(title="Historical AQI Trend", yaxis_range=[0, y_max], showlegend=False, height=360)
    style_axes(fig, t)
    st.plotly_chart(fig, use_container_width=True)


def render_pollutant_chart(latest_reading, t):
    keys = [k for k in POLLUTANT_LABELS if latest_reading.get(k) is not None]
    if not keys:
        st.info("No pollutant breakdown available for the latest reading.")
        return

    values = [latest_reading[k] for k in keys]
    labels = [POLLUTANT_LABELS[k] for k in keys]
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=t["line"],
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(title="Latest Pollutant Readings", height=320, margin=dict(l=10, r=40, t=30, b=10))
    fig.update_xaxes(showgrid=True, gridcolor=t["gridline"], zeroline=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(plot_bgcolor=t["surface"], paper_bgcolor=t["surface"], font=dict(color=t["text_primary"]))
    st.plotly_chart(fig, use_container_width=True)


def render_weather_metrics(latest_reading):
    cols = st.columns(len(WEATHER_LABELS))
    for col, (key, (label, unit)) in zip(cols, WEATHER_LABELS.items()):
        value = latest_reading.get(key)
        col.metric(label, f"{value:.1f} {unit}" if value is not None else "—")


def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

    with st.sidebar:
        st.title("🌫️ AQI Predictor")
        st.caption("3-day air quality forecasting dashboard")

        cities = fetch_cities()
        if cities:
            city = st.selectbox("City", options=cities, index=0)
        else:
            city = st.text_input("City", value="karachi")

        history_range = st.select_slider(
            "History range",
            options=["24h", "3 days", "7 days", "30 days"],
            value="7 days",
        )
        history_hours = {"24h": 24, "3 days": 72, "7 days": 168, "30 days": 720}[history_range]

        if st.button("🔄 Refresh now", use_container_width=True):
            fetch_prediction.clear()
            fetch_history.clear()
            st.rerun()

        st.divider()
        st.caption("Model: XGBoost (day-wise, 3-horizon)")
        st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t = THEME
    inject_css(t)

    st.title("Air Quality Index Predictor")
    st.caption(f"Live forecast for **{city.title()}**")

    try:
        prediction = fetch_prediction(city)
    except Exception as exc:
        st.error(f"Could not generate a forecast for `{city}`: {exc}")
        st.stop()

    if "error" in prediction:
        st.warning(prediction["error"])
        st.stop()

    forecast = prediction["forecast"]
    current_aqi = forecast["current_aqi"]

    hero_col, weather_col = st.columns([1.3, 1])
    with hero_col:
        render_hero(current_aqi)
        render_legend()

    try:
        history = fetch_history(city, history_hours)
    except Exception:
        history = {"error": "unreachable"}

    latest_reading = None
    if "readings" in history and history["readings"]:
        latest_reading = history["readings"][-1]

    with weather_col:
        st.markdown("**Latest conditions**")
        if latest_reading:
            render_weather_metrics(latest_reading)
        else:
            st.info("No weather readings available yet.")

    st.subheader("3-Day Forecast")
    render_forecast_cards(forecast)
    render_forecast_chart(current_aqi, forecast, t)

    st.subheader("History & Pollutants")
    hist_col, poll_col = st.columns([1.4, 1])
    with hist_col:
        if "readings" in history and history["readings"]:
            render_history_chart(history["readings"], t)
        else:
            st.info("No historical data available yet for this city.")
    with poll_col:
        if latest_reading:
            render_pollutant_chart(latest_reading, t)
        else:
            st.info("No pollutant data available.")

    st.divider()
    st.caption(
        "Forecast accuracy degrades with horizon — Day 1 is the most reliable prediction, "
        "Day 3 should be treated as directional guidance only."
    )


if __name__ == "__main__":
    main()
