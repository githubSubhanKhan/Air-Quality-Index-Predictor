"""
Composing the 3-day AQI alert email.

Kept apart from the sending in ``mailer.py`` so the wording and the severity
rules can be tested without an SMTP server anywhere near them — the part most
likely to be wrong is what the message *says*, not the socket.

Two decisions worth stating:

* **Every send reports the whole 3-day outlook**, and the threshold decides
  the *severity*, not whether there is anything to say. A mail that arrives
  only on bad days is indistinguishable from a mail system that has quietly
  broken.
* **The reading time is on the face of the message.** The forecast is anchored
  on the most recent complete feature row, which can lag the clock when the
  hourly pipeline misses runs. A forecast presented without saying what it was
  computed from invites the reader to assume it is current.
"""

from datetime import datetime, timedelta, timezone

from app.src.features.aqi import (
    DEFAULT_ALERT_THRESHOLD,
    categorise,
)

# Category colours for the HTML table. Kept here rather than imported from the
# dashboard: this is email styling, and the dashboard's palette is tuned for a
# screen it controls.
CATEGORY_COLOURS = {
    "Good": "#0ca30c",
    "Moderate": "#c98a00",
    "Unhealthy for Sensitive Groups": "#d4622a",
    "Unhealthy": "#d03b3b",
    "Very Unhealthy": "#4a3aa7",
    "Hazardous": "#6b1414",
}

HORIZON_KEYS = ("day_1", "day_2", "day_3")

FOOTER = (
    "You are receiving this because this address was entered on the AQI "
    "Predictor dashboard. Forecast accuracy falls off with distance: day 1 is "
    "the most reliable, day 3 is directional guidance only."
)


def _colour(label: str) -> str:
    return CATEGORY_COLOURS.get(label, "#52514e")


def forecast_days(forecast: dict, reading_time: datetime = None) -> list:
    """
    The three horizons as dated rows, worst-category information attached.

    Each horizon is the mean AQI over a 24-hour window measured from the
    reading the forecast was computed on, so the dates are derived from that
    reading rather than from "now".
    """

    if reading_time is None:
        reading_time = datetime.now(timezone.utc)

    rows = []

    for offset, key in enumerate(HORIZON_KEYS, start=1):
        value = forecast.get(key)

        if value is None:
            continue

        start = reading_time + timedelta(hours=(offset - 1) * 24)

        end = reading_time + timedelta(hours=offset * 24)

        category = categorise(value)

        rows.append({
            "horizon": key,
            "day": offset,
            "label": f"Day {offset}",
            "aqi": round(float(value), 1),
            "category": category,
            "window_start": start,
            "window_end": end,
            "window": (
                f"{start:%a %d %b %H:%M} - {end:%a %d %b %H:%M} UTC"
            ),
        })

    return rows


def build_alert(
    city: str,
    forecast: dict,
    reading_time: datetime = None,
    threshold: int = DEFAULT_ALERT_THRESHOLD,
    model_details: dict = None,
) -> dict:
    """
    Subject, plain-text body and HTML body for one alert.

    Returns a dict with ``subject``, ``text``, ``html``, ``breaches`` (the
    horizons at or above ``threshold``) and ``worst`` (the most severe
    category in the window), so the caller can log or display what was sent
    without re-deriving it.
    """

    if not forecast:
        raise ValueError("Cannot build an alert without a forecast")

    days = forecast_days(forecast, reading_time)

    if not days:
        raise ValueError("The forecast contained no day_1/2/3 values")

    current = forecast.get("current_aqi")

    current_category = categorise(current) if current is not None else None

    breaches = [day for day in days if day["aqi"] >= threshold]

    worst = max(days, key=lambda day: day["aqi"])

    threshold_category = categorise(threshold)

    title = city.strip().title()

    if breaches:
        first = breaches[0]

        subject = (
            f"AQI alert for {title}: {worst['category'].label} air forecast "
            f"({worst['aqi']:.0f} AQI on {worst['label']})"
        )

        headline = (
            f"{len(breaches)} of the next 3 days are forecast to reach "
            f"{threshold_category.label} or worse."
        )

        lead = (
            f"The first is {first['label']} "
            f"({first['window_start']:%a %d %b}) at {first['aqi']:.0f} AQI - "
            f"{first['category'].label}."
        )

    else:
        subject = (
            f"{title} AQI outlook: {worst['category'].label} "
            f"(peak {worst['aqi']:.0f} AQI over 3 days)"
        )

        headline = (
            f"No day in the next 3 is forecast to reach "
            f"{threshold_category.label} ({threshold}+ AQI)."
        )

        lead = (
            f"The worst is {worst['label']} at {worst['aqi']:.0f} AQI - "
            f"{worst['category'].label}."
        )

    return {
        "subject": subject,
        "text": _text_body(
            title, days, current, current_category, headline, lead,
            worst, threshold, threshold_category, reading_time, model_details,
        ),
        "html": _html_body(
            title, days, current, current_category, headline, lead,
            worst, threshold, threshold_category, reading_time, model_details,
        ),
        "breaches": [day["label"] for day in breaches],
        "worst": worst["category"].label,
        "worst_aqi": worst["aqi"],
        "threshold": threshold,
        "is_alert": bool(breaches),
    }


def _provenance(model_details: dict) -> list:
    """One line per horizon naming the model version behind it."""

    if not model_details:
        return []

    lines = []

    for horizon, entry in model_details.items():
        candidate = entry.get("candidate") or entry.get("model_type") or "model"

        version = entry.get("version")

        mae = (entry.get("metrics") or {}).get("mae")

        parts = [f"{horizon}: {candidate}"]

        if version:
            parts.append(f"v{version}")

        if mae is not None:
            parts.append(f"MAE {mae:.2f}")

        lines.append(" ".join(parts) if len(parts) == 1 else " · ".join(parts))

    return lines


def _text_body(
    title, days, current, current_category, headline, lead,
    worst, threshold, threshold_category, reading_time, model_details,
) -> str:
    lines = [
        f"{title} - 3-day air quality forecast",
        "",
        headline,
        lead,
        "",
    ]

    if current is not None:
        lines += [
            f"Right now: {current:.0f} AQI - {current_category.label}",
            "",
        ]

    lines.append("Forecast")

    for day in days:
        flag = "  <-- at or above your threshold" if day["aqi"] >= threshold else ""

        lines.append(
            f"  {day['label']}  {day['aqi']:>5.0f} AQI  "
            f"{day['category'].label}{flag}"
        )
        lines.append(f"           {day['window']}")

    lines += [
        "",
        f"What to do ({worst['category'].label} - the worst of the three days)",
        f"  {worst['category'].headline}",
        f"  {worst['category'].advice}",
        "",
        f"Alert threshold: {threshold}+ AQI ({threshold_category.label})",
    ]

    if reading_time is not None:
        lines.append(
            f"Computed from the reading at {reading_time:%Y-%m-%d %H:%M} UTC"
        )

    provenance = _provenance(model_details)

    if provenance:
        lines += ["", "Models"] + [f"  {line}" for line in provenance]

    lines += ["", "-" * 60, FOOTER]

    return "\n".join(lines)


def _html_body(
    title, days, current, current_category, headline, lead,
    worst, threshold, threshold_category, reading_time, model_details,
) -> str:
    """
    Inline-styled HTML.

    Email clients strip <style> blocks and most of CSS, so everything is an
    inline attribute on a table. Deliberately plain — it has to survive
    Gmail, Outlook and a phone.
    """

    accent = _colour(worst["category"].label)

    rows = []

    for day in days:
        colour = _colour(day["category"].label)

        breached = day["aqi"] >= threshold

        rows.append(
            f'<tr>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #e1e0d9;'
            f'font-weight:600;color:#0b0b0b;">{day["label"]}'
            + (
                '<span style="color:#d03b3b;font-weight:700;"> &#9888;</span>'
                if breached else ""
            )
            + f'<div style="font-weight:400;font-size:12px;color:#898781;">'
            f'{day["window"]}</div></td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #e1e0d9;'
            f'font-size:22px;font-weight:700;color:#0b0b0b;text-align:right;">'
            f'{day["aqi"]:.0f}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #e1e0d9;'
            f'text-align:right;">'
            f'<span style="background:{colour};color:#ffffff;padding:3px 10px;'
            f'border-radius:999px;font-size:12px;white-space:nowrap;">'
            f'{day["category"].label}</span></td>'
            f'</tr>'
        )

    current_block = ""

    if current is not None:
        current_block = (
            f'<p style="margin:0 0 18px;color:#52514e;font-size:14px;">'
            f'Right now: <strong style="color:#0b0b0b;">{current:.0f} AQI</strong>'
            f' &middot; <span style="color:{_colour(current_category.label)};'
            f'font-weight:600;">{current_category.label}</span></p>'
        )

    provenance = _provenance(model_details)

    provenance_block = ""

    if provenance:
        items = "".join(
            f'<div style="color:#898781;font-size:12px;">{line}</div>'
            for line in provenance
        )
        provenance_block = (
            f'<div style="margin-top:18px;">'
            f'<div style="color:#52514e;font-size:12px;font-weight:600;'
            f'margin-bottom:4px;">Models</div>{items}</div>'
        )

    reading_block = ""

    if reading_time is not None:
        reading_block = (
            f'<div style="color:#898781;font-size:12px;margin-top:6px;">'
            f'Computed from the reading at '
            f'{reading_time:%Y-%m-%d %H:%M} UTC</div>'
        )

    return f"""\
<div style="background:#fcfcfb;padding:24px;font-family:-apple-system,
'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;
              border:1px solid rgba(11,11,11,0.10);border-radius:14px;
              overflow:hidden;">
    <div style="background:{accent};padding:16px 20px;">
      <div style="color:#ffffff;font-size:13px;letter-spacing:0.04em;
                  text-transform:uppercase;opacity:0.9;">
        {'Air quality alert' if worst['aqi'] >= threshold else 'Air quality outlook'}
      </div>
      <div style="color:#ffffff;font-size:20px;font-weight:700;margin-top:2px;">
        {title} &middot; next 3 days
      </div>
    </div>

    <div style="padding:20px;">
      <p style="margin:0 0 6px;color:#0b0b0b;font-size:15px;font-weight:600;">
        {headline}</p>
      <p style="margin:0 0 18px;color:#52514e;font-size:14px;">{lead}</p>
      {current_block}

      <table style="width:100%;border-collapse:collapse;
                    border-top:1px solid #e1e0d9;">
        {''.join(rows)}
      </table>

      <div style="margin-top:20px;padding:14px 16px;background:#f2f1ed;
                  border-radius:10px;">
        <div style="color:#0b0b0b;font-size:14px;font-weight:600;
                    margin-bottom:4px;">
          What to do &mdash; {worst['category'].label}</div>
        <div style="color:#52514e;font-size:13px;line-height:1.5;">
          {worst['category'].headline} {worst['category'].advice}</div>
      </div>

      <div style="color:#898781;font-size:12px;margin-top:16px;">
        Alert threshold: {threshold}+ AQI ({threshold_category.label})</div>
      {reading_block}
      {provenance_block}
    </div>

    <div style="padding:14px 20px;background:#f2f1ed;color:#898781;
                font-size:11px;line-height:1.5;">{FOOTER}</div>
  </div>
</div>"""
