"""
Open-Meteo Weather API Client for Module 3 — Irrigation Scheduling.

Provides functions to fetch current, forecast, and historical weather data
from the free Open-Meteo API (https://open-meteo.com). No API key required.

Features
--------
* Automatic retries with exponential back-off (3 attempts).
* Simple file-based JSON cache to avoid duplicate network calls.
* Fully typed return values and comprehensive docstrings.

NOTE: This module is part of the Smart Farm Advisory System and uses the
central configuration from ``src.utils.config``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.utils.config import (
    DATA_INTERIM,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_URL,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Cache configuration
# ──────────────────────────────────────────────────────────────────────

CACHE_DIR: Path = DATA_INTERIM / "weather_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# How long cached responses remain valid (in seconds)
CACHE_TTL_CURRENT: int = 15 * 60        # 15 minutes for current weather
CACHE_TTL_FORECAST: int = 60 * 60       # 1 hour for forecasts
CACHE_TTL_HISTORICAL: int = 7 * 24 * 3600  # 7 days for historical data

# Retry configuration
MAX_RETRIES: int = 3
BACKOFF_BASE: float = 1.0  # seconds – doubles on each retry


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────


def _cache_key(url: str, params: Dict[str, Any]) -> str:
    """Create a deterministic hash for a request URL + parameters."""
    blob = json.dumps({"url": url, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_cache(key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    """Return cached JSON if it exists and has not expired, else ``None``."""
    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None

    age = time.time() - cache_file.stat().st_mtime
    if age > ttl_seconds:
        logger.debug("Cache expired (age=%.0f s, ttl=%d s): %s", age, ttl_seconds, key[:12])
        cache_file.unlink(missing_ok=True)
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        logger.debug("Cache hit: %s", key[:12])
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(key: str, data: Dict[str, Any]) -> None:
    """Persist *data* as JSON to the cache directory."""
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Cache written: %s", key[:12])
    except OSError as exc:
        logger.warning("Failed to write cache file %s: %s", cache_file, exc)


def _request_with_retries(
    url: str,
    params: Dict[str, Any],
    *,
    cache_ttl: int,
) -> Dict[str, Any]:
    """Send a GET request to *url* with retries and caching.

    Parameters
    ----------
    url:
        API endpoint URL.
    params:
        Query-string parameters.
    cache_ttl:
        Time-to-live for the cached response in seconds.

    Returns
    -------
    dict
        Parsed JSON response from the API.

    Raises
    ------
    requests.exceptions.RequestException
        If all retry attempts fail.
    """
    key = _cache_key(url, params)
    cached = _read_cache(key, cache_ttl)
    if cached is not None:
        return cached

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "API request (attempt %d/%d): %s params=%s",
                attempt,
                MAX_RETRIES,
                url,
                {k: v for k, v in params.items() if k != "hourly"},
            )
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

            # Open-Meteo returns an "error" key on bad requests
            if "error" in data and data["error"]:
                raise ValueError(f"Open-Meteo API error: {data.get('reason', data)}")

            _write_cache(key, data)
            return data

        except (requests.exceptions.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %.1f s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error("All %d attempts exhausted for %s", MAX_RETRIES, url)

    raise last_exc  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def get_current_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch **current** weather conditions for a location.

    Parameters
    ----------
    latitude:
        Location latitude in decimal degrees (e.g. ``20.5937``).
    longitude:
        Location longitude in decimal degrees (e.g. ``78.9629``).

    Returns
    -------
    dict
        Keys: ``temperature_c``, ``relative_humidity_pct``,
        ``wind_speed_kmh``, ``precipitation_mm``,
        ``shortwave_radiation_wm2``, ``timestamp``.
    """
    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "wind_speed_10m,shortwave_radiation"
        ),
        "timezone": "auto",
    }

    raw = _request_with_retries(
        OPEN_METEO_FORECAST_URL, params, cache_ttl=CACHE_TTL_CURRENT,
    )

    current = raw.get("current", {})
    return {
        "temperature_c": current.get("temperature_2m"),
        "relative_humidity_pct": current.get("relative_humidity_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "precipitation_mm": current.get("precipitation", 0.0),
        "shortwave_radiation_wm2": current.get("shortwave_radiation", 0.0),
        "timestamp": current.get("time"),
    }


def get_forecast(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """Fetch a daily weather forecast for up to 16 days.

    Parameters
    ----------
    latitude:
        Location latitude.
    longitude:
        Location longitude.
    days:
        Number of forecast days (1–16).  Defaults to ``7``.

    Returns
    -------
    list[dict]
        One dict per day with keys: ``date``, ``temp_max_c``,
        ``temp_min_c``, ``temp_mean_c``, ``precipitation_sum_mm``,
        ``wind_speed_max_kmh``, ``shortwave_radiation_sum_mjm2``,
        ``et0_fao_mm``.
    """
    days = max(1, min(days, 16))

    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,wind_speed_10m_max,"
            "shortwave_radiation_sum,et0_fao_evapotranspiration"
        ),
        "timezone": "auto",
        "forecast_days": days,
    }

    raw = _request_with_retries(
        OPEN_METEO_FORECAST_URL, params, cache_ttl=CACHE_TTL_FORECAST,
    )

    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    n = len(dates)

    forecasts: List[Dict[str, Any]] = []
    for i in range(n):
        forecasts.append(
            {
                "date": dates[i],
                "temp_max_c": daily.get("temperature_2m_max", [None] * n)[i],
                "temp_min_c": daily.get("temperature_2m_min", [None] * n)[i],
                "temp_mean_c": daily.get("temperature_2m_mean", [None] * n)[i],
                "precipitation_sum_mm": daily.get("precipitation_sum", [0.0] * n)[i],
                "wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None] * n)[i],
                "shortwave_radiation_sum_mjm2": daily.get(
                    "shortwave_radiation_sum", [None] * n
                )[i],
                "et0_fao_mm": daily.get(
                    "et0_fao_evapotranspiration", [None] * n
                )[i],
            }
        )

    return forecasts


def get_historical(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> List[Dict[str, Any]]:
    """Fetch historical daily weather data for a date range.

    Parameters
    ----------
    latitude:
        Location latitude.
    longitude:
        Location longitude.
    start_date:
        Start date as ``"YYYY-MM-DD"`` string.
    end_date:
        End date as ``"YYYY-MM-DD"`` string.

    Returns
    -------
    list[dict]
        One dict per day with the same schema as :func:`get_forecast`.
    """
    params: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,wind_speed_10m_max,"
            "shortwave_radiation_sum,et0_fao_evapotranspiration"
        ),
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
    }

    raw = _request_with_retries(
        OPEN_METEO_HISTORICAL_URL, params, cache_ttl=CACHE_TTL_HISTORICAL,
    )

    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    n = len(dates)

    records: List[Dict[str, Any]] = []
    for i in range(n):
        records.append(
            {
                "date": dates[i],
                "temp_max_c": daily.get("temperature_2m_max", [None] * n)[i],
                "temp_min_c": daily.get("temperature_2m_min", [None] * n)[i],
                "temp_mean_c": daily.get("temperature_2m_mean", [None] * n)[i],
                "precipitation_sum_mm": daily.get("precipitation_sum", [0.0] * n)[i],
                "wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None] * n)[i],
                "shortwave_radiation_sum_mjm2": daily.get(
                    "shortwave_radiation_sum", [None] * n
                )[i],
                "et0_fao_mm": daily.get(
                    "et0_fao_evapotranspiration", [None] * n
                )[i],
            }
        )

    return records


def clear_cache() -> int:
    """Delete all cached weather API responses.

    Returns
    -------
    int
        Number of cache files removed.
    """
    removed = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        removed += 1
    logger.info("Cleared %d cached weather responses.", removed)
    return removed


# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.common import setup_logging

    setup_logging("weather_api", level="DEBUG")

    # Default location: central India
    lat, lon = 20.5937, 78.9629

    print("\n── Current Weather ──")
    current = get_current_weather(lat, lon)
    for k, v in current.items():
        print(f"  {k}: {v}")

    print("\n── 3-Day Forecast ──")
    forecast = get_forecast(lat, lon, days=3)
    for day in forecast:
        print(f"  {day['date']}: {day['temp_max_c']}°C / {day['temp_min_c']}°C, "
              f"rain={day['precipitation_sum_mm']} mm")

    print("\n── Historical (last 5 days) ──")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    hist = get_historical(lat, lon, start, end)
    for day in hist:
        print(f"  {day['date']}: {day['temp_mean_c']}°C, "
              f"rain={day['precipitation_sum_mm']} mm")
