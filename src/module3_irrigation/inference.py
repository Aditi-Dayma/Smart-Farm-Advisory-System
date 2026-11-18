"""
Module 3 — Irrigation Scheduling: Inference / Daily Advisory.

Combines the weather API, FAO-56 calculations, ML soil-moisture
prediction, and water-balance rules into a single daily advisory.

Usage::

    python -m src.module3_irrigation.inference \\
        --lat 19.07 --lon 72.87 --crop rice --days-planted 45 --soil loamy
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.utils.config import MODELS_DIR
from src.utils.common import setup_logging, Timer
from src.module3_irrigation.weather_api import get_current_weather, get_forecast
from src.module3_irrigation.fao56 import calculate_et0, calculate_etc, KC_TABLE
from src.module3_irrigation.water_balance import (
    SoilWaterBalance,
    IrrigationAdvisory,
    SOIL_PRESETS,
)

logger = setup_logging("module3_irrigation.inference")

# Feature order must match training
FEATURE_COLUMNS = [
    "temperature_max", "temperature_min", "temperature_mean",
    "humidity", "wind_speed", "solar_radiation", "rainfall",
    "crop_stage", "days_since_planting",
    "soil_type_encoded", "crop_encoded",
    "current_soil_moisture", "et0", "etc",
]

CROP_LIST = ["rice", "maize", "cotton", "banana", "chickpea"]
SOIL_LIST = list(SOIL_PRESETS.keys())


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────


def predict_irrigation_need(
    latitude: float,
    longitude: float,
    crop: str,
    days_since_planting: int,
    soil_type: str = "loamy",
    field_area_ha: float = 1.0,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate a full irrigation advisory for a farm.

    Parameters
    ----------
    latitude, longitude : float
        Farm location coordinates.
    crop : str
        Crop being grown.
    days_since_planting : int
        Days since the crop was planted.
    soil_type : str
        Soil type (sandy, loamy, clay, silt, sandy_loam).
    field_area_ha : float
        Field area in hectares (for volume calculation).
    model_path : Path, optional
        Path to trained regression model.

    Returns
    -------
    dict with advisory details.
    """
    # ── 1. Get weather ───────────────────────────────────────────────
    with Timer("Fetching weather data"):
        try:
            current = get_current_weather(latitude, longitude)
            forecast = get_forecast(latitude, longitude, days=2)
        except Exception as e:
            logger.error("Weather API failed: %s — using defaults", e)
            current = {
                "temperature": 28, "humidity": 65, "wind_speed": 2.5,
                "rain": 0, "solar_radiation": 18,
                "temperature_max": 32, "temperature_min": 24,
            }
            forecast = [{"precipitation_sum": 0}]

    # ── 2. Calculate ET₀ and ETc ─────────────────────────────────────
    temp_max = current.get("temperature_max", current.get("temperature", 30) + 4)
    temp_min = current.get("temperature_min", current.get("temperature", 30) - 4)

    try:
        et0 = calculate_et0(
            temp_min=temp_min,
            temp_max=temp_max,
            humidity_mean=current.get("humidity", 65),
            wind_speed=current.get("wind_speed", 2),
            solar_radiation=current.get("solar_radiation", 18),
            altitude=200,
            latitude=latitude,
            day_of_year=datetime.now().timetuple().tm_yday,
        )
    except Exception:
        et0 = 4.5  # reasonable default

    etc_res = calculate_etc(et0, crop, days_since_planting)
    etc = etc_res["etc"] if isinstance(etc_res, dict) else etc_res
    rain_today = current.get("rain", 0)
    forecast_rain_24h = forecast[0].get("precipitation_sum", 0) if forecast else 0

    # ── 3. ML soil-moisture prediction (optional) ────────────────────
    ml_prediction = None
    try:
        model = joblib.load(model_path or MODELS_DIR / "irrigation_best_model.pkl")
        crop_kc = KC_TABLE.get(crop)
        total_days = crop_kc.total_season_days if crop_kc else 150
        crop_stage = min(days_since_planting / total_days, 1.0)
        soil_idx = SOIL_LIST.index(soil_type) if soil_type in SOIL_LIST else 1
        crop_idx = CROP_LIST.index(crop) if crop in CROP_LIST else 0

        features = np.array([[
            temp_max, temp_min, (temp_max + temp_min) / 2,
            current.get("humidity", 65),
            current.get("wind_speed", 2),
            current.get("solar_radiation", 18),
            rain_today,
            crop_stage, days_since_planting,
            soil_idx, crop_idx,
            0.25,  # assume moderate current moisture
            et0, etc,
        ]])
        ml_prediction = float(model.predict(features)[0])
        logger.info("ML predicted soil moisture: %.4f", ml_prediction)
    except Exception as e:
        logger.warning("ML model not available: %s — using formula only", e)

    # ── 4. Water balance advisory ────────────────────────────────────
    swb = SoilWaterBalance(soil_type=soil_type, root_depth_m=0.5)
    if ml_prediction is not None:
        swb.moisture = ml_prediction

    advisory = swb.daily_advisory(
        etc_mm=etc,
        rain_today_mm=rain_today,
        forecast_rain_24h_mm=forecast_rain_24h,
    )

    # ── 5. Build result ──────────────────────────────────────────────
    volume_litres = advisory.amount_mm * field_area_ha * 10_000 / 1000 if advisory.irrigate else 0

    result = {
        "advisory": advisory.to_dict(),
        "weather": {
            "temperature_max": round(temp_max, 1),
            "temperature_min": round(temp_min, 1),
            "humidity": round(current.get("humidity", 65), 1),
            "wind_speed": round(current.get("wind_speed", 2), 1),
            "rain_today_mm": round(rain_today, 1),
            "forecast_rain_24h_mm": round(forecast_rain_24h, 1),
        },
        "calculations": {
            "et0_mm_day": round(et0, 2),
            "etc_mm_day": round(etc, 2),
            "crop_kc": round(etc / max(et0, 0.01), 2),
        },
        "farm": {
            "crop": crop,
            "days_since_planting": days_since_planting,
            "soil_type": soil_type,
            "field_area_ha": field_area_ha,
        },
        "irrigation_volume_litres": round(volume_litres, 0),
        "ml_soil_moisture_prediction": round(ml_prediction, 4) if ml_prediction else None,
    }

    return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Irrigation Advisory Inference")
    parser.add_argument("--lat", type=float, default=19.07, help="Latitude")
    parser.add_argument("--lon", type=float, default=72.87, help="Longitude")
    parser.add_argument("--crop", type=str, default="rice", help="Crop name")
    parser.add_argument("--days-planted", type=int, default=45, help="Days since planting")
    parser.add_argument("--soil", type=str, default="loamy", help="Soil type")
    parser.add_argument("--area", type=float, default=1.0, help="Field area (hectares)")
    args = parser.parse_args()

    result = predict_irrigation_need(
        latitude=args.lat, longitude=args.lon,
        crop=args.crop, days_since_planting=args.days_planted,
        soil_type=args.soil, field_area_ha=args.area,
    )

    print("\n💧 Irrigation Advisory")
    print("=" * 50)
    adv = result["advisory"]
    status = "💧 IRRIGATE" if adv["irrigate"] else "✅ NO IRRIGATION NEEDED"
    print(f"  Status        : {status}")
    print(f"  Amount        : {adv['amount_mm']} mm ({result['irrigation_volume_litres']:.0f} litres)")
    print(f"  Reason        : {adv['reason']}")
    print(f"  Soil Moisture : {adv['soil_moisture_pct']}%")
    print(f"  ET₀           : {result['calculations']['et0_mm_day']} mm/day")
    print(f"  ETc           : {result['calculations']['etc_mm_day']} mm/day")
    print(f"  Rain Today    : {adv['rain_today_mm']} mm")
    print(f"  Forecast Rain : {adv['forecast_rain_24h_mm']} mm")
    print("=" * 50)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
