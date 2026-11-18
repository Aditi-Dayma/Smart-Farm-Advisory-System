"""
Module 3 — Irrigation Scheduling: Soil Water Balance Tracker.

Implements a simple soil water-balance model:
    Water_in (rain + irrigation) − Water_out (ETc) = ΔSoil moisture

Triggers irrigation when depletion exceeds a threshold and skips if
heavy rain is in the forecast.

Usage::

    from src.module3_irrigation.water_balance import SoilWaterBalance
    swb = SoilWaterBalance(soil_type="loamy")
    advisory = swb.daily_advisory(crop="rice", growth_stage_day=45, weather=..., forecast=...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.common import setup_logging

logger = setup_logging("module3_irrigation.water_balance")

# ──────────────────────────────────────────────────────────────────────
# Soil parameter presets  (mm per m depth)
# ──────────────────────────────────────────────────────────────────────

SOIL_PRESETS: Dict[str, Dict[str, float]] = {
    "sandy": {
        "field_capacity": 0.12,       # volumetric (m³/m³)
        "wilting_point": 0.04,
        "saturation": 0.40,
        "infiltration_rate_mm_hr": 50,
    },
    "loamy": {
        "field_capacity": 0.28,
        "wilting_point": 0.10,
        "saturation": 0.45,
        "infiltration_rate_mm_hr": 20,
    },
    "clay": {
        "field_capacity": 0.38,
        "wilting_point": 0.20,
        "saturation": 0.50,
        "infiltration_rate_mm_hr": 5,
    },
    "silt": {
        "field_capacity": 0.32,
        "wilting_point": 0.12,
        "saturation": 0.47,
        "infiltration_rate_mm_hr": 15,
    },
    "sandy_loam": {
        "field_capacity": 0.22,
        "wilting_point": 0.08,
        "saturation": 0.43,
        "infiltration_rate_mm_hr": 30,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class IrrigationAdvisory:
    """Result of a daily irrigation check."""
    irrigate: bool
    amount_mm: float
    reason: str
    soil_moisture_pct: float
    depletion_pct: float
    etc_mm: float
    rain_today_mm: float
    forecast_rain_24h_mm: float
    next_check: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "irrigate": self.irrigate,
            "amount_mm": round(self.amount_mm, 1),
            "reason": self.reason,
            "soil_moisture_pct": round(self.soil_moisture_pct, 1),
            "depletion_pct": round(self.depletion_pct, 1),
            "etc_mm": round(self.etc_mm, 2),
            "rain_today_mm": round(self.rain_today_mm, 1),
            "forecast_rain_24h_mm": round(self.forecast_rain_24h_mm, 1),
            "next_check": self.next_check.isoformat(),
        }

    def __str__(self) -> str:
        status = "💧 IRRIGATE" if self.irrigate else "✅ NO IRRIGATION NEEDED"
        lines = [
            f"  Status           : {status}",
            f"  Amount           : {self.amount_mm:.1f} mm",
            f"  Reason           : {self.reason}",
            f"  Soil Moisture    : {self.soil_moisture_pct:.1f}%",
            f"  Depletion        : {self.depletion_pct:.1f}%",
            f"  Crop Water Need  : {self.etc_mm:.2f} mm/day",
            f"  Rain Today       : {self.rain_today_mm:.1f} mm",
            f"  Forecast Rain 24h: {self.forecast_rain_24h_mm:.1f} mm",
            f"  Next Check       : {self.next_check.strftime('%Y-%m-%d')}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Soil Water Balance class
# ──────────────────────────────────────────────────────────────────────


class SoilWaterBalance:
    """
    Simple daily soil-water balance tracker.

    Tracks moisture in the root zone and decides when to irrigate.
    """

    def __init__(
        self,
        soil_type: str = "loamy",
        root_depth_m: float = 0.6,
        field_capacity: Optional[float] = None,
        wilting_point: Optional[float] = None,
        initial_moisture: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        soil_type : str
            One of 'sandy', 'loamy', 'clay', 'silt', 'sandy_loam'.
        root_depth_m : float
            Effective root zone depth in metres.
        field_capacity : float, optional
            Volumetric water content at field capacity (overrides preset).
        wilting_point : float, optional
            Volumetric water content at wilting point (overrides preset).
        initial_moisture : float, optional
            Starting volumetric water content (defaults to field capacity).
        """
        preset = SOIL_PRESETS.get(soil_type.lower(), SOIL_PRESETS["loamy"])
        self.soil_type = soil_type
        self.root_depth_m = root_depth_m
        self.field_capacity = field_capacity or preset["field_capacity"]
        self.wilting_point = wilting_point or preset["wilting_point"]
        self.saturation = preset.get("saturation", 0.45)

        # Total Available Water (TAW) in mm
        self.taw_mm = (self.field_capacity - self.wilting_point) * root_depth_m * 1000

        # Current moisture (volumetric)
        self.moisture = initial_moisture if initial_moisture is not None else self.field_capacity

        # Tracking
        self.history: List[Dict[str, float]] = []

        logger.info(
            "SoilWaterBalance — %s soil, root_depth=%.1fm, TAW=%.1fmm, FC=%.2f, WP=%.2f",
            soil_type, root_depth_m, self.taw_mm, self.field_capacity, self.wilting_point,
        )

    @property
    def current_water_mm(self) -> float:
        """Current water stored in root zone (mm)."""
        return self.moisture * self.root_depth_m * 1000

    @property
    def fc_water_mm(self) -> float:
        """Water at field capacity (mm)."""
        return self.field_capacity * self.root_depth_m * 1000

    def get_depletion(self) -> float:
        """
        Current depletion as a fraction of TAW (0 = at FC, 1 = at WP).
        """
        if self.taw_mm <= 0:
            return 0.0
        depletion = (self.field_capacity - self.moisture) * self.root_depth_m * 1000
        return max(0.0, min(1.0, depletion / self.taw_mm))

    def update(
        self,
        rain_mm: float = 0.0,
        irrigation_mm: float = 0.0,
        etc_mm: float = 0.0,
    ) -> float:
        """
        Update the water balance for one day.

        Parameters
        ----------
        rain_mm : float
            Effective rainfall (mm).
        irrigation_mm : float
            Irrigation applied (mm).
        etc_mm : float
            Crop evapotranspiration (mm).

        Returns
        -------
        float : updated volumetric moisture.
        """
        # Convert mm to volumetric
        delta_mm = rain_mm + irrigation_mm - etc_mm
        delta_vol = delta_mm / (self.root_depth_m * 1000)

        self.moisture += delta_vol

        # Cap at saturation and floor at wilting point
        self.moisture = max(self.wilting_point, min(self.saturation, self.moisture))

        # Record
        self.history.append({
            "rain_mm": rain_mm,
            "irrigation_mm": irrigation_mm,
            "etc_mm": etc_mm,
            "moisture_vol": self.moisture,
            "depletion_frac": self.get_depletion(),
        })

        return self.moisture

    def should_irrigate(self, threshold: float = 0.5) -> bool:
        """
        Check if irrigation is needed.

        Parameters
        ----------
        threshold : float
            Depletion fraction threshold (0–1). Default 0.5 means irrigate
            when 50% of TAW is depleted — the common MAD for many crops.
        """
        return self.get_depletion() >= threshold

    def calculate_irrigation_amount(self) -> float:
        """Calculate mm needed to bring soil back to field capacity."""
        deficit_mm = (self.field_capacity - self.moisture) * self.root_depth_m * 1000
        return max(0.0, deficit_mm)

    @staticmethod
    def forecast_check(
        forecast_rain_24h_mm: float,
        rain_threshold_mm: float = 10.0,
    ) -> Tuple[bool, str]:
        """
        Check if irrigation should be skipped due to forecast rain.

        Returns (skip_irrigation, reason).
        """
        if forecast_rain_24h_mm >= rain_threshold_mm:
            return True, f"Heavy rain forecast ({forecast_rain_24h_mm:.1f} mm in next 24h) — skipping irrigation"
        return False, ""

    def daily_advisory(
        self,
        etc_mm: float,
        rain_today_mm: float = 0.0,
        forecast_rain_24h_mm: float = 0.0,
        depletion_threshold: float = 0.5,
        rain_skip_threshold_mm: float = 10.0,
    ) -> IrrigationAdvisory:
        """
        Generate a daily irrigation advisory.

        Parameters
        ----------
        etc_mm : float
            Today's crop evapotranspiration.
        rain_today_mm : float
            Rainfall received today.
        forecast_rain_24h_mm : float
            Forecast rain in the next 24 hours.
        depletion_threshold : float
            When to trigger irrigation.
        rain_skip_threshold_mm : float
            Skip irrigation if forecast exceeds this.
        """
        # Update balance with today's data
        self.update(rain_mm=rain_today_mm, irrigation_mm=0.0, etc_mm=etc_mm)

        depletion = self.get_depletion()
        moisture_pct = (self.moisture - self.wilting_point) / (self.field_capacity - self.wilting_point) * 100
        moisture_pct = max(0, min(100, moisture_pct))

        # Decision logic
        if depletion < depletion_threshold:
            return IrrigationAdvisory(
                irrigate=False,
                amount_mm=0.0,
                reason=f"Soil moisture adequate — depletion at {depletion*100:.0f}% (threshold: {depletion_threshold*100:.0f}%)",
                soil_moisture_pct=moisture_pct,
                depletion_pct=depletion * 100,
                etc_mm=etc_mm,
                rain_today_mm=rain_today_mm,
                forecast_rain_24h_mm=forecast_rain_24h_mm,
            )

        # Check forecast
        skip, skip_reason = self.forecast_check(forecast_rain_24h_mm, rain_skip_threshold_mm)
        if skip:
            return IrrigationAdvisory(
                irrigate=False,
                amount_mm=0.0,
                reason=skip_reason,
                soil_moisture_pct=moisture_pct,
                depletion_pct=depletion * 100,
                etc_mm=etc_mm,
                rain_today_mm=rain_today_mm,
                forecast_rain_24h_mm=forecast_rain_24h_mm,
            )

        # Irrigation needed
        amount = self.calculate_irrigation_amount()
        return IrrigationAdvisory(
            irrigate=True,
            amount_mm=amount,
            reason=f"Soil depletion at {depletion*100:.0f}%, exceeds {depletion_threshold*100:.0f}% threshold",
            soil_moisture_pct=moisture_pct,
            depletion_pct=depletion * 100,
            etc_mm=etc_mm,
            rain_today_mm=rain_today_mm,
            forecast_rain_24h_mm=forecast_rain_24h_mm,
        )
