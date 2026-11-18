"""
Module 4 — Pesticide Advisory: Spray Timing Rules.

Weather-gated timing logic that checks whether it is safe and optimal
to spray, respects spray intervals, and enforces pre-harvest intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from src.utils.common import setup_logging

logger = setup_logging("module4_pesticide.timing_rules")


@dataclass
class TimingResult:
    """Result of a spray timing check."""
    safe_to_spray: bool
    reason: str
    optimal_time: str  # "early_morning" or "evening"
    next_safe_window: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "safe_to_spray": self.safe_to_spray,
            "reason": self.reason,
            "optimal_time": self.optimal_time,
            "next_safe_window": self.next_safe_window.isoformat() if self.next_safe_window else None,
        }


class SprayTimingAdvisor:
    """
    Weather-gated spray timing advisor.

    Rules:
    - Skip if rain forecast within 6 hours
    - Skip if wind speed > 15 km/h
    - Skip if temperature > 35°C (phytotoxicity risk)
    - Prefer early morning (6–8 AM) or evening (4–6 PM)
    """

    def __init__(
        self,
        rain_window_hours: int = 6,
        max_wind_kmh: float = 15.0,
        max_temp_c: float = 35.0,
    ):
        self.rain_window_hours = rain_window_hours
        self.max_wind_kmh = max_wind_kmh
        self.max_temp_c = max_temp_c

    def check_weather_window(
        self,
        forecast: Dict[str, Any],
        hours_ahead: int = 6,
    ) -> Tuple[bool, str]:
        """
        Check if weather conditions are suitable for spraying.

        Parameters
        ----------
        forecast : dict
            Weather data with keys: rain_next_hours (mm), wind_speed (km/h),
            temperature (°C).  Keys are flexible.
        hours_ahead : int
            How many hours ahead to check for rain.

        Returns
        -------
        (safe_to_spray, reason)
        """
        # Rain check
        rain = forecast.get("rain_next_hours",
               forecast.get("precipitation", 
               forecast.get("rain", 0)))
        if rain > 0.5:
            return False, f"Rain forecast ({rain:.1f} mm) within {hours_ahead}h — spray will wash off"

        # Wind check
        wind = forecast.get("wind_speed",
               forecast.get("wind_speed_kmh", 0))
        if wind > self.max_wind_kmh:
            return False, f"Wind speed too high ({wind:.1f} km/h > {self.max_wind_kmh} km/h) — spray drift risk"

        # Temperature check
        temp = forecast.get("temperature",
               forecast.get("temperature_max",
               forecast.get("temp", 25)))
        if temp > self.max_temp_c:
            return False, f"Temperature too high ({temp:.1f}°C > {self.max_temp_c}°C) — phytotoxicity risk"

        return True, "Weather conditions suitable for spraying"

    def get_optimal_time(
        self,
        month: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Return the optimal spray time window.

        - Summer months (Apr–Sep) → early morning (6–8 AM) to avoid heat
        - Winter months (Oct–Mar) → early morning (8–10 AM) or evening (4–6 PM)
        """
        if month is None:
            month = datetime.now().month
        if temperature is None:
            temperature = 30 if month in range(4, 10) else 22

        if temperature > 30 or month in (4, 5, 6, 7, 8, 9):
            return "early_morning (6:00–8:00 AM)"
        return "evening (4:00–6:00 PM)"

    @staticmethod
    def check_spray_interval(
        last_spray_date: Optional[date],
        interval_days: int,
    ) -> Tuple[bool, Optional[date]]:
        """
        Check if enough time has passed since the last application.

        Returns (can_spray, next_allowed_date).
        """
        if last_spray_date is None:
            return True, None

        today = date.today()
        next_allowed = last_spray_date + timedelta(days=interval_days)
        can_spray = today >= next_allowed
        return can_spray, next_allowed

    @staticmethod
    def check_phi(
        expected_harvest_date: Optional[date],
        phi_days: int,
    ) -> Tuple[bool, Optional[date]]:
        """
        Check pre-harvest interval — is it still safe to spray?

        Returns (can_spray, last_safe_spray_date).
        """
        if expected_harvest_date is None:
            return True, None

        today = date.today()
        last_safe = expected_harvest_date - timedelta(days=phi_days)
        can_spray = today <= last_safe
        return can_spray, last_safe

    def full_timing_check(
        self,
        forecast: Dict[str, Any],
        last_spray_date: Optional[date] = None,
        spray_interval_days: int = 7,
        expected_harvest_date: Optional[date] = None,
        phi_days: int = 14,
    ) -> TimingResult:
        """
        Run all timing checks and return a consolidated result.
        """
        reasons = []

        # Weather window
        weather_ok, weather_reason = self.check_weather_window(forecast)
        if not weather_ok:
            return TimingResult(
                safe_to_spray=False,
                reason=weather_reason,
                optimal_time=self.get_optimal_time(),
                next_safe_window=datetime.now() + timedelta(hours=self.rain_window_hours),
            )

        # Spray interval
        interval_ok, next_date = self.check_spray_interval(last_spray_date, spray_interval_days)
        if not interval_ok:
            return TimingResult(
                safe_to_spray=False,
                reason=f"Spray interval not met — next allowed spray on {next_date}",
                optimal_time=self.get_optimal_time(),
                next_safe_window=datetime.combine(next_date, datetime.min.time()) if next_date else None,
            )

        # PHI
        phi_ok, last_safe = self.check_phi(expected_harvest_date, phi_days)
        if not phi_ok:
            return TimingResult(
                safe_to_spray=False,
                reason=f"Pre-harvest interval violation — spraying must stop by {last_safe} ({phi_days} days before harvest)",
                optimal_time=self.get_optimal_time(),
            )

        return TimingResult(
            safe_to_spray=True,
            reason="All timing checks passed — safe to spray",
            optimal_time=self.get_optimal_time(),
        )
