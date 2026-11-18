"""
Unit tests for Module 3: Irrigation Scheduling, FAO-56 & Water Balance.
"""

import pytest
from src.module3_irrigation.fao56 import calculate_etc, KC_TABLE
from src.module3_irrigation.water_balance import SoilWaterBalance, IrrigationAdvisory


def test_kc_table_contents():
    assert "rice" in KC_TABLE
    assert "maize" in KC_TABLE
    assert "cotton" in KC_TABLE
    rice_kc = KC_TABLE["rice"]
    assert rice_kc.kc_mid > rice_kc.kc_end


def test_calculate_etc():
    # Base ET0 = 5.0 mm/day
    et0 = 5.0
    etc_res = calculate_etc(et0, "rice", days_since_planting=60)
    etc_val = etc_res["etc"] if isinstance(etc_res, dict) else etc_res
    # Rice Kc_mid is ~1.20, so ETc should be ~6.0 mm/day
    assert 5.0 < etc_val <= 7.0


def test_water_balance_depletion():
    swb = SoilWaterBalance(soil_type="loamy", root_depth_m=0.5)
    initial_depletion = swb.get_depletion()
    assert initial_depletion == 0.0  # Starts at Field Capacity

    # Add 10mm of evapotranspiration with no rain
    swb.update(rain_mm=0.0, irrigation_mm=0.0, etc_mm=10.0)
    assert swb.get_depletion() > 0.0


def test_rain_skip_logic():
    # If heavy rain (> 10mm) is forecast, irrigation should be skipped
    skip, reason = SoilWaterBalance.forecast_check(forecast_rain_24h_mm=15.0, rain_threshold_mm=10.0)
    assert skip is True
    assert "Heavy rain" in reason

    skip2, _ = SoilWaterBalance.forecast_check(forecast_rain_24h_mm=2.0, rain_threshold_mm=10.0)
    assert skip2 is False
