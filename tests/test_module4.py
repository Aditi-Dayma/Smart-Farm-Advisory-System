"""
Unit tests for Module 4: Pesticide Advisory Rule Engine, Lookup Table & Timing Gates.
"""

import pytest
from datetime import date, timedelta
from src.module4_pesticide.lookup_table import get_treatments, get_organic_alternatives
from src.module4_pesticide.timing_rules import SprayTimingAdvisor
from src.module4_pesticide.advisory import generate_pesticide_advisory, DISCLAIMER


def test_lookup_table_treatments():
    treatments = get_treatments("Early Blight", "Tomato")
    assert len(treatments) >= 2
    types = [t.treatment_type for t in treatments]
    assert "chemical" in types
    assert "biocontrol" in types


def test_weather_spray_timing_gates():
    advisor = SprayTimingAdvisor()

    # High wind condition (> 15 km/h) -> unsafe
    safe_wind, reason_wind = advisor.check_weather_window({"wind_speed": 18.0, "rain_next_hours": 0.0})
    assert safe_wind is False
    assert "Wind speed" in reason_wind

    # Imminent rain (> 0.5 mm) -> unsafe
    safe_rain, reason_rain = advisor.check_weather_window({"wind_speed": 5.0, "rain_next_hours": 3.0})
    assert safe_rain is False
    assert "Rain forecast" in reason_rain

    # Optimal weather -> safe
    safe_optimal, _ = advisor.check_weather_window({"wind_speed": 6.0, "rain_next_hours": 0.0, "temperature": 27.0})
    assert safe_optimal is True


def test_advisory_mandatory_disclaimer(sample_disease_result, sample_weather_forecast):
    advisory = generate_pesticide_advisory(
        disease_detection_result=sample_disease_result,
        weather_forecast=sample_weather_forecast,
    )
    assert advisory.disclaimer == DISCLAIMER
    assert len(advisory.recommended_treatments) > 0
    assert "DISCLAIMER" in advisory.format_text()
