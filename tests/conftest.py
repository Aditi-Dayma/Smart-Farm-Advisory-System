"""
Pytest configuration and shared fixtures for Smart Farm Advisory System test suite.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_soil_weather_features():
    """Sample soil and weather feature dictionary for crop suitability testing."""
    return {
        "N": 45.0,
        "P": 50.0,
        "K": 40.0,
        "temperature": 26.5,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }


@pytest.fixture
def sample_crop_calendar_df():
    """Minimal sample crop calendar dataframe."""
    data = [
        {"crop": "rice", "season": "Kharif", "sowing_start_month": 6, "sowing_end_month": 7, "harvest_start_month": 10, "harvest_end_month": 11},
        {"crop": "chickpea", "season": "Rabi", "sowing_start_month": 10, "sowing_end_month": 11, "harvest_start_month": 2, "harvest_end_month": 3},
        {"crop": "watermelon", "season": "Zaid", "sowing_start_month": 2, "sowing_end_month": 3, "harvest_start_month": 5, "harvest_end_month": 6},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_disease_result():
    """Sample output from Module 2 (Disease Detection)."""
    return {
        "crop": "Tomato",
        "disease": "Early Blight",
        "confidence": 0.92,
        "all_predictions": [
            {"crop": "Tomato", "disease": "Early Blight", "confidence": 0.92},
            {"crop": "Tomato", "disease": "Late Blight", "confidence": 0.05},
        ],
    }


@pytest.fixture
def sample_weather_forecast():
    """Sample weather forecast dictionary for timing and water balance."""
    return {
        "temperature": 28.0,
        "temperature_max": 32.0,
        "temperature_min": 24.0,
        "humidity": 65.0,
        "wind_speed": 6.5,
        "rain_next_hours": 0.0,
        "precipitation": 0.0,
    }
