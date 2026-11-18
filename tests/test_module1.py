"""
Unit tests for Module 1: Crop Suitability Classification & Season Filter.
"""

import pytest
import numpy as np
import pandas as pd
from src.module1_crop.inference import is_crop_in_season, get_current_season


def test_get_current_season():
    assert get_current_season(7) == "Kharif"
    assert get_current_season(11) == "Rabi"
    assert get_current_season(4) == "Zaid"


def test_season_filter_window(sample_crop_calendar_df):
    # Rice is sowable in July (month 7)
    assert is_crop_in_season("rice", 7, sample_crop_calendar_df) is True
    # Rice is not sowable in January (month 1)
    assert is_crop_in_season("rice", 1, sample_crop_calendar_df) is False
    # Chickpea is sowable in November (month 11)
    assert is_crop_in_season("chickpea", 11, sample_crop_calendar_df) is True


def test_outlier_clipping_logic():
    # Verify IQR boundary math
    series = pd.Series([10, 12, 14, 15, 16, 18, 100])
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    clipped = series.clip(upper=upper)
    assert clipped.max() <= upper
    assert clipped.max() < 100
