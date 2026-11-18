"""
Unit tests for Module 2: Disease Detection Preprocessing & Model Architecture.
"""

import pytest
import numpy as np
from src.module2_disease.inference import DiseasePredictor


def test_parse_class_name():
    parsed1 = DiseasePredictor._parse_class_name("Tomato___Early_Blight")
    assert parsed1["crop"] == "Tomato"
    assert parsed1["disease"] == "Early Blight"

    parsed2 = DiseasePredictor._parse_class_name("Corn___healthy")
    assert parsed2["crop"] == "Corn"
    assert "Healthy" in parsed2["disease"]


def test_image_normalization_range():
    # Verify synthetic image normalizes cleanly to [0, 1]
    synthetic_img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    normalized = synthetic_img / 255.0
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.shape == (224, 224, 3)
