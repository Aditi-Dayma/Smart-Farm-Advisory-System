"""
Central Configuration for Smart Farm Advisory System.

This module consolidates all project-wide settings including file paths,
cloud resource identifiers, model hyperparameters, API endpoints, and
domain-specific constants used across the four advisory modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────────────
# 1. PROJECT PATHS
# ──────────────────────────────────────────────────────────────────────

# Root of the repository (two levels up from this file)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_INTERIM: Path = DATA_DIR / "interim"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

CROP_CALENDAR_PATH: Path = DATA_RAW / "crop_calendar.csv"

# Ensure critical directories exist at import time
for _dir in (DATA_RAW, DATA_PROCESSED, DATA_INTERIM, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 2. AWS / S3 CONFIGURATION  (parameterised via env vars)
# ──────────────────────────────────────────────────────────────────────

AWS_REGION: str = os.getenv("SFAS_AWS_REGION", "ap-south-1")

S3_BUCKET_DATA: str = os.getenv("SFAS_S3_BUCKET_DATA", "smart-farm-data")
S3_BUCKET_MODELS: str = os.getenv("SFAS_S3_BUCKET_MODELS", "smart-farm-models")
S3_BUCKET_ARTIFACTS: str = os.getenv("SFAS_S3_BUCKET_ARTIFACTS", "smart-farm-artifacts")

SAGEMAKER_ROLE: str = os.getenv("SFAS_SAGEMAKER_ROLE", "arn:aws:iam::role/SageMakerExecutionRole")
SAGEMAKER_INSTANCE_TYPE: str = os.getenv("SFAS_SAGEMAKER_INSTANCE", "ml.m5.xlarge")

# ──────────────────────────────────────────────────────────────────────
# 3. API ENDPOINTS
# ──────────────────────────────────────────────────────────────────────

OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"

API_ENDPOINTS: Dict[str, str] = {
    "weather_forecast": OPEN_METEO_FORECAST_URL,
    "weather_historical": OPEN_METEO_HISTORICAL_URL,
}

# Default query parameters for Open-Meteo requests
OPEN_METEO_DEFAULT_PARAMS: Dict[str, Any] = {
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration",
    "timezone": "auto",
}

# ──────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER DEFAULTS  (per module)
# ──────────────────────────────────────────────────────────────────────

HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    # Module 1 — Crop Recommendation
    "crop_recommendation": {
        "model_type": "random_forest",
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "test_size": 0.20,
        "random_state": 42,
        "cv_folds": 5,
    },
    # Module 2 — Disease Detection (CNN)
    "disease_detection": {
        "architecture": "resnet50",
        "image_size": (224, 224),
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-4,
        "dropout_rate": 0.3,
        "early_stopping_patience": 5,
        "num_classes": 38,  # PlantVillage default
        "augmentation": True,
        "random_state": 42,
    },
    # Module 3 — Irrigation Scheduling
    "irrigation_scheduling": {
        "model_type": "xgboost",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "test_size": 0.20,
        "random_state": 42,
        "cv_folds": 5,
    },
    # Module 4 — Pesticide Advisory
    "pesticide_advisory": {
        "model_type": "gradient_boosting",
        "n_estimators": 250,
        "max_depth": 10,
        "learning_rate": 0.05,
        "test_size": 0.20,
        "random_state": 42,
        "cv_folds": 5,
    },
}

# ──────────────────────────────────────────────────────────────────────
# 5. DOMAIN CONSTANTS
# ──────────────────────────────────────────────────────────────────────

SEASONS: List[str] = [
    "Kharif",   # Jun–Oct  (monsoon)
    "Rabi",     # Nov–Mar  (winter)
    "Zaid",     # Mar–Jun  (summer)
]

GROWTH_STAGES: List[str] = [
    "Germination",
    "Seedling",
    "Vegetative",
    "Flowering",
    "Fruiting",
    "Maturity",
    "Harvest",
]

SOIL_TYPES: List[str] = [
    "Alluvial",
    "Black (Regur)",
    "Red",
    "Laterite",
    "Desert (Arid)",
    "Mountain",
    "Peaty & Marshy",
]

NUTRIENT_LABELS: List[str] = ["N", "P", "K"]

IRRIGATION_METHODS: List[str] = [
    "Drip",
    "Sprinkler",
    "Flood",
    "Furrow",
    "Center Pivot",
]

SEVERITY_LEVELS: List[str] = [
    "None",
    "Low",
    "Moderate",
    "High",
    "Critical",
]

# ──────────────────────────────────────────────────────────────────────
# 6. LOGGING DEFAULTS
# ──────────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("SFAS_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ──────────────────────────────────────────────────────────────────────
# 7. MISCELLANEOUS
# ──────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42
