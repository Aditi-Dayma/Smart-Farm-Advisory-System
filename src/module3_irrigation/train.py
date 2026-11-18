"""
Module 3 — Irrigation Scheduling: Regression Model Training.

Generates SIMULATED training data (clearly labelled), trains regression
models to predict next-day soil moisture, and compares ML performance
against the pure FAO-56 formula baseline.

NOTE: This model is trained on **simulated data** for demonstration
purposes.  In production, real soil-moisture sensor data should be used.

Usage::

    python -m src.module3_irrigation.train [--output-dir PATH] [--days 365]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils.common import setup_logging, Timer
from src.module3_irrigation.fao56 import calculate_et0, calculate_etc, KC_TABLE
from src.module3_irrigation.water_balance import SoilWaterBalance, SOIL_PRESETS

logger = setup_logging("module3_irrigation.train")

# ──────────────────────────────────────────────────────────────────────
# Simulated data generation
# ──────────────────────────────────────────────────────────────────────


def generate_simulated_data(
    n_days: int = 365,
    crops: list[str] | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate simulated daily weather + soil-moisture data.

    Uses FAO-56 water balance with added noise to simulate real
    sensor readings.  The data is clearly labelled as simulated.

    Returns DataFrame with features and target (next_day_soil_moisture).
    """
    rng = np.random.RandomState(seed)
    crops = crops or ["rice", "maize", "cotton", "banana", "chickpea"]
    soil_types = list(SOIL_PRESETS.keys())

    records = []

    for crop_name in crops:
        if crop_name not in KC_TABLE:
            logger.warning("Crop '%s' not in KC_TABLE — skipping.", crop_name)
            continue

        crop_kc = KC_TABLE[crop_name]
        total_days = crop_kc.total_season_days

        for soil_type in soil_types[:3]:  # sandy, loamy, clay
            swb = SoilWaterBalance(soil_type=soil_type, root_depth_m=0.5)

            for day in range(min(n_days, total_days)):
                # Simulate weather with seasonal patterns
                day_of_year = day + 150  # start from June (Kharif)
                seasonal_temp = 25 + 8 * np.sin(2 * np.pi * day_of_year / 365)

                temp_max = seasonal_temp + rng.normal(3, 1.5)
                temp_min = seasonal_temp - rng.normal(5, 1.5)
                temp_mean = (temp_max + temp_min) / 2
                humidity = np.clip(60 + rng.normal(0, 15), 20, 95)
                wind_speed = np.clip(rng.exponential(2) + 0.5, 0.3, 12)
                solar_radiation = np.clip(rng.normal(18, 4), 5, 30)

                # Rainfall: monsoon-influenced pattern
                rain_prob = 0.4 * np.sin(2 * np.pi * day_of_year / 365) ** 2
                rain = rng.exponential(15) if rng.random() < rain_prob else 0.0
                rain = np.clip(rain, 0, 80)

                # Calculate ET₀ and ETc
                try:
                    et0 = calculate_et0(
                        temp_min=temp_min, temp_max=temp_max,
                        humidity_mean=humidity, wind_speed=wind_speed,
                        solar_radiation=solar_radiation,
                        altitude=200, latitude=20.0,
                        day_of_year=day_of_year,
                    )
                except Exception:
                    et0 = 4.0  # fallback

                etc_res = calculate_etc(et0, crop_name, day)
                etc = etc_res["etc"] if isinstance(etc_res, dict) else etc_res

                # Update water balance
                current_moisture = swb.moisture
                swb.update(rain_mm=rain, irrigation_mm=0.0, etc_mm=etc)
                next_moisture = swb.moisture

                # Add sensor noise
                next_moisture_noisy = next_moisture + rng.normal(0, 0.005)
                next_moisture_noisy = np.clip(next_moisture_noisy, swb.wilting_point, swb.saturation)

                # Encode crop stage
                stage_frac = day / max(total_days, 1)

                records.append({
                    "temperature_max": round(temp_max, 1),
                    "temperature_min": round(temp_min, 1),
                    "temperature_mean": round(temp_mean, 1),
                    "humidity": round(humidity, 1),
                    "wind_speed": round(wind_speed, 1),
                    "solar_radiation": round(solar_radiation, 1),
                    "rainfall": round(rain, 1),
                    "crop_stage": round(stage_frac, 3),
                    "days_since_planting": day,
                    "soil_type_encoded": list(SOIL_PRESETS.keys()).index(soil_type),
                    "crop_encoded": crops.index(crop_name),
                    "current_soil_moisture": round(current_moisture, 4),
                    "et0": round(et0, 2),
                    "etc": round(etc, 2),
                    # Target
                    "next_day_soil_moisture": round(next_moisture_noisy, 4),
                    # Metadata (not features)
                    "_crop": crop_name,
                    "_soil_type": soil_type,
                    "_day": day,
                    "_simulated": True,
                })

    df = pd.DataFrame(records)
    logger.info("Generated %d simulated records for %d crops", len(df), len(crops))
    return df


# ──────────────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "temperature_max", "temperature_min", "temperature_mean",
    "humidity", "wind_speed", "solar_radiation", "rainfall",
    "crop_stage", "days_since_planting",
    "soil_type_encoded", "crop_encoded",
    "current_soil_moisture", "et0", "etc",
]

TARGET_COLUMN = "next_day_soil_moisture"


def train_models(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and compare regression models."""
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = FIGURES_DIR / "module3_irrigation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED,
    )
    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # ── Models ───────────────────────────────────────────────────────
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_SEED,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    best_model = None
    best_name = ""
    best_r2 = -np.inf

    for name, model in models.items():
        with Timer(f"Training {name}"):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": round(mae, 6), "RMSE": round(rmse, 6), "R2": round(r2, 4)}
        logger.info("  %-25s  MAE=%.6f  RMSE=%.6f  R²=%.4f", name, mae, rmse, r2)

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    # ── Formula baseline ─────────────────────────────────────────────
    # The "formula prediction" is just current_moisture - etc + rain (simplified)
    etc_idx = FEATURE_COLUMNS.index("etc")
    rain_idx = FEATURE_COLUMNS.index("rainfall")
    moisture_idx = FEATURE_COLUMNS.index("current_soil_moisture")

    formula_pred = X_test[:, moisture_idx] + (X_test[:, rain_idx] - X_test[:, etc_idx]) / (0.5 * 1000)
    formula_pred = np.clip(formula_pred, 0.04, 0.50)

    formula_mae = mean_absolute_error(y_test, formula_pred)
    formula_rmse = np.sqrt(mean_squared_error(y_test, formula_pred))
    formula_r2 = r2_score(y_test, formula_pred)
    results["FormulaBaseline"] = {"MAE": round(formula_mae, 6), "RMSE": round(formula_rmse, 6), "R2": round(formula_r2, 4)}
    logger.info("  %-25s  MAE=%.6f  RMSE=%.6f  R²=%.4f", "FormulaBaseline", formula_mae, formula_rmse, formula_r2)

    # ── Save best model ──────────────────────────────────────────────
    model_path = output_dir / "irrigation_best_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Best model (%s) saved → %s", best_name, model_path)

    # Save feature columns
    with open(output_dir / "irrigation_feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    # ── Comparison plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(results.keys())
    r2_values = [results[n]["R2"] for n in model_names]
    colors = ["#2ecc71" if n == best_name else "#3498db" if n != "FormulaBaseline" else "#e74c3c" for n in model_names]
    bars = ax.bar(model_names, r2_values, color=colors, edgecolor="black")
    ax.set_ylabel("R² Score")
    ax.set_title("Irrigation Model Comparison — ML vs Formula Baseline")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # ── Feature importance ───────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color="teal")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([FEATURE_COLUMNS[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        fig.savefig(fig_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    # ── Save results ─────────────────────────────────────────────────
    all_results = {
        "model_comparison": results,
        "best_model": best_name,
        "note": "This model is trained on SIMULATED data for demonstration purposes.",
    }
    with open(fig_dir / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run_training_pipeline(
    n_days: int = 365,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate data, train models, and save results."""
    with Timer("Generating simulated data"):
        df = generate_simulated_data(n_days=n_days)

    # Save simulated data
    data_dir = DATA_PROCESSED / "module3_irrigation"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "simulated_irrigation_data.csv", index=False)
    logger.info("Simulated data saved → %s", data_dir / "simulated_irrigation_data.csv")

    with Timer("Training models"):
        results = train_models(df, output_dir=output_dir)

    return results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train irrigation regression model")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_training_pipeline(n_days=args.days, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
