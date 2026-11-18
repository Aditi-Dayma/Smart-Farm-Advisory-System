"""
AWS SageMaker Training Entry Point: Irrigation Soil Moisture Regression (Module 3).

Trains Gradient Boosting Regression model on weather, FAO-56 evapotranspiration (ETc),
and soil features. Evaluates MAE, RMSE, R2, and serializes the model.
"""

import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "temperature_max", "temperature_min", "temperature_mean",
    "humidity", "wind_speed", "solar_radiation", "rainfall",
    "crop_stage", "days_since_planting",
    "soil_type_encoded", "crop_encoded",
    "current_soil_moisture", "et0", "etc",
]
TARGET = "next_day_soil_moisture"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()

    train_dir = Path(args.train)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_file = list(train_dir.glob("*.csv"))[0]
    df = pd.read_csv(csv_file)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=42,
    )
    print("Training GradientBoostingRegressor for soil moisture...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"val_mae: {mae:.6f}")
    print(f"val_rmse: {rmse:.6f}")
    print(f"val_r2: {r2:.4f}")

    joblib.dump(model, model_dir / "model.joblib")
    with open(model_dir / "features.json", "w") as f:
        json.dump(FEATURES, f)

    print("Model and metadata stored in SM_MODEL_DIR.")


if __name__ == "__main__":
    main()
