"""
AWS SageMaker Processing Job: Tabular Data Preprocessing for Module 1.

Executed inside an SKLearn container on SageMaker Processing.
Reads raw data from /opt/ml/processing/input/ and writes processed
features, targets, and fitted scaler/encoders to /opt/ml/processing/output/.
"""

import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

NUMERICAL_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEASONS = ["Kharif", "Rabi", "Zaid"]


def determine_season(row: pd.Series) -> str:
    """Heuristic season assignment based on rainfall & temperature."""
    rf = row["rainfall"]
    temp = row["temperature"]
    if rf > 150 or (temp > 25 and rf > 80):
        return "Kharif"
    elif temp < 22 or rf < 50:
        return "Rabi"
    else:
        return "Zaid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Locate input CSV
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {input_path}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns")

    # Season assignment if not present
    if "season" not in df.columns:
        df["season"] = df.apply(determine_season, axis=1)

    # IQR Outlier Capping on numerical features
    for col in NUMERICAL_COLS:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = np.clip(df[col], lower_bound, upper_bound)

    # One-hot encode season
    for s in SEASONS:
        df[f"season_{s}"] = (df["season"] == s).astype(float)

    feature_cols = NUMERICAL_COLS + [f"season_{s}" for s in SEASONS]

    # Target encoding
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[feature_cols].copy()

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    # Save outputs for SageMaker training & evaluation channels
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    pd.Series(y_train, name="label").to_csv(output_path / "y_train.csv", index=False)
    pd.Series(y_test, name="label").to_csv(output_path / "y_test.csv", index=False)

    # Save artifacts (scaler, label encoder, feature columns metadata)
    joblib.dump(scaler, output_path / "scaler.pkl")
    joblib.dump(le, output_path / "label_encoder.pkl")
    with open(output_path / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print("Tabular preprocessing completed successfully.")


if __name__ == "__main__":
    main()
