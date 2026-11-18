"""
AWS SageMaker Training Entry Point: Crop Suitability Classification (Module 1).

Parses hyperparameters from environment / CLI arguments.
Supports Hyperparameter Tuning Jobs (HPO) with metric regex logging.
"""

import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--model-type", type=str, default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)

    # SageMaker channels
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()

    train_path = Path(args.train)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(train_path / "X_train.csv")
    y_train = pd.read_csv(train_path / "y_train.csv").squeeze()

    X_test = pd.read_csv(train_path / "X_test.csv")
    y_test = pd.read_csv(train_path / "y_test.csv").squeeze()

    if args.model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )

    print(f"Training {args.model_type} model on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # SageMaker metric definitions regex targets:
    # "val_accuracy: ([-+]?[0-9]*\.?[0-9]+)"
    # "val_macro_f1: ([-+]?[0-9]*\.?[0-9]+)"
    print(f"val_accuracy: {acc:.4f}")
    print(f"val_macro_f1: {f1:.4f}")

    # Save artifacts into SM_MODEL_DIR for S3 upload
    joblib.dump(model, model_dir / "model.joblib")
    print(f"Model saved to {model_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
