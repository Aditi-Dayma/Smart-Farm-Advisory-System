"""
Module 1 — Crop Recommendation: Model Training Pipeline.

Trains three classifiers (Logistic Regression, Random Forest, XGBoost),
performs hyperparameter tuning with stratified cross-validation, selects
the best model by macro F1 score, and serialises it for downstream use.

Supports SageMaker training entry-point format (reads ``SM_CHANNEL_TRAIN``,
``SM_MODEL_DIR``, and ``SM_HP_*`` environment variables when present).

Usage::

    # Local
    python -m src.module1_crop.train

    # With custom processed-data directory
    python -m src.module1_crop.train --data-dir data/processed/module1_crop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.utils.config import (
    DATA_PROCESSED,
    HYPERPARAMS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils.common import load_csv, setup_logging, Timer

# ──────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────

logger = setup_logging("module1_crop.train")

# ──────────────────────────────────────────────────────────────────────
# SageMaker-aware argument parser
# ──────────────────────────────────────────────────────────────────────


def _parse_sagemaker_env() -> Dict[str, Any]:
    """Read SageMaker training environment variables if present.

    Returns
    -------
    dict[str, Any]
        Keys: ``data_dir``, ``model_dir``, and any ``SM_HP_*`` hypers.
    """
    sm: Dict[str, Any] = {}
    sm["data_dir"] = os.environ.get("SM_CHANNEL_TRAIN")
    sm["model_dir"] = os.environ.get("SM_MODEL_DIR")

    # Collect hyper-parameters prefixed with SM_HP_
    for key, val in os.environ.items():
        if key.startswith("SM_HP_"):
            hp_name = key[len("SM_HP_"):].lower()
            # Try numeric conversion
            try:
                val = int(val)  # type: ignore[assignment]
            except ValueError:
                try:
                    val = float(val)  # type: ignore[assignment]
                except ValueError:
                    pass
            sm[hp_name] = val

    return sm


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI / SageMaker arguments.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``data_dir`` and ``model_dir``.
    """
    sm_env = _parse_sagemaker_env()

    default_data = (
        Path(sm_env["data_dir"])
        if sm_env.get("data_dir")
        else DATA_PROCESSED / "module1_crop"
    )
    default_model = (
        Path(sm_env["model_dir"])
        if sm_env.get("model_dir")
        else MODELS_DIR / "module1_crop"
    )

    parser = argparse.ArgumentParser(
        description="Module 1 — Crop Recommendation: Model Training",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help="Directory containing processed train data (X_train.csv, y_train.csv).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model,
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=HYPERPARAMS["crop_recommendation"]["cv_folds"],
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_SEED,
        help="Global random seed.",
    )
    return parser.parse_args(argv)


# ──────────────────────────────────────────────────────────────────────
# Model definitions & hyperparameter grids
# ──────────────────────────────────────────────────────────────────────


def get_model_configs(random_state: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """Return a list of model configurations to train and evaluate.

    Each dict has:

    * ``name`` — human-readable name
    * ``model`` — unfitted estimator instance
    * ``param_grid`` — hyperparameter grid for search (may be empty
      for the baseline)

    Parameters
    ----------
    random_state:
        Random seed forwarded to estimators.

    Returns
    -------
    list[dict[str, Any]]
        Model configurations.
    """
    configs: List[Dict[str, Any]] = [
        # ── Baseline ──────────────────────────────────────────────
        {
            "name": "LogisticRegression",
            "model": LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_grid": {},  # no tuning for baseline
        },
        # ── Random Forest ─────────────────────────────────────────
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        # ── XGBoost ───────────────────────────────────────────────
        {
            "name": "XGBClassifier",
            "model": XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 8, 12],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            },
        },
    ]
    return configs


# ──────────────────────────────────────────────────────────────────────
# Cross-validated training (randomised search for large grids)
# ──────────────────────────────────────────────────────────────────────


def _train_single_model(
    name: str,
    model: Any,
    param_grid: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, Dict[str, float]]:
    """Train a single model with optional randomised hyperparameter search.

    If *param_grid* is empty the model is trained directly with cross-validation.
    Otherwise :class:`~sklearn.model_selection.RandomizedSearchCV` is used
    (capped at 30 iterations to keep runtime practical).

    Parameters
    ----------
    name:
        Model display name (for logging).
    model:
        Unfitted sklearn-compatible estimator.
    param_grid:
        Hyperparameter grid for randomised search.
    X_train:
        Training feature matrix.
    y_train:
        Training target vector.
    cv_folds:
        Number of stratified CV folds.
    random_state:
        Random seed.

    Returns
    -------
    tuple[estimator, dict[str, float]]
        (Best fitted model, dict of mean CV metrics).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    if param_grid:
        from sklearn.model_selection import RandomizedSearchCV

        n_iter = min(30, _grid_size(param_grid))
        logger.info(
            "⚙  %s — RandomizedSearchCV (%d iters, %d folds) …",
            name, n_iter, cv_folds,
        )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            refit=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        logger.info("  Best params: %s", search.best_params_)
        logger.info("  Best CV F1 (macro): %.4f", search.best_score_)

        # Re-compute full scoring dict via cross_validate on the best estimator
        cv_results = cross_validate(
            best_model, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
    else:
        logger.info("⚙  %s — cross-validation (%d folds, no tuning) …", name, cv_folds)
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        # Fit on full training data
        best_model = model.fit(X_train, y_train)

    metrics: Dict[str, float] = {
        k.replace("test_", "cv_"): float(np.mean(v))
        for k, v in cv_results.items()
        if k.startswith("test_")
    }
    return best_model, metrics


def _grid_size(grid: Dict[str, List[Any]]) -> int:
    """Calculate total combinations in a parameter grid."""
    size = 1
    for vals in grid.values():
        size *= len(vals)
    return size


# ──────────────────────────────────────────────────────────────────────
# Full training pipeline
# ──────────────────────────────────────────────────────────────────────


def run_training(
    data_dir: Path,
    model_dir: Path,
    cv_folds: int = 5,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute the full training pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``X_train.csv`` and ``y_train.csv``.
    model_dir:
        Output directory for the best model and training report.
    cv_folds:
        Number of cross-validation folds.
    random_state:
        Global random seed.

    Returns
    -------
    dict[str, Any]
        Training summary with per-model metrics and the best model name.
    """
    # ── Load data ─────────────────────────────────────────────────
    X_train = load_csv(data_dir / "X_train.csv")
    y_train = load_csv(data_dir / "y_train.csv").iloc[:, 0]

    logger.info("Training data loaded: X=%s  y=%s", X_train.shape, y_train.shape)

    # ── Train all models ──────────────────────────────────────────
    configs = get_model_configs(random_state=random_state)
    results: Dict[str, Dict[str, Any]] = {}
    best_name: str = ""
    best_f1: float = -1.0
    best_model: Any = None

    for cfg in configs:
        name = cfg["name"]
        with Timer(f"Training {name}"):
            model, metrics = _train_single_model(
                name=name,
                model=cfg["model"],
                param_grid=cfg["param_grid"],
                X_train=X_train.values,
                y_train=y_train.values,
                cv_folds=cv_folds,
                random_state=random_state,
            )

        results[name] = {
            "metrics": metrics,
            "params": model.get_params() if hasattr(model, "get_params") else {},
        }

        # Pretty-print metrics
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v:.4f}")

        f1 = metrics.get("cv_f1_macro", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # ── Save best model ───────────────────────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info("🏆 Best model: %s (F1-macro=%.4f) saved → %s", best_name, best_f1, model_path)

    # ── Save training report ──────────────────────────────────────
    report = {
        "best_model": best_name,
        "best_f1_macro": best_f1,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "models": {},
    }
    for name, res in results.items():
        report["models"][name] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in res["metrics"].items()
        }
    report_path = model_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Training report saved → %s", report_path)

    print(f"\n{'*' * 60}")
    print(f"  🏆 Best model: {best_name}  |  Macro-F1: {best_f1:.4f}")
    print(f"  Saved to: {model_path}")
    print(f"{'*' * 60}\n")

    return report


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    args = parse_args()
    with Timer("Module 1 training pipeline"):
        summary = run_training(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
