"""
Module 1 — Crop Recommendation: Data Preprocessing Pipeline.

Loads the Kaggle Crop Recommendation CSV, engineers a season feature,
handles outliers (IQR), scales numerical features, one-hot encodes
categoricals, checks class balance (optionally applies SMOTE), splits
data, and saves processed artefacts to ``data/processed/``.

Usage::

    python -m src.module1_crop.preprocess [--csv PATH] [--smote-threshold FLOAT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import (
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES_DIR,
    HYPERPARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    SEASONS,
    SOIL_TYPES,
)
from src.utils.common import load_csv, save_csv, setup_logging, Timer

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

NUMERICAL_FEATURES: List[str] = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
]

CROP_HP: Dict = HYPERPARAMS["crop_recommendation"]
TEST_SIZE: float = CROP_HP["test_size"]

logger = setup_logging("module1_crop.preprocess")

# ──────────────────────────────────────────────────────────────────────
# Season mapping heuristic
# ──────────────────────────────────────────────────────────────────────


def infer_season(temperature: float, rainfall: float) -> str:
    """Infer an Indian cropping season from temperature and rainfall patterns.

    Heuristic rules (approximate):

    * **Kharif** (monsoon, Jun–Oct): high rainfall (≥ 150 mm) *and*
      warm temperatures (≥ 25 °C).
    * **Rabi** (winter, Nov–Mar): cooler temperatures (< 25 °C) *and*
      moderate rainfall (< 150 mm).
    * **Zaid** (summer, Mar–Jun): high temperatures (≥ 30 °C) *and*
      low rainfall (< 100 mm).

    Parameters
    ----------
    temperature:
        Average temperature in °C.
    rainfall:
        Rainfall in mm.

    Returns
    -------
    str
        One of ``"Kharif"``, ``"Rabi"``, or ``"Zaid"``.
    """
    if rainfall >= 150 and temperature >= 25:
        return "Kharif"
    elif temperature < 25 and rainfall < 150:
        return "Rabi"
    elif temperature >= 30 and rainfall < 100:
        return "Zaid"
    elif rainfall >= 150:
        return "Kharif"
    elif temperature < 25:
        return "Rabi"
    else:
        return "Zaid"


def add_season_feature(
    df: pd.DataFrame,
    season_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add a ``season`` column to *df*.

    If *season_col* already exists in the dataframe, it is used directly.
    Otherwise the season is inferred from ``temperature`` and ``rainfall``
    using :func:`infer_season`.

    Parameters
    ----------
    df:
        Input dataframe (must contain ``temperature`` and ``rainfall``).
    season_col:
        Name of an existing column to use as the season label.

    Returns
    -------
    pd.DataFrame
        Dataframe with a ``season`` column.
    """
    df = df.copy()
    if season_col and season_col in df.columns:
        df.rename(columns={season_col: "season"}, inplace=True)
        logger.info("Using existing season column '%s'.", season_col)
    elif "season" in df.columns:
        logger.info("Season column already present in dataframe.")
    else:
        logger.info("Inferring season from temperature and rainfall …")
        df["season"] = df.apply(
            lambda row: infer_season(row["temperature"], row["rainfall"]),
            axis=1,
        )
    return df


# ──────────────────────────────────────────────────────────────────────
# Outlier handling (IQR)
# ──────────────────────────────────────────────────────────────────────


def cap_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Cap outliers at the 1.5×IQR fences (Winsorisation).

    For each column the lower and upper fences are computed as:

    * ``lower = Q1 - factor * IQR``
    * ``upper = Q3 + factor * IQR``

    Values outside these bounds are clipped.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Numerical columns to process.
    factor:
        IQR multiplier (default ``1.5``).

    Returns
    -------
    pd.DataFrame
        Dataframe with outliers capped.
    """
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        logger.info(
            "  %-15s  IQR=[%.2f, %.2f]  capped %d outliers",
            col, lower, upper, before,
        )
    return df


# ──────────────────────────────────────────────────────────────────────
# Feature scaling
# ──────────────────────────────────────────────────────────────────────


def scale_features(
    df: pd.DataFrame,
    columns: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Apply :class:`~sklearn.preprocessing.StandardScaler` to *columns*.

    Parameters
    ----------
    df:
        Input dataframe.
    columns:
        Numerical columns to scale.
    scaler:
        Pre-fitted scaler.  If ``None`` a new one is fitted on *df*.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        (Scaled dataframe, fitted scaler).
    """
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        logger.info("StandardScaler fitted & applied to %d features.", len(columns))
    else:
        df[columns] = scaler.transform(df[columns])
        logger.info("StandardScaler applied (pre-fitted) to %d features.", len(columns))
    return df, scaler


# ──────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────


def one_hot_encode_season(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """One-hot encode the ``season`` column (Kharif / Rabi / Zaid).

    Parameters
    ----------
    df:
        Dataframe with a ``season`` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``season_Kharif``, ``season_Rabi``, ``season_Zaid``
        columns and the original ``season`` column dropped.
    """
    df = df.copy()
    for s in SEASONS:
        df[f"season_{s}"] = (df["season"] == s).astype(int)
    df.drop(columns=["season"], inplace=True)
    logger.info("One-hot encoded season → %s", [f"season_{s}" for s in SEASONS])
    return df


def encode_soil_type(
    df: pd.DataFrame,
    encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """Label-encode ``soil_type`` if the column is present.

    Parameters
    ----------
    df:
        Input dataframe.
    encoder:
        Pre-fitted encoder.  If ``None`` a new one is fitted.

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder | None]
        (Dataframe with encoded ``soil_type``, fitted encoder) or
        ``(df, None)`` if the column does not exist.
    """
    if "soil_type" not in df.columns:
        logger.info("No 'soil_type' column found — skipping encoding.")
        return df, None

    df = df.copy()
    if encoder is None:
        encoder = LabelEncoder()
        df["soil_type"] = encoder.fit_transform(df["soil_type"])
        logger.info("LabelEncoder fitted for soil_type (%d classes).", len(encoder.classes_))
    else:
        df["soil_type"] = encoder.transform(df["soil_type"])
        logger.info("LabelEncoder applied (pre-fitted) for soil_type.")
    return df, encoder


# ──────────────────────────────────────────────────────────────────────
# Class balance check & SMOTE
# ──────────────────────────────────────────────────────────────────────


def check_and_balance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    smote_threshold: float = 0.5,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Print class distribution and apply SMOTE if severely imbalanced.

    The *imbalance ratio* is defined as
    ``min_class_count / max_class_count``.  If it falls below
    *smote_threshold*, SMOTE is applied.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target series.
    smote_threshold:
        Ratio below which SMOTE is triggered (default ``0.5``).
    random_state:
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (Resampled X, resampled y).
    """
    dist = y.value_counts().sort_index()
    logger.info("Class distribution:\n%s", dist.to_string())
    print("\n📊 Class distribution:")
    print(dist.to_string())
    print()

    imbalance_ratio = dist.min() / dist.max()
    logger.info("Imbalance ratio (min/max): %.4f", imbalance_ratio)

    if imbalance_ratio < smote_threshold:
        logger.info("Imbalance ratio below %.2f — applying SMOTE …", smote_threshold)
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=random_state)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(
                "SMOTE applied: %d → %d samples.", len(y), len(y_res),
            )
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed — skipping SMOTE. "
                "Install with: pip install imbalanced-learn"
            )
    else:
        logger.info("Class distribution looks balanced (ratio=%.4f) — no SMOTE needed.", imbalance_ratio)

    return X, y


# ──────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot histograms of numerical feature distributions.

    Parameters
    ----------
    df:
        Dataframe with numerical columns.
    features:
        Column names to plot.
    save_dir:
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.hist(df[feat], bins=40, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_title(feat, fontsize=12)
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions (Crop Recommendation)", fontsize=14, y=1.02)
    fig.tight_layout()

    save_path = save_dir / "m1_feature_distributions.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature distribution plot saved → %s", save_path)
    return save_path


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    save_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot and save a Pearson correlation heatmap of *features*.

    Parameters
    ----------
    df:
        Dataframe containing the columns in *features*.
    features:
        Columns to include in the heatmap.
    save_dir:
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    fig.tight_layout()

    save_path = save_dir / "m1_correlation_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Correlation heatmap saved → %s", save_path)
    return save_path


def plot_class_balance(
    y: pd.Series,
    save_dir: Path = FIGURES_DIR,
    *,
    title: str = "Class Balance — Crop Labels",
) -> Path:
    """Bar chart of target-class frequencies.

    Parameters
    ----------
    y:
        Target series.
    save_dir:
        Directory to save the figure.
    title:
        Plot title.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    dist = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    dist.plot(kind="bar", ax=ax, color="teal", edgecolor="black", alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Crop")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    save_path = save_dir / "m1_class_balance.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class balance chart saved → %s", save_path)
    return save_path


# ──────────────────────────────────────────────────────────────────────
# Main preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────


def run_preprocessing(
    csv_path: Optional[Path] = None,
    smote_threshold: float = 0.5,
) -> Dict[str, Path]:
    """Execute the full preprocessing pipeline for Module 1.

    Steps:

    1. Load raw CSV.
    2. Add / infer season feature.
    3. Cap outliers with IQR.
    4. Generate visualisations (before scaling).
    5. Encode target labels.
    6. One-hot encode season (and soil_type if present).
    7. Scale numerical features.
    8. Check class balance & apply SMOTE if needed.
    9. Train/test split (stratified).
    10. Save all artefacts.

    Parameters
    ----------
    csv_path:
        Path to the raw Kaggle Crop Recommendation CSV.  Defaults to
        ``data/raw/Crop_recommendation.csv``.
    smote_threshold:
        Imbalance ratio below which SMOTE is triggered.

    Returns
    -------
    dict[str, Path]
        Mapping of artefact names to their saved file paths.
    """
    artefacts: Dict[str, Path] = {}

    # ── 1. Load data ──────────────────────────────────────────────
    if csv_path is None:
        csv_path = DATA_RAW / "Crop_recommendation.csv"
    df = load_csv(csv_path)

    logger.info("Shape: %s  |  Columns: %s", df.shape, list(df.columns))
    logger.info("Missing values:\n%s", df.isnull().sum().to_string())

    # ── 2. Season feature ─────────────────────────────────────────
    df = add_season_feature(df)
    logger.info("Season distribution:\n%s", df["season"].value_counts().to_string())

    # ── 3. Outlier capping ────────────────────────────────────────
    with Timer("Outlier capping (IQR)"):
        df = cap_outliers_iqr(df, NUMERICAL_FEATURES)

    # ── 4. Visualisations (on original scale) ─────────────────────
    with Timer("Generating visualisations"):
        artefacts["feat_dist_plot"] = plot_feature_distributions(df, NUMERICAL_FEATURES)
        artefacts["corr_heatmap"] = plot_correlation_heatmap(df, NUMERICAL_FEATURES)
        artefacts["class_balance_plot"] = plot_class_balance(df["label"])

    # ── 5. Encode target labels ───────────────────────────────────
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])
    logger.info("Label classes (%d): %s", len(label_encoder.classes_), list(label_encoder.classes_))

    # ── 6. Encode categoricals ────────────────────────────────────
    df = one_hot_encode_season(df)
    df, soil_encoder = encode_soil_type(df)

    # ── 7. Scale numerical features ───────────────────────────────
    df, scaler = scale_features(df, NUMERICAL_FEATURES)

    # ── 8. Separate features / target ─────────────────────────────
    target_col = "label_encoded"
    drop_cols = ["label", target_col] if "label" in df.columns else [target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # ── 9. Class balance / SMOTE ──────────────────────────────────
    X, y = check_and_balance(X, y, smote_threshold=smote_threshold)

    # ── 10. Train/test split ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── 11. Save processed data ──────────────────────────────────
    out_dir = DATA_PROCESSED / "module1_crop"
    out_dir.mkdir(parents=True, exist_ok=True)

    artefacts["X_train"] = save_csv(X_train, out_dir / "X_train.csv")
    artefacts["X_test"] = save_csv(X_test, out_dir / "X_test.csv")
    artefacts["y_train"] = save_csv(y_train.to_frame(), out_dir / "y_train.csv")
    artefacts["y_test"] = save_csv(y_test.to_frame(), out_dir / "y_test.csv")

    # Feature names for downstream modules
    feature_names_path = out_dir / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    artefacts["feature_names"] = feature_names_path
    logger.info("Feature names saved → %s", feature_names_path)

    # ── 12. Save scaler & encoders ────────────────────────────────
    encoder_dir = MODELS_DIR / "module1_crop"
    encoder_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = encoder_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    artefacts["scaler"] = scaler_path
    logger.info("StandardScaler saved → %s", scaler_path)

    le_path = encoder_dir / "label_encoder.joblib"
    joblib.dump(label_encoder, le_path)
    artefacts["label_encoder"] = le_path
    logger.info("LabelEncoder saved → %s", le_path)

    if soil_encoder is not None:
        se_path = encoder_dir / "soil_encoder.joblib"
        joblib.dump(soil_encoder, se_path)
        artefacts["soil_encoder"] = se_path
        logger.info("SoilEncoder saved → %s", se_path)

    logger.info("✅ Preprocessing complete. Artefacts: %s", list(artefacts.keys()))
    return artefacts


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Module 1 — Crop Recommendation: Preprocessing Pipeline",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to Kaggle Crop_recommendation.csv (default: data/raw/Crop_recommendation.csv)",
    )
    parser.add_argument(
        "--smote-threshold",
        type=float,
        default=0.5,
        help="Imbalance ratio below which SMOTE is applied (default: 0.5)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    with Timer("Module 1 preprocessing pipeline"):
        artefacts = run_preprocessing(
            csv_path=args.csv,
            smote_threshold=args.smote_threshold,
        )
    print("\n✅ Saved artefacts:")
    for name, path in artefacts.items():
        print(f"  {name:25s} → {path}")
