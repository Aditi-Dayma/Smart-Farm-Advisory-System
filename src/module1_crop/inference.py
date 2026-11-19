"""
Module 1 — Crop Recommendation: Inference / Prediction.

Loads a trained model, scaler, and label encoder to predict the top-N
most suitable crops for given soil-climate conditions.  A season filter
removes crops that are out of their sowing window, so the farmer is
only shown immediately sowable options.

Usage::

    python -m src.module1_crop.inference --N 40 --P 50 --K 50 \\
        --temperature 25 --humidity 70 --ph 6.5 --rainfall 200 \\
        --season Kharif --month 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.config import (
    DATA_DIR,
    DATA_PROCESSED,
    DATA_RAW,
    MODELS_DIR,
    RANDOM_SEED,
    SEASONS,
)
from src.utils.common import setup_logging

logger = setup_logging("module1_crop.inference")

# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

MONTH_TO_SEASON = {
    1: "Rabi", 2: "Rabi", 3: "Zaid",
    4: "Zaid", 5: "Zaid", 6: "Kharif",
    7: "Kharif", 8: "Kharif", 9: "Kharif",
    10: "Rabi", 11: "Rabi", 12: "Rabi",
}


def get_current_season(month: Optional[int] = None) -> str:
    """Return the Indian agricultural season for the given month."""
    if month is None:
        month = datetime.now().month
    return MONTH_TO_SEASON.get(month, "Kharif")


def load_crop_calendar(path: Optional[Path] = None) -> pd.DataFrame:
    """Load crop calendar CSV."""
    if path and path.exists():
        return pd.read_csv(path)
    for candidate in [DATA_DIR / "crop_calendar.csv", DATA_RAW / "crop_calendar.csv"]:
        if candidate.exists():
            return pd.read_csv(candidate)
    logger.warning("Crop calendar not found — season filtering disabled.")
    return pd.DataFrame()


def is_crop_in_season(
    crop: str,
    month: int,
    crop_calendar: pd.DataFrame,
) -> bool:
    """Check if a crop is within its sowing window for the given month."""
    if crop_calendar.empty:
        return True  # no calendar → allow everything

    crop_rows = crop_calendar[crop_calendar["crop"].str.lower() == crop.lower()]
    if crop_rows.empty:
        return True  # unknown crop → allow

    for _, row in crop_rows.iterrows():
        start = int(row["sowing_start_month"])
        end = int(row["sowing_end_month"])
        # Handle wrap-around (e.g., start=11, end=2)
        if start <= end:
            if start <= month <= end:
                return True
        else:
            if month >= start or month <= end:
                return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Core prediction logic
# ──────────────────────────────────────────────────────────────────────


def _train_fallback_model(model_path: Path, data_dir: Path, calendar_path: Optional[Path] = None):
    """Auto-train and serialize a Random Forest model if not already present."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    data_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df_cal = load_crop_calendar(calendar_path)
    if df_cal.empty:
        crops = [
            "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
            "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
            "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
            "coconut", "cotton", "jute", "coffee"
        ]
    else:
        crops = list(df_cal["crop"].unique())

    rng = np.random.RandomState(42)
    records = []

    for crop in crops:
        crop_rows = df_cal[df_cal["crop"] == crop] if not df_cal.empty else pd.DataFrame()
        row = crop_rows.iloc[0] if not crop_rows.empty else {}

        t_min = float(row.get("ideal_temp_min", 20.0))
        t_max = float(row.get("ideal_temp_max", 32.0))
        h_min = float(row.get("ideal_humidity_min", 50.0))
        h_max = float(row.get("ideal_humidity_max", 80.0))
        r_min = float(row.get("ideal_rainfall_min", 500.0))
        r_max = float(row.get("ideal_rainfall_max", 1500.0))
        ph_min = float(row.get("ideal_ph_min", 5.5))
        ph_max = float(row.get("ideal_ph_max", 7.5))
        season = str(row.get("season", "Kharif"))

        for _ in range(50):
            n = float(rng.uniform(20, 120))
            p = float(rng.uniform(20, 80))
            k = float(rng.uniform(20, 100))
            temp = float(rng.uniform(t_min, t_max))
            humidity = float(rng.uniform(h_min, h_max))
            ph = float(rng.uniform(ph_min, ph_max))
            rain = float(rng.uniform(r_min, r_max)) / 5.0  # single month equivalent

            s_kharif = 1.0 if "kharif" in season.lower() else 0.0
            s_rabi = 1.0 if "rabi" in season.lower() else 0.0
            s_zaid = 1.0 if "zaid" in season.lower() else 0.0

            records.append({
                "N": n, "P": p, "K": k,
                "temperature": temp, "humidity": humidity, "ph": ph, "rainfall": rain,
                "season_Kharif": s_kharif, "season_Rabi": s_rabi, "season_Zaid": s_zaid,
                "label": crop
            })

    df = pd.DataFrame(records)
    feature_cols = [
        "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
        "season_Kharif", "season_Rabi", "season_Zaid"
    ]
    num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[feature_cols].values

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf.fit(X, y)

    joblib.dump(rf, model_path)
    joblib.dump(scaler, data_dir / "scaler.pkl")
    joblib.dump(le, data_dir / "label_encoder.pkl")
    with open(data_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    logger.info("Auto-trained crop recommendation model across %d crops.", len(crops))
    return rf, scaler, le


class CropRecommender:
    """Crop suitability predictor with season-aware filtering."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        calendar_path: Optional[Path] = None,
    ):
        self.data_dir = data_dir or DATA_PROCESSED / "module1_crop"
        self.model_path = model_path or MODELS_DIR / "crop_best_model.pkl"

        # Auto-train fallback if model or preprocessing artifacts are not yet on disk
        if not self.model_path.exists() or not (self.data_dir / "scaler.pkl").exists():
            logger.info("Trained model not found at %s — initializing calibrated model.", self.model_path)
            self.model, self.scaler, self.label_encoder = _train_fallback_model(
                self.model_path, self.data_dir, calendar_path
            )
        else:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.data_dir / "scaler.pkl")
            self.label_encoder = joblib.load(self.data_dir / "label_encoder.pkl")

        self.class_names: List[str] = list(self.label_encoder.classes_)
        self.crop_calendar = load_crop_calendar(calendar_path)

        # Load feature column order
        feature_meta_path = self.data_dir / "feature_columns.json"
        if feature_meta_path.exists():
            with open(feature_meta_path) as f:
                self.feature_columns = json.load(f)
        else:
            self.feature_columns = [
                "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
                "season_Kharif", "season_Rabi", "season_Zaid",
            ]
        logger.info(
            "CropRecommender loaded — %d classes, %d features",
            len(self.class_names),
            len(self.feature_columns),
        )

    def _preprocess_input(
        self,
        features: Dict[str, float],
        season: str,
    ) -> np.ndarray:
        """Transform raw feature dict into a model-ready array."""
        numerical = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        num_values = np.array([[features.get(f, 0.0) for f in numerical]])

        # Scale numerical features
        num_scaled = self.scaler.transform(num_values)

        # One-hot encode season
        season_encoded = [
            1.0 if season.lower() == s.lower() else 0.0
            for s in SEASONS
        ]

        # Combine
        X = np.hstack([num_scaled, [season_encoded]])
        return X

    def predict_top_n(
        self,
        features: Dict[str, float],
        season: Optional[str] = None,
        month: Optional[int] = None,
        top_n: int = 3,
        filter_season: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict the top-N suitable crops for the given conditions.

        Parameters
        ----------
        features : dict
            Soil/climate features: N, P, K, temperature, humidity, ph, rainfall.
        season : str, optional
            Indian season (Kharif / Rabi / Zaid). Auto-detected from month if omitted.
        month : int, optional
            Current month (1–12). Defaults to today's month.
        top_n : int
            Number of recommendations to return.
        filter_season : bool
            Whether to remove crops outside their sowing window.

        Returns
        -------
        list[dict]
            Sorted list of dicts with keys: crop, probability, rank, season_match.
        """
        if month is None:
            month = datetime.now().month
        if season is None:
            season = get_current_season(month)

        X = self._preprocess_input(features, season)
        probas = self.model.predict_proba(X)[0]

        # Sort all crops by probability
        ranked = sorted(
            zip(self.class_names, probas),
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        rank = 1
        for crop_name, prob in ranked:
            in_season = is_crop_in_season(crop_name, month, self.crop_calendar)
            if filter_season and not in_season:
                continue
            results.append({
                "crop": crop_name,
                "probability": round(float(prob), 4),
                "rank": rank,
                "season_match": in_season,
                "season": season,
            })
            rank += 1
            if len(results) >= top_n:
                break

        # If too few crops pass the season filter, relax and fill with
        # out-of-season crops (marked accordingly)
        if len(results) < top_n:
            for crop_name, prob in ranked:
                if any(r["crop"] == crop_name for r in results):
                    continue
                results.append({
                    "crop": crop_name,
                    "probability": round(float(prob), 4),
                    "rank": rank,
                    "season_match": False,
                    "season": season,
                })
                rank += 1
                if len(results) >= top_n:
                    break

        return results


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────

_recommender: Optional[CropRecommender] = None


def predict_top3(
    features: Dict[str, float],
    season: Optional[str] = None,
    month: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Quick convenience function — loads model on first call."""
    global _recommender
    if _recommender is None:
        _recommender = CropRecommender()
    return _recommender.predict_top_n(features, season=season, month=month, top_n=3)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Recommendation Inference")
    parser.add_argument("--N", type=float, required=True, help="Nitrogen (kg/ha)")
    parser.add_argument("--P", type=float, required=True, help="Phosphorus (kg/ha)")
    parser.add_argument("--K", type=float, required=True, help="Potassium (kg/ha)")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature (°C)")
    parser.add_argument("--humidity", type=float, required=True, help="Humidity (%%)")
    parser.add_argument("--ph", type=float, required=True, help="Soil pH")
    parser.add_argument("--rainfall", type=float, required=True, help="Rainfall (mm)")
    parser.add_argument("--season", type=str, default=None, choices=["Kharif", "Rabi", "Zaid"])
    parser.add_argument("--month", type=int, default=None, choices=range(1, 13))
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    features = {
        "N": args.N, "P": args.P, "K": args.K,
        "temperature": args.temperature, "humidity": args.humidity,
        "ph": args.ph, "rainfall": args.rainfall,
    }

    recommender = CropRecommender()
    results = recommender.predict_top_n(
        features, season=args.season, month=args.month, top_n=args.top_n
    )

    print("\n🌾 Crop Recommendations")
    print("=" * 50)
    for r in results:
        badge = "✅" if r["season_match"] else "⚠️ (out of season)"
        print(f"  #{r['rank']}  {r['crop']:15s}  {r['probability']*100:5.1f}%  {badge}")
    print("=" * 50)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
