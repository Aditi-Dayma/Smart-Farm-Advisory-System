"""
Module 4 — Pesticide Advisory: Treatment Lookup Table.

Comprehensive database of disease → treatment mappings from official
recommendations (State Agriculture University Package of Practices,
product labels, and ICAR guidelines).

⚠️  DISCLAIMER: This is decision-support information only.  Actual
pesticide application must follow current product labels and be guided
by a certified agronomist.  The developers assume no liability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# Treatment data class
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Treatment:
    """A single pesticide/biocontrol treatment recommendation."""
    treatment_type: str           # "chemical" or "biocontrol"
    product_name: str             # Common trade name
    active_ingredient: str        # Chemical / biological agent
    dose_per_litre: str           # g or ml per litre of water
    dose_per_hectare: str         # g or ml per hectare
    spray_volume_litres_per_hectare: int  # recommended spray volume
    spray_interval_days: int      # minimum days between sprays
    max_applications: int         # max sprays per season
    pre_harvest_interval_days: int  # PHI — days before harvest to stop
    safety_precautions: List[str] = field(default_factory=list)
    source: str = "State Agriculture University Package of Practices"

    def to_dict(self) -> dict:
        return {
            "treatment_type": self.treatment_type,
            "product_name": self.product_name,
            "active_ingredient": self.active_ingredient,
            "dose_per_litre": self.dose_per_litre,
            "dose_per_hectare": self.dose_per_hectare,
            "spray_volume_litres_per_hectare": self.spray_volume_litres_per_hectare,
            "spray_interval_days": self.spray_interval_days,
            "max_applications": self.max_applications,
            "pre_harvest_interval_days": self.pre_harvest_interval_days,
            "safety_precautions": self.safety_precautions,
            "source": self.source,
        }


# ──────────────────────────────────────────────────────────────────────
# Standard safety precautions
# ──────────────────────────────────────────────────────────────────────

STANDARD_SAFETY = [
    "Wear protective clothing, gloves, and face mask during application",
    "Do not spray during windy conditions (>15 km/h)",
    "Wash hands and face thoroughly after spraying",
    "Keep pesticides away from children, animals, and water sources",
    "Dispose of empty containers safely — do not reuse",
    "Observe the pre-harvest interval (PHI) strictly",
]

ORGANIC_SAFETY = [
    "Wear gloves during preparation and application",
    "Prepare fresh solution for each application",
    "Store bio-agents in cool, dry place away from sunlight",
]


# ──────────────────────────────────────────────────────────────────────
# PESTICIDE DATABASE
# ──────────────────────────────────────────────────────────────────────
# Key: (crop, disease) → lower-cased and normalised
# Each disease mapped to 2–3 treatment options (chemical + biocontrol)
# ──────────────────────────────────────────────────────────────────────

PESTICIDE_DATABASE: Dict[str, List[Treatment]] = {

    # ═══════════════════════ TOMATO ═══════════════════════════════════

    "tomato_early_blight": [
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 6, 5, STANDARD_SAFETY,
                  "Product Label — Mancozeb 75% WP"),
        Treatment("chemical", "Chlorothalonil 75% WP", "Chlorothalonil",
                  "2.0 g/L", "500 g/ha", 500, 10, 4, 7, STANDARD_SAFETY,
                  "ICAR Package of Practices"),
        Treatment("biocontrol", "Trichoderma viride", "Trichoderma viride (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY,
                  "State Agriculture University — Organic Package"),
    ],

    "tomato_late_blight": [
        Treatment("chemical", "Metalaxyl + Mancozeb", "Metalaxyl 8% + Mancozeb 64% WP",
                  "2.5 g/L", "625 g/ha", 500, 10, 3, 14, STANDARD_SAFETY,
                  "ICAR Package of Practices"),
        Treatment("chemical", "Cymoxanil + Mancozeb", "Cymoxanil 8% + Mancozeb 64% WP",
                  "3.0 g/L", "750 g/ha", 500, 7, 4, 7, STANDARD_SAFETY),
        Treatment("biocontrol", "Pseudomonas fluorescens", "Pseudomonas fluorescens (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    "tomato_leaf_mold": [
        Treatment("chemical", "Copper Oxychloride 50% WP", "Copper Oxychloride",
                  "3.0 g/L", "750 g/ha", 500, 10, 4, 7, STANDARD_SAFETY),
        Treatment("chemical", "Hexaconazole 5% EC", "Hexaconazole",
                  "1.0 ml/L", "500 ml/ha", 500, 15, 3, 14, STANDARD_SAFETY),
        Treatment("biocontrol", "Trichoderma harzianum", "Trichoderma harzianum (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    "tomato_septoria_leaf_spot": [
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 6, 5, STANDARD_SAFETY),
        Treatment("chemical", "Copper Hydroxide 77% WP", "Copper Hydroxide",
                  "2.0 g/L", "500 g/ha", 500, 7, 5, 3, STANDARD_SAFETY),
        Treatment("biocontrol", "Bacillus subtilis", "Bacillus subtilis",
                  "5 g/L", "2.5 kg/ha", 500, 7, 10, 0, ORGANIC_SAFETY),
    ],

    "tomato_spider_mites": [
        Treatment("chemical", "Dicofol 18.5% EC", "Dicofol",
                  "2.5 ml/L", "1250 ml/ha", 500, 14, 2, 7, STANDARD_SAFETY),
        Treatment("chemical", "Abamectin 1.9% EC", "Abamectin",
                  "0.5 ml/L", "250 ml/ha", 500, 10, 2, 3, STANDARD_SAFETY),
        Treatment("biocontrol", "Neem Oil (Azadirachtin)", "Azadirachtin 0.03% (Neem)",
                  "5 ml/L", "2500 ml/ha", 500, 7, 10, 0, ORGANIC_SAFETY),
    ],

    "tomato_target_spot": [
        Treatment("chemical", "Azoxystrobin 23% SC", "Azoxystrobin",
                  "1.0 ml/L", "500 ml/ha", 500, 10, 3, 5, STANDARD_SAFETY),
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 6, 5, STANDARD_SAFETY),
        Treatment("biocontrol", "Trichoderma viride", "Trichoderma viride (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    "tomato_mosaic_virus": [
        Treatment("chemical", "Imidacloprid 17.8% SL (vector control)", "Imidacloprid",
                  "0.3 ml/L", "150 ml/ha", 500, 15, 2, 14,
                  STANDARD_SAFETY + ["Targets whitefly/aphid vectors — no direct cure for virus"]),
        Treatment("biocontrol", "Neem Oil", "Azadirachtin 0.03%",
                  "5 ml/L", "2500 ml/ha", 500, 7, 10, 0,
                  ORGANIC_SAFETY + ["Remove and destroy infected plants promptly"]),
    ],

    "tomato_yellow_leaf_curl": [
        Treatment("chemical", "Thiamethoxam 25% WG (vector control)", "Thiamethoxam",
                  "0.3 g/L", "150 g/ha", 500, 15, 3, 14,
                  STANDARD_SAFETY + ["Targets whitefly vectors — no direct cure for virus"]),
        Treatment("biocontrol", "Verticillium lecanii", "Verticillium lecanii (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 10, 6, 0,
                  ORGANIC_SAFETY + ["Use yellow sticky traps for whitefly monitoring"]),
    ],

    "tomato_bacterial_spot": [
        Treatment("chemical", "Streptomycin Sulphate + Tetracycline", "Streptomycin + Tetracycline",
                  "0.5 g/L", "250 g/ha", 500, 10, 3, 14, STANDARD_SAFETY),
        Treatment("chemical", "Copper Oxychloride 50% WP", "Copper Oxychloride",
                  "3.0 g/L", "750 g/ha", 500, 7, 5, 7, STANDARD_SAFETY),
        Treatment("biocontrol", "Pseudomonas fluorescens", "Pseudomonas fluorescens (1% WP)",
                  "10 g/L", "5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    # ═══════════════════════ POTATO ═══════════════════════════════════

    "potato_early_blight": [
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 6, 7, STANDARD_SAFETY),
        Treatment("chemical", "Propineb 70% WP", "Propineb",
                  "3.0 g/L", "750 g/ha", 500, 7, 5, 7, STANDARD_SAFETY),
        Treatment("biocontrol", "Trichoderma viride", "Trichoderma viride (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    "potato_late_blight": [
        Treatment("chemical", "Metalaxyl + Mancozeb", "Metalaxyl 8% + Mancozeb 64% WP",
                  "2.5 g/L", "625 g/ha", 500, 10, 3, 14, STANDARD_SAFETY,
                  "ICAR Package of Practices"),
        Treatment("chemical", "Fenamidone + Mancozeb", "Fenamidone 10% + Mancozeb 50% WDG",
                  "3.0 g/L", "750 g/ha", 500, 7, 3, 10, STANDARD_SAFETY),
        Treatment("biocontrol", "Pseudomonas fluorescens", "Pseudomonas fluorescens (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    # ═══════════════════════ CORN (MAIZE) ════════════════════════════

    "corn_common_rust": [
        Treatment("chemical", "Propiconazole 25% EC", "Propiconazole",
                  "1.0 ml/L", "500 ml/ha", 500, 15, 2, 21, STANDARD_SAFETY),
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 5, 5, STANDARD_SAFETY),
        Treatment("biocontrol", "Trichoderma harzianum", "Trichoderma harzianum (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 10, 6, 0, ORGANIC_SAFETY),
    ],

    "corn_gray_leaf_spot": [
        Treatment("chemical", "Azoxystrobin 23% SC", "Azoxystrobin",
                  "1.0 ml/L", "500 ml/ha", 500, 14, 2, 7, STANDARD_SAFETY),
        Treatment("chemical", "Tebuconazole 25.9% EC", "Tebuconazole",
                  "1.0 ml/L", "500 ml/ha", 500, 14, 2, 21, STANDARD_SAFETY),
        Treatment("biocontrol", "Pseudomonas fluorescens", "Pseudomonas fluorescens (1% WP)",
                  "10 g/L", "5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    "corn_northern_leaf_blight": [
        Treatment("chemical", "Propiconazole 25% EC", "Propiconazole",
                  "1.0 ml/L", "500 ml/ha", 500, 15, 2, 21, STANDARD_SAFETY),
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 5, 5, STANDARD_SAFETY),
        Treatment("biocontrol", "Trichoderma viride", "Trichoderma viride (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    # ═══════════════════════ GRAPE ═══════════════════════════════════

    "grape_black_rot": [
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 5, 21, STANDARD_SAFETY),
        Treatment("chemical", "Myclobutanil 10% WP", "Myclobutanil",
                  "1.0 g/L", "250 g/ha", 500, 10, 3, 14, STANDARD_SAFETY),
        Treatment("biocontrol", "Bacillus subtilis", "Bacillus subtilis",
                  "5 g/L", "2.5 kg/ha", 500, 7, 10, 0, ORGANIC_SAFETY),
    ],

    "grape_esca": [
        Treatment("chemical", "Thiophanate-methyl 70% WP", "Thiophanate-methyl",
                  "1.5 g/L", "375 g/ha", 500, 14, 2, 21, STANDARD_SAFETY,
                  "Applied as pruning wound protection"),
        Treatment("biocontrol", "Trichoderma harzianum", "Trichoderma harzianum (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 14, 4, 0,
                  ORGANIC_SAFETY + ["Apply to pruning wounds as paste"]),
    ],

    "grape_leaf_blight": [
        Treatment("chemical", "Copper Oxychloride 50% WP", "Copper Oxychloride",
                  "3.0 g/L", "750 g/ha", 500, 10, 4, 14, STANDARD_SAFETY),
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 5, 21, STANDARD_SAFETY),
        Treatment("biocontrol", "Pseudomonas fluorescens", "Pseudomonas fluorescens (1% WP)",
                  "10 g/L", "5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    # ═══════════════════════ APPLE ═══════════════════════════════════

    "apple_apple_scab": [
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 6, 21, STANDARD_SAFETY,
                  "ICAR Package of Practices — Apple"),
        Treatment("chemical", "Dodine 65% WP", "Dodine",
                  "1.0 g/L", "250 g/ha", 500, 10, 3, 14, STANDARD_SAFETY),
        Treatment("biocontrol", "Bacillus subtilis", "Bacillus subtilis",
                  "5 g/L", "2.5 kg/ha", 500, 7, 10, 0, ORGANIC_SAFETY),
    ],

    "apple_black_rot": [
        Treatment("chemical", "Captan 50% WP", "Captan",
                  "2.5 g/L", "625 g/ha", 500, 10, 4, 7, STANDARD_SAFETY),
        Treatment("chemical", "Thiophanate-methyl 70% WP", "Thiophanate-methyl",
                  "1.0 g/L", "250 g/ha", 500, 14, 3, 14, STANDARD_SAFETY),
        Treatment("biocontrol", "Trichoderma viride", "Trichoderma viride (1% WP)",
                  "5 g/L", "2.5 kg/ha", 500, 7, 8, 0, ORGANIC_SAFETY),
    ],

    "apple_cedar_apple_rust": [
        Treatment("chemical", "Myclobutanil 10% WP", "Myclobutanil",
                  "1.0 g/L", "250 g/ha", 500, 10, 3, 14, STANDARD_SAFETY),
        Treatment("chemical", "Mancozeb 75% WP", "Mancozeb",
                  "2.5 g/L", "625 g/ha", 500, 7, 5, 21, STANDARD_SAFETY),
        Treatment("biocontrol", "Neem Oil", "Azadirachtin 0.03%",
                  "5 ml/L", "2500 ml/ha", 500, 7, 10, 0,
                  ORGANIC_SAFETY + ["Also remove nearby cedar/juniper hosts if possible"]),
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Lookup functions
# ──────────────────────────────────────────────────────────────────────


def _normalise_key(crop: str, disease: str) -> str:
    """Normalise crop + disease into a lookup key."""
    key = f"{crop}_{disease}".lower()
    key = key.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    # Handle PlantVillage naming: "Tomato___Early_Blight" → "tomato_early_blight"
    key = key.replace("___", "_").replace("__", "_")
    return key


def get_treatments(disease_name: str, crop_name: str = "") -> List[Treatment]:
    """
    Look up treatments for a given disease.

    Parameters
    ----------
    disease_name : str
        Disease name (e.g., "Early Blight", "Late_Blight", "Tomato___Early_Blight").
    crop_name : str, optional
        Crop name (e.g., "Tomato"). If empty, extracted from disease_name.

    Returns
    -------
    List of Treatment objects, or empty list if disease not found.
    """
    # Try to parse PlantVillage format "Crop___Disease"
    if "___" in disease_name or "__" in disease_name:
        parts = disease_name.replace("___", "__").split("__")
        if len(parts) >= 2:
            crop_name = parts[0].strip()
            disease_name = parts[1].strip()

    key = _normalise_key(crop_name, disease_name)

    # Try exact match first
    if key in PESTICIDE_DATABASE:
        return PESTICIDE_DATABASE[key]

    # Fuzzy match: try partial key matching
    for db_key, treatments in PESTICIDE_DATABASE.items():
        if disease_name.lower().replace(" ", "_") in db_key:
            return treatments

    return []


def get_organic_alternatives(disease_name: str, crop_name: str = "") -> List[Treatment]:
    """Return only biocontrol/organic treatments for a disease."""
    all_treatments = get_treatments(disease_name, crop_name)
    return [t for t in all_treatments if t.treatment_type == "biocontrol"]


def get_all_diseases() -> List[str]:
    """Return list of all supported diseases."""
    return sorted(PESTICIDE_DATABASE.keys())
