"""
Module 4 — Pesticide Advisory: Main Advisory Generator.

Combines disease detection results, weather forecast, and farm profile
to produce a governed pesticide advisory with safety disclaimers.

⚠️  DISCLAIMER: This is decision-support information ONLY.  Always
consult a certified agronomist before applying any pesticide.  Follow
product label instructions.  The developers assume NO liability for
crop damage, health effects, or environmental harm.

Usage::

    python -m src.module4_pesticide.advisory --disease "Early Blight" --crop "Tomato"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from src.module4_pesticide.lookup_table import (
    Treatment,
    get_treatments,
    get_organic_alternatives,
)
from src.module4_pesticide.timing_rules import SprayTimingAdvisor, TimingResult
from src.utils.common import setup_logging

logger = setup_logging("module4_pesticide.advisory")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DISCLAIMER = (
    "⚠️ DISCLAIMER: This is decision-support information ONLY.  "
    "Always consult a certified agronomist before applying any pesticide.  "
    "Follow product label instructions strictly.  "
    "The developers assume NO liability for crop damage, health effects, "
    "or environmental harm resulting from the use of this advisory."
)

SAFETY_EQUIPMENT = [
    "Long-sleeved shirt and trousers",
    "Chemical-resistant gloves",
    "Face mask or respirator",
    "Protective goggles",
    "Rubber boots",
    "Hat or head covering",
]


# ──────────────────────────────────────────────────────────────────────
# Advisory data class
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PesticideAdvisory:
    """Complete pesticide advisory output."""
    disease_identified: str
    crop: str
    confidence: float
    recommended_treatments: List[Dict[str, Any]]
    spray_now: bool
    timing_reason: str
    optimal_spray_time: str
    next_spray_window: Optional[datetime]
    disclaimer: str = DISCLAIMER
    safety_equipment: List[str] = field(default_factory=lambda: SAFETY_EQUIPMENT.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_identified": self.disease_identified,
            "crop": self.crop,
            "confidence": self.confidence,
            "recommended_treatments": self.recommended_treatments,
            "spray_now": self.spray_now,
            "timing_reason": self.timing_reason,
            "optimal_spray_time": self.optimal_spray_time,
            "next_spray_window": self.next_spray_window.isoformat() if self.next_spray_window else None,
            "disclaimer": self.disclaimer,
            "safety_equipment": self.safety_equipment,
        }

    def format_text(self) -> str:
        """Format as human-readable advisory text."""
        lines = [
            "",
            "=" * 65,
            "🧪  PESTICIDE ADVISORY",
            "=" * 65,
            "",
            f"  ⚠️  {self.disclaimer}",
            "",
            "-" * 65,
            f"  Disease  : {self.disease_identified}",
            f"  Crop     : {self.crop}",
            f"  Confidence: {self.confidence * 100:.1f}%",
            "-" * 65,
            "",
        ]

        # Spray timing
        status = "✅ SAFE TO SPRAY NOW" if self.spray_now else "❌ DO NOT SPRAY NOW"
        lines.append(f"  Spray Status  : {status}")
        lines.append(f"  Reason        : {self.timing_reason}")
        lines.append(f"  Optimal Time  : {self.optimal_spray_time}")
        if self.next_spray_window:
            lines.append(f"  Next Window   : {self.next_spray_window.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Treatments
        lines.append("  RECOMMENDED TREATMENTS:")
        lines.append("  " + "-" * 55)
        for i, t in enumerate(self.recommended_treatments, 1):
            ttype = "🧬 Biocontrol" if t.get("treatment_type") == "biocontrol" else "💊 Chemical"
            lines.append(f"  {i}. [{ttype}]  {t['product_name']}")
            lines.append(f"     Active: {t['active_ingredient']}")
            lines.append(f"     Dose  : {t['dose_per_litre']} per litre | {t['dose_per_hectare']} per hectare")
            lines.append(f"     Spray Vol: {t['spray_volume_litres_per_hectare']} L/ha")
            lines.append(f"     Interval : Every {t['spray_interval_days']} days (max {t['max_applications']} sprays)")
            lines.append(f"     PHI      : {t['pre_harvest_interval_days']} days before harvest")
            lines.append("")

        # Safety
        lines.append("  SAFETY EQUIPMENT REQUIRED:")
        for item in self.safety_equipment:
            lines.append(f"    • {item}")
        lines.append("")

        # Footer disclaimer
        lines.append("  " + "=" * 55)
        lines.append(f"  ⚠️  {self.disclaimer}")
        lines.append("  " + "=" * 55)

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main advisory function
# ──────────────────────────────────────────────────────────────────────


def generate_pesticide_advisory(
    disease_detection_result: Dict[str, Any],
    weather_forecast: Dict[str, Any],
    farm_profile: Optional[Dict[str, Any]] = None,
    last_spray_date: Optional[date] = None,
    prefer_organic: bool = False,
) -> PesticideAdvisory:
    """
    Generate a complete pesticide advisory.

    Parameters
    ----------
    disease_detection_result : dict
        Output from Module 2 with keys: crop, disease, confidence.
    weather_forecast : dict
        Current/forecast weather with keys: rain_next_hours, wind_speed, temperature.
    farm_profile : dict, optional
        Farm info with keys: expected_harvest_date, field_area_ha, etc.
    last_spray_date : date, optional
        Date of the last pesticide application.
    prefer_organic : bool
        If True, rank biocontrol options higher.

    Returns
    -------
    PesticideAdvisory object.
    """
    crop = disease_detection_result.get("crop", "Unknown")
    disease = disease_detection_result.get("disease", "Unknown")
    confidence = disease_detection_result.get("confidence", 0.0)

    farm_profile = farm_profile or {}

    logger.info("Generating advisory — Crop: %s, Disease: %s, Confidence: %.2f", crop, disease, confidence)

    # ── 1. Look up treatments ────────────────────────────────────────
    if prefer_organic:
        treatments = get_organic_alternatives(disease, crop)
        if not treatments:
            treatments = get_treatments(disease, crop)
            logger.info("No organic options found — falling back to all treatments")
    else:
        treatments = get_treatments(disease, crop)

    if not treatments:
        logger.warning("No treatments found for %s — %s", crop, disease)
        return PesticideAdvisory(
            disease_identified=disease,
            crop=crop,
            confidence=confidence,
            recommended_treatments=[],
            spray_now=False,
            timing_reason="No treatment data available for this disease",
            optimal_spray_time="N/A",
            next_spray_window=None,
        )

    # ── 2. Check spray timing ────────────────────────────────────────
    advisor = SprayTimingAdvisor()

    # Use first treatment's interval and PHI as defaults
    spray_interval = treatments[0].spray_interval_days
    phi_days = treatments[0].pre_harvest_interval_days
    expected_harvest = None
    if "expected_harvest_date" in farm_profile:
        try:
            expected_harvest = date.fromisoformat(str(farm_profile["expected_harvest_date"]))
        except (ValueError, TypeError):
            pass

    timing = advisor.full_timing_check(
        forecast=weather_forecast,
        last_spray_date=last_spray_date,
        spray_interval_days=spray_interval,
        expected_harvest_date=expected_harvest,
        phi_days=phi_days,
    )

    # ── 3. Rank treatments ───────────────────────────────────────────
    # If disease severity is low (confidence < 0.5), prefer biocontrol
    ranked_treatments = sorted(
        treatments,
        key=lambda t: (
            0 if (prefer_organic or confidence < 0.5) and t.treatment_type == "biocontrol"
            else 1,
            -t.max_applications,  # more applications = more flexible
        ),
    )

    treatment_dicts = [t.to_dict() for t in ranked_treatments]

    # ── 4. Assemble advisory ─────────────────────────────────────────
    advisory = PesticideAdvisory(
        disease_identified=disease,
        crop=crop,
        confidence=confidence,
        recommended_treatments=treatment_dicts,
        spray_now=timing.safe_to_spray,
        timing_reason=timing.reason,
        optimal_spray_time=timing.optimal_time,
        next_spray_window=timing.next_safe_window,
    )

    logger.info("Advisory generated — spray_now=%s, %d treatments", timing.safe_to_spray, len(treatment_dicts))
    return advisory


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pesticide Advisory Generator")
    parser.add_argument("--disease", type=str, required=True, help="Disease name")
    parser.add_argument("--crop", type=str, required=True, help="Crop name")
    parser.add_argument("--confidence", type=float, default=0.85, help="Detection confidence")
    parser.add_argument("--wind", type=float, default=5.0, help="Wind speed (km/h)")
    parser.add_argument("--temp", type=float, default=28.0, help="Temperature (°C)")
    parser.add_argument("--rain", type=float, default=0.0, help="Forecast rain (mm)")
    parser.add_argument("--organic", action="store_true", help="Prefer organic/biocontrol")
    args = parser.parse_args()

    disease_result = {
        "crop": args.crop,
        "disease": args.disease,
        "confidence": args.confidence,
    }
    forecast = {
        "wind_speed": args.wind,
        "temperature": args.temp,
        "rain_next_hours": args.rain,
    }

    advisory = generate_pesticide_advisory(
        disease_detection_result=disease_result,
        weather_forecast=forecast,
        prefer_organic=args.organic,
    )

    print(advisory.format_text())
    print("\n--- JSON ---")
    print(json.dumps(advisory.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
