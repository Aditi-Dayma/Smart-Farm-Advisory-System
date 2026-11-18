"""
FAO-56 Penman-Monteith Reference Evapotranspiration (ET₀) Module.

Implements the full FAO-56 methodology for computing reference crop
evapotranspiration and crop water requirements (ETc = ET₀ × Kc).

Key references
--------------
* Allen, R.G., et al. (1998). "Crop Evapotranspiration — Guidelines for
  Computing Crop Water Requirements." FAO Irrigation & Drainage Paper 56.

NOTE: This module is part of the Smart Farm Advisory System.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════
# 1.  CROP COEFFICIENT (Kc) TABLE — FAO-56 Reference Values
# ══════════════════════════════════════════════════════════════════════
#
# Each crop has four growth stages:
#   • initial   (Kc_ini)   — planting to ~10 % ground cover
#   • development (Kc_dev) — 10 % ground cover to effective full cover
#   • mid-season  (Kc_mid) — effective full cover to start of maturity
#   • late-season  (Kc_end)— maturity to harvest / senescence
#
# duration_days = [ini, dev, mid, late] — typical stage lengths (days)
# kc_values     = [Kc_ini, Kc_dev, Kc_mid, Kc_end]
#
# Kc_dev is not a single constant; it transitions linearly from Kc_ini
# to Kc_mid.  For look-up convenience we store the mid-point of that
# transition.  The ``interpolate_kc`` function handles the linear ramp.
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CropKc:
    """Crop coefficient parameters for one crop."""

    name: str
    kc_ini: float
    kc_mid: float
    kc_end: float
    duration_days: Tuple[int, int, int, int]  # (ini, dev, mid, late)

    @property
    def total_season_days(self) -> int:
        return sum(self.duration_days)


# 22 crops — values sourced / adapted from FAO-56 Tables 11 & 12
KC_TABLE: Dict[str, CropKc] = {
    "rice": CropKc("rice", kc_ini=1.05, kc_mid=1.20, kc_end=0.90,
                    duration_days=(30, 30, 60, 30)),
    "wheat": CropKc("wheat", kc_ini=0.30, kc_mid=1.15, kc_end=0.25,
                     duration_days=(20, 25, 60, 30)),
    "maize": CropKc("maize", kc_ini=0.30, kc_mid=1.20, kc_end=0.60,
                     duration_days=(20, 35, 40, 30)),
    "sugarcane": CropKc("sugarcane", kc_ini=0.40, kc_mid=1.25, kc_end=0.75,
                         duration_days=(35, 60, 190, 120)),
    "cotton": CropKc("cotton", kc_ini=0.35, kc_mid=1.20, kc_end=0.60,
                      duration_days=(30, 50, 55, 45)),
    "soybean": CropKc("soybean", kc_ini=0.40, kc_mid=1.15, kc_end=0.50,
                       duration_days=(20, 30, 60, 25)),
    "groundnut": CropKc("groundnut", kc_ini=0.40, kc_mid=1.15, kc_end=0.60,
                          duration_days=(25, 35, 45, 25)),
    "sunflower": CropKc("sunflower", kc_ini=0.35, kc_mid=1.10, kc_end=0.35,
                          duration_days=(25, 35, 45, 25)),
    "mustard": CropKc("mustard", kc_ini=0.35, kc_mid=1.10, kc_end=0.35,
                       duration_days=(20, 30, 35, 20)),
    "barley": CropKc("barley", kc_ini=0.30, kc_mid=1.15, kc_end=0.25,
                      duration_days=(15, 25, 50, 30)),
    "millet": CropKc("millet", kc_ini=0.30, kc_mid=1.00, kc_end=0.30,
                      duration_days=(15, 25, 40, 25)),
    "sorghum": CropKc("sorghum", kc_ini=0.30, kc_mid=1.10, kc_end=0.55,
                       duration_days=(20, 30, 40, 30)),
    "chickpea": CropKc("chickpea", kc_ini=0.40, kc_mid=1.00, kc_end=0.35,
                        duration_days=(25, 35, 40, 20)),
    "lentil": CropKc("lentil", kc_ini=0.40, kc_mid=1.10, kc_end=0.30,
                      duration_days=(20, 30, 60, 25)),
    "potato": CropKc("potato", kc_ini=0.50, kc_mid=1.15, kc_end=0.75,
                      duration_days=(25, 30, 45, 30)),
    "tomato": CropKc("tomato", kc_ini=0.60, kc_mid=1.15, kc_end=0.80,
                      duration_days=(30, 40, 40, 25)),
    "onion": CropKc("onion", kc_ini=0.70, kc_mid=1.05, kc_end=0.75,
                     duration_days=(15, 25, 70, 40)),
    "cabbage": CropKc("cabbage", kc_ini=0.70, kc_mid=1.05, kc_end=0.95,
                       duration_days=(20, 25, 60, 15)),
    "banana": CropKc("banana", kc_ini=0.50, kc_mid=1.10, kc_end=1.00,
                      duration_days=(120, 60, 180, 60)),
    "mango": CropKc("mango", kc_ini=0.50, kc_mid=0.85, kc_end=0.60,
                     duration_days=(60, 90, 120, 60)),
    "coconut": CropKc("coconut", kc_ini=0.50, kc_mid=1.00, kc_end=1.00,
                       duration_days=(365, 0, 0, 0)),  # perennial — single Kc ~1.0
    "tea": CropKc("tea", kc_ini=0.95, kc_mid=1.00, kc_end=1.00,
                   duration_days=(365, 0, 0, 0)),  # perennial
}


def list_crops() -> List[str]:
    """Return the names of all crops with Kc data."""
    return sorted(KC_TABLE.keys())


def get_crop_kc(crop: str) -> CropKc:
    """Look up Kc parameters for *crop* (case-insensitive).

    Raises
    ------
    KeyError
        If the crop is not in the table.
    """
    key = crop.strip().lower()
    if key not in KC_TABLE:
        raise KeyError(
            f"Unknown crop '{crop}'.  Available: {', '.join(list_crops())}"
        )
    return KC_TABLE[key]


def interpolate_kc(crop: str, days_since_planting: int) -> float:
    """Return the crop coefficient Kc for a given day after planting.

    Uses linear interpolation during the *development* stage (from
    ``Kc_ini`` to ``Kc_mid``) and during the *late* stage (from
    ``Kc_mid`` to ``Kc_end``).  During *initial* and *mid-season* the
    value is constant.

    Parameters
    ----------
    crop:
        Crop name (case-insensitive).
    days_since_planting:
        Number of days since planting / sowing (day 0 = planting day).

    Returns
    -------
    float
        Interpolated Kc value (dimensionless).
    """
    ckc = get_crop_kc(crop)
    d_ini, d_dev, d_mid, d_late = ckc.duration_days

    # Perennial crops (dev/mid/late = 0) — return constant Kc_mid
    if d_dev == 0 and d_mid == 0 and d_late == 0:
        return ckc.kc_mid

    # Clamp to total season length
    dsp = max(0, days_since_planting)

    # Stage boundaries (cumulative days)
    end_ini = d_ini
    end_dev = end_ini + d_dev
    end_mid = end_dev + d_mid
    end_late = end_mid + d_late  # == total_season_days

    if dsp <= end_ini:
        # ── Initial stage: constant Kc_ini ──
        return ckc.kc_ini

    elif dsp <= end_dev:
        # ── Development stage: linear ramp Kc_ini → Kc_mid ──
        fraction = (dsp - end_ini) / d_dev
        return ckc.kc_ini + fraction * (ckc.kc_mid - ckc.kc_ini)

    elif dsp <= end_mid:
        # ── Mid-season stage: constant Kc_mid ──
        return ckc.kc_mid

    elif dsp <= end_late:
        # ── Late-season stage: linear decline Kc_mid → Kc_end ──
        fraction = (dsp - end_mid) / d_late
        return ckc.kc_mid + fraction * (ckc.kc_end - ckc.kc_mid)

    else:
        # Past harvest — return Kc_end
        return ckc.kc_end


# ══════════════════════════════════════════════════════════════════════
# 2.  FAO-56 PENMAN-MONTEITH REFERENCE EVAPOTRANSPIRATION (ET₀)
# ══════════════════════════════════════════════════════════════════════


def compute_et0(
    temp_min: float,
    temp_max: float,
    temp_mean: Optional[float] = None,
    rh_mean: float = 50.0,
    wind_speed_2m: float = 2.0,
    solar_radiation: Optional[float] = None,
    sunshine_hours: Optional[float] = None,
    day_of_year: int = 1,
    latitude_deg: float = 20.0,
    altitude_m: float = 0.0,
) -> float:
    """Compute daily reference evapotranspiration ET₀ (mm day⁻¹).

    Implements **Equation 6** of FAO-56 (Penman-Monteith):

    .. math::

        ET_0 = \\frac{0.408 \\, \\Delta (R_n - G)
                      + \\gamma \\, \\frac{900}{T + 273} \\, u_2 \\, (e_s - e_a)}
                     {\\Delta + \\gamma (1 + 0.34 \\, u_2)}

    Parameters
    ----------
    temp_min : float
        Minimum daily temperature (°C).
    temp_max : float
        Maximum daily temperature (°C).
    temp_mean : float, optional
        Mean daily temperature (°C).  Defaults to ``(temp_min + temp_max) / 2``.
    rh_mean : float
        Mean daily relative humidity (%).
    wind_speed_2m : float
        Wind speed at 2 m height (m s⁻¹).  If your data is at 10 m, divide
        by ≈ 1.33 (FAO-56 Eq. 47) before passing.
    solar_radiation : float, optional
        Incoming shortwave solar radiation (MJ m⁻² day⁻¹).  If ``None``,
        estimated from *sunshine_hours* or the Hargreaves radiation formula.
    sunshine_hours : float, optional
        Actual daily sunshine hours (used only when *solar_radiation* is
        ``None``).
    day_of_year : int
        Julian day of year (1–365/366).
    latitude_deg : float
        Latitude in decimal degrees (positive = N, negative = S).
    altitude_m : float
        Elevation above sea level (m).

    Returns
    -------
    float
        Reference evapotranspiration ET₀ in mm day⁻¹.

    Notes
    -----
    All intermediate computations follow the step-by-step procedure in
    FAO-56 Chapter 3 and Example 17.
    """

    # ----- Step 0: Mean temperature -----
    if temp_mean is None:
        temp_mean = (temp_min + temp_max) / 2.0

    T = temp_mean  # shorthand (°C)

    # ----- Step 1: Atmospheric pressure (kPa) -----
    # FAO-56 Eq. 7:  P = 101.3 × ((293 − 0.0065 z) / 293)^5.26
    P = 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26

    # ----- Step 2: Psychrometric constant γ (kPa °C⁻¹) -----
    # FAO-56 Eq. 8:  γ = (cp × P) / (ε × λ)
    #   cp = 1.013e-3 MJ kg⁻¹ °C⁻¹, ε = 0.622, λ = 2.45 MJ kg⁻¹
    gamma = 0.665e-3 * P  # simplified form from FAO-56

    # ----- Step 3: Saturation vapour pressure e°(T) (kPa) -----
    # FAO-56 Eq. 11:  e°(T) = 0.6108 × exp(17.27T / (T + 237.3))
    def _esat(t: float) -> float:
        return 0.6108 * math.exp((17.27 * t) / (t + 237.3))

    e_sat_tmax = _esat(temp_max)
    e_sat_tmin = _esat(temp_min)

    # Mean saturation vapour pressure (FAO-56 Eq. 12)
    e_s = (e_sat_tmax + e_sat_tmin) / 2.0

    # ----- Step 4: Actual vapour pressure ea (kPa) -----
    # Approximation from mean RH (FAO-56 Eq. 19):
    #   ea = (rh_mean / 100) × e_s
    e_a = (rh_mean / 100.0) * e_s

    # Vapour pressure deficit
    vpd = e_s - e_a  # kPa

    # ----- Step 5: Slope of saturation vapour pressure curve Δ (kPa °C⁻¹) -----
    # FAO-56 Eq. 13:
    delta = (4098.0 * _esat(T)) / ((T + 237.3) ** 2)

    # ----- Step 6: Solar geometry -----
    # Latitude in radians
    phi = math.radians(latitude_deg)

    # Solar declination δ (FAO-56 Eq. 24)
    d_r = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    solar_decl = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)

    # Sunset hour angle ωs (FAO-56 Eq. 25)
    ws_arg = -math.tan(phi) * math.tan(solar_decl)
    ws_arg = max(-1.0, min(1.0, ws_arg))  # clamp for polar regions
    omega_s = math.acos(ws_arg)

    # Extraterrestrial radiation Ra (MJ m⁻² day⁻¹) (FAO-56 Eq. 21)
    Gsc = 0.0820  # solar constant (MJ m⁻² min⁻¹)
    Ra = (
        (24.0 * 60.0 / math.pi)
        * Gsc
        * d_r
        * (
            omega_s * math.sin(phi) * math.sin(solar_decl)
            + math.cos(phi) * math.cos(solar_decl) * math.sin(omega_s)
        )
    )
    Ra = max(Ra, 0.0)

    # Daylight hours N (FAO-56 Eq. 34)
    N = (24.0 / math.pi) * omega_s

    # ----- Step 7: Solar (shortwave) radiation Rs -----
    if solar_radiation is not None:
        Rs = solar_radiation
    elif sunshine_hours is not None and N > 0:
        # Ångström formula (FAO-56 Eq. 35):
        #   Rs = (a_s + b_s × n/N) × Ra
        a_s, b_s = 0.25, 0.50  # recommended defaults
        Rs = (a_s + b_s * (sunshine_hours / N)) * Ra
    else:
        # Hargreaves radiation formula (FAO-56 Eq. 50)
        #   Rs = kRs × sqrt(Tmax − Tmin) × Ra
        # kRs ≈ 0.16 for interior, 0.19 for coastal
        kRs = 0.17
        Rs = kRs * math.sqrt(max(temp_max - temp_min, 0.0)) * Ra

    # Clear-sky radiation Rso (FAO-56 Eq. 37)
    Rso = (0.75 + 2e-5 * altitude_m) * Ra
    Rso = max(Rso, 0.001)  # avoid division by zero

    # ----- Step 8: Net radiation Rn -----
    # Net shortwave Rns (FAO-56 Eq. 38)   α = 0.23 (grass reference)
    alpha = 0.23
    Rns = (1.0 - alpha) * Rs

    # Net longwave Rnl (FAO-56 Eq. 39)
    sigma = 4.903e-9  # Stefan-Boltzmann (MJ m⁻² day⁻¹ K⁻⁴)
    Tmax_K4 = (temp_max + 273.16) ** 4
    Tmin_K4 = (temp_min + 273.16) ** 4

    Rs_over_Rso = min(Rs / Rso, 1.0)

    Rnl = (
        sigma
        * ((Tmax_K4 + Tmin_K4) / 2.0)
        * (0.34 - 0.14 * math.sqrt(max(e_a, 0.0)))
        * (1.35 * Rs_over_Rso - 0.35)
    )

    # Net radiation (FAO-56 Eq. 40)
    Rn = Rns - Rnl

    # ----- Step 9: Soil heat flux G -----
    # For daily calculations, FAO-56 recommends G ≈ 0.
    G = 0.0

    # ----- Step 10: FAO-56 Penman-Monteith (Eq. 6) -----
    numerator = (
        0.408 * delta * (Rn - G)
        + gamma * (900.0 / (T + 273.0)) * wind_speed_2m * vpd
    )
    denominator = delta + gamma * (1.0 + 0.34 * wind_speed_2m)

    et0 = numerator / denominator

    # ET₀ should not be negative
    return max(et0, 0.0)


# ══════════════════════════════════════════════════════════════════════
# 3.  CROP WATER REQUIREMENT  (ETc = ET₀ × Kc)
# ══════════════════════════════════════════════════════════════════════


def calculate_etc(
    et0: float,
    crop: str,
    days_since_planting: int,
) -> Dict[str, Any]:
    """Calculate crop evapotranspiration ETc for a specific day.

    Parameters
    ----------
    et0 : float
        Reference evapotranspiration ET₀ (mm day⁻¹).
    crop : str
        Crop name (case-insensitive).
    days_since_planting : int
        Days elapsed since planting.

    Returns
    -------
    dict
        ``et0`` (mm day⁻¹), ``kc`` (dimensionless), ``etc`` (mm day⁻¹),
        ``growth_stage`` (str), ``crop`` (str).
    """
    kc = interpolate_kc(crop, days_since_planting)
    etc = et0 * kc

    # Determine human-readable growth-stage name
    ckc = get_crop_kc(crop)
    stage = _stage_name(ckc, days_since_planting)

    return {
        "et0": round(et0, 3),
        "kc": round(kc, 3),
        "etc": round(etc, 3),
        "growth_stage": stage,
        "crop": crop.lower(),
    }


def _stage_name(ckc: CropKc, dsp: int) -> str:
    """Human-readable growth-stage label for a given day after planting."""
    d_ini, d_dev, d_mid, d_late = ckc.duration_days

    # Perennial
    if d_dev == 0 and d_mid == 0 and d_late == 0:
        return "perennial"

    cum = 0
    for stage, dur in zip(
        ["initial", "development", "mid-season", "late-season"],
        [d_ini, d_dev, d_mid, d_late],
    ):
        cum += dur
        if dsp <= cum:
            return stage

    return "post-harvest"


# ══════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: a typical summer day in central India
    et0 = compute_et0(
        temp_min=25.0,
        temp_max=38.0,
        rh_mean=45.0,
        wind_speed_2m=1.8,
        solar_radiation=22.5,  # MJ m⁻² day⁻¹
        day_of_year=150,       # ~end of May
        latitude_deg=20.0,
        altitude_m=300.0,
    )
    print(f"ET₀ = {et0:.2f} mm/day")

    # Crop water requirement for wheat at day 60
    result = calculate_etc(et0, "wheat", days_since_planting=60)
    print(f"ETc = {result['etc']:.2f} mm/day  (Kc={result['kc']:.2f}, "
          f"stage={result['growth_stage']})")

    # Print Kc curve for maize
    print("\nMaize Kc curve:")
    for d in range(0, 130, 5):
        kc = interpolate_kc("maize", d)
        bar = "█" * int(kc * 20)
        print(f"  Day {d:3d}: Kc = {kc:.2f}  {bar}")
