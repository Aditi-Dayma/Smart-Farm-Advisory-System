# %% [markdown]
# # Module 4 & End-to-End Integration Demo: Smart Farm Advisory System
# 
# Simulates a complete agricultural lifecycle workflow for a farm located in
# Maharashtra, India (clay soil, Kharif season):
# 1. Crop Suitability Recommendation based on soil test (N, P, K, pH) & climate
# 2. Disease Detection on uploaded tomato leaf image
# 3. Governed Pesticide Advisory (chemical & biocontrol) with safety & weather gates
# 4. Daily Irrigation Scheduling with FAO-56 crop factor & rain-skip logic

# %%
import json
from datetime import datetime, date

# %% [markdown]
# ## 1. Module 1: Crop Suitability Selection

# %%
soil_test = {
    "N": 40.0,
    "P": 50.0,
    "K": 50.0,
    "temperature": 27.5,
    "humidity": 75.0,
    "ph": 6.8,
    "rainfall": 180.0,
}
season = "Kharif"
month = 7  # July

print("--- Step 1: Crop Suitability Input ---")
print(json.dumps(soil_test, indent=2))
print(f"Season: {season}, Month: {month}")

# Simulated inference output
recommended_crops = [
    {"rank": 1, "crop": "rice", "probability": 0.89, "season_match": True},
    {"rank": 2, "crop": "maize", "probability": 0.76, "season_match": True},
    {"rank": 3, "crop": "cotton", "probability": 0.61, "season_match": True},
]
print("\nRecommended Crops (Top 3):")
for r in recommended_crops:
    badge = "✅ Sowable Now" if r["season_match"] else "❌ Out of Season"
    print(f"  #{r['rank']}: {r['crop'].capitalize():10s} (Score: {r['probability']*100:.1f}%) {badge}")

# %% [markdown]
# ## 2. Module 2: Image Disease Diagnosis

# %%
detection_result = {
    "crop": "Tomato",
    "disease": "Early Blight",
    "confidence": 0.942,
    "pathogen": "Alternaria solani",
}
print("\n--- Step 2: Diagnostic Vision Output ---")
print(f"Leaf Image Classification: {detection_result['disease']} in {detection_result['crop']}")
print(f"Model Confidence: {detection_result['confidence']*100:.1f}%")

# %% [markdown]
# ## 3. Module 4: Governed Pesticide Advisory Rule Engine

# %%
# Weather check: morning spray window
current_weather = {
    "wind_speed_kmh": 6.2,
    "rain_next_6h_mm": 0.0,
    "temp_c": 28.0
}

safe_to_spray = current_weather["wind_speed_kmh"] <= 15.0 and current_weather["rain_next_6h_mm"] <= 0.5
print("\n--- Step 3: Governed Pesticide Advisory ---")
print(f"Weather Gate Passed: {safe_to_spray}")
print(f"Optimal Spray Timing: Early Morning (06:00 - 08:00 AM)")
print("\nRecommended Treatments:")
print("  [Chemical] Mancozeb 75% WP @ 2.5 g/L water (Interval: 7 days, PHI: 5 days)")
print("  [Biocontrol] Trichoderma viride 1% WP @ 5 g/L water (Interval: 7 days, PHI: 0 days)")
print("\n⚠️ DISCLAIMER: Decision support only. Consult certified agronomist before application.")

# %% [markdown]
# ## 4. Module 3: Irrigation Scheduling (FAO-56 Water Balance)

# %%
et0 = 4.8  # mm/day (Penman-Monteith)
kc = 1.15 # Mid-season crop factor for tomato
etc = et0 * kc
rain_forecast_24h = 0.0

irrigation_needed = (etc > 5.0) and (rain_forecast_24h < 10.0)
water_liters_per_ha = etc * 10000

print("\n--- Step 4: Daily Irrigation Advisory ---")
print(f"Reference ET0: {et0} mm/day | Crop ETc: {etc:.2f} mm/day")
print(f"Status: {'💧 IRRIGATE TODAY' if irrigation_needed else '✅ NO IRRIGATION NEEDED'}")
print(f"Water Application: {etc:.1f} mm ({int(water_liters_per_ha):,} Liters / hectare)")
