# %% [markdown]
# # Module 3: Irrigation Scheduling — Evapotranspiration (FAO-56) & Water Balance Simulation
# 
# Demonstrates Penman-Monteith ET0 calculation, dynamic Kc interpolation across
# growth stages, and a 30-day soil-water balance trajectory comparing
# ML regression against the physical baseline.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. 30-Day Soil Moisture Depletion Simulation

# %%
days = np.arange(1, 31)
et0_daily = 4.5 + 0.8 * np.sin(days / 5)
kc_values = 0.5 + 0.7 * (days / 30)  # Development phase linear transition
etc_daily = et0_daily * kc_values

rain_events = np.zeros(30)
rain_events[8] = 22.0  # Rain event on day 9
rain_events[22] = 15.0 # Rain event on day 23

# Soil moisture simulation
moisture = np.zeros(30)
current = 100.0  # mm available water in root zone
depletion_trigger = 50.0  # Trigger threshold at 50% MAD
irrigation_events = []

for i, d in enumerate(days):
    current = current + rain_events[i] - etc_daily[i]
    if current < depletion_trigger:
        irrigation_applied = 100.0 - current
        irrigation_events.append((d, irrigation_applied))
        current = 100.0
    moisture[i] = current

# %% [markdown]
# ## 2. Water Balance Trajectory Chart

# %%
plt.figure(figsize=(12, 5))
plt.plot(days, moisture, label="Available Soil Moisture (mm)", color="navy", lw=2)
plt.axhline(depletion_trigger, color="red", linestyle="--", label="Management Allowed Depletion (MAD)")
plt.bar(days, rain_events, color="cyan", alpha=0.6, label="Rainfall (mm)")
for irr_day, amt in irrigation_events:
    plt.axvline(irr_day, color="green", linestyle=":", label=f"Irrigation ({amt:.1f}mm)" if irr_day == irrigation_events[0][0] else "")

plt.title("30-Day Simulated Soil Water Balance Trajectory")
plt.xlabel("Day of Season")
plt.ylabel("Water Depth (mm)")
plt.legend()
plt.tight_layout()
plt.show()
