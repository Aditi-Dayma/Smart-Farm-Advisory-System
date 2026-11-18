# %% [markdown]
# # Module 1: Crop Suitability — Exploratory Data Analysis & Feature Engineering
# 
# This notebook conducts exploratory data analysis on the 22-crop agricultural dataset,
# examines nutrient distributions (N, P, K), evaluates climate envelopes (temperature,
# humidity, rainfall, pH), inspects outliers, and demonstrates seasonal encoding.

# %%
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. Load Crop Recommendation & Calendar Data

# %%
PROJECT_ROOT = Path("..").resolve()
data_path = PROJECT_ROOT / "data" / "crop_calendar.csv"
print(f"Loading calendar from: {data_path}")

df_cal = pd.read_csv(data_path)
df_cal.head()

# %% [markdown]
# ## 2. Feature Ranges by Crop Group

# %%
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_cal, x="crop", y="water_requirement_mm_per_season", ax=ax, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("Seasonal Water Requirement (mm) Across Crops (ICAR Guidelines)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Outlier Analysis & IQR Capping Demonstration

# %%
# Synthetic demonstration of IQR bounding
np.random.seed(42)
sample_rf = np.concatenate([np.random.normal(120, 25, 200), [350, 400, 420]]) # extreme outliers
s = pd.Series(sample_rf)

q1 = s.quantile(0.25)
q3 = s.quantile(0.75)
iqr = q3 - q1
upper_cap = q3 + 1.5 * iqr

print(f"Q1: {q1:.1f}, Q3: {q3:.1f}, IQR: {iqr:.1f}, Upper Cap Threshold: {upper_cap:.1f}")
print(f"Values clipped above cap: {sum(s > upper_cap)}")
