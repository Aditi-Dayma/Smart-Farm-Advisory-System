# %% [markdown]
# # Module 2: Plant Disease Detection — Dataset Inspection & Augmentation Analysis
# 
# Explores PlantVillage image distribution across the 5 selected agricultural crops
# (Tomato, Potato, Corn, Grape, Apple), analyzes class balance, and evaluates
# data augmentation transformations.

# %%
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## 1. Target Crop Classes in PlantVillage

# %%
crops = ["Tomato", "Potato", "Corn", "Grape", "Apple"]
classes = [
    "Tomato___Early_Blight", "Tomato___Late_Blight", "Tomato___healthy",
    "Potato___Early_Blight", "Potato___Late_Blight", "Potato___healthy",
    "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___healthy",
    "Grape___Black_Rot", "Grape___Esca", "Grape___healthy",
    "Apple___Apple_Scab", "Apple___Black_Rot", "Apple___healthy"
]
print(f"Tracking {len(classes)} classes across {len(crops)} crops.")

# %% [markdown]
# ## 2. Simulated Class Balance Distribution

# %%
# Typical distribution in PlantVillage subset
counts = [1000, 1000, 1591, 1000, 1000, 152, 1192, 513, 1162, 1180, 1383, 423, 630, 621, 1645]

plt.figure(figsize=(10, 6))
plt.barh(classes, counts, color="seagreen")
plt.xlabel("Number of Labeled Images")
plt.title("PlantVillage Class Counts for Selected Agricultural Commodities")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
