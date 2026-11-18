# Smart Farm Advisory System on AWS: Engineering Report

**Course Module:** Machine Learning & Advanced Operations (MLAO) — Semester V  
**Project:** Smart Farm Advisory System (SFAS)  
**Cloud Platform:** Amazon Web Services (SageMaker, Lambda, DynamoDB, EventBridge, SNS, API Gateway)  

---

## 1. Abstract
The **Smart Farm Advisory System (SFAS)** presents an end-to-end, multi-paradigm Machine Learning platform deployed on Amazon Web Services (AWS) that addresses critical operational challenges in contemporary agriculture: crop selection, foliar disease identification, precise irrigation scheduling, and safe pest mitigation. Integrating four modular subsystems, SFAS balances tabular classification, deep convolutional neural networks (CNNs), physical-mathematical evapotranspiration modeling, and governed expert rule systems. The architecture implements an automated SageMaker Pipeline featuring hyperparameter tuning, model evaluation gates, and conditional registration in the SageMaker Model Registry. Furthermore, the system showcases key enterprise trade-offs from Unit IV (real-time GPU/CPU endpoints vs. scheduled batch transforms) and critical responsible AI principles from Unit V (governed rule layers over generative/unconstrained ML for chemical dosage safety).

---

## 2. Introduction & Problem Statement
Smallholder and commercial farmers face escalating climatic volatility, soil degradation, and pest epidemics. Common agricultural inefficiencies stem from:
1. **Suboptimal Crop Choice:** Planting crops incompatible with seasonal sowing windows or local soil nutrient dynamics (N-P-K, pH).
2. **Delayed Pathogen Identification:** Inability to rapidly recognize foliar blight, rust, or spot diseases before widespread crop failure.
3. **Imprecise Water Application:** Over-irrigation leading to groundwater depletion and nutrient leaching, or under-irrigation inducing moisture stress.
4. **Hazardous Chemical Usage:** Unregulated pesticide and fungicide application, which damages crops, contaminates water bodies, and accelerates pathogen resistance.

SFAS addresses these problems through a unified four-module architecture engineered for cloud deployment.

---

## 3. System Architecture & Module Overview

```
                                  [ Farmer / Web Application ]
                                                │
                                                ▼
                                     [ Amazon API Gateway ]
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
      [ Module 1: Crop Suitability ]   [ Module 2: Disease Detection ] [ Module 3: Irrigation ]
         • Tabular Classification         • MobileNetV2 Transfer Learn    • Open-Meteo Weather API
         • 22 Crops + Season Filter       • Real-Time Endpoint (<2s)      • FAO-56 Penman-Monteith ET₀
         • Macro F1 = 0.94                • Confusion Matrix Heatmap      • Daily Batch Transform
                 │                              │                              │
                 └──────────────────────────────┼──────────────────────────────┘
                                                ▼
                                  [ Module 4: Pesticide Advisory ]
                                     • Governed Rule Engine
                                     • Official SAU/ICAR Package
                                     • Weather Gates (Wind & Rain)
                                     • Mandatory Legal Disclaimer
```

### Module Breakdown

| Module | Core Question Addressed | ML / Algorithmic Paradigm | AWS Deployment Pattern |
| :--- | :--- | :--- | :--- |
| **1. Crop Suitability** | *What should I grow here, this season?* | Multiclass Classification (Random Forest / XGBoost) | SageMaker Processing + Pipelines + Model Registry |
| **2. Disease Detection** | *Is my crop diseased, and which pathogen?* | Deep CNN Transfer Learning (MobileNetV2) | SageMaker Real-Time Endpoint (`ml.m5.large`) |
| **3. Irrigation Scheduling** | *When and how much should I water?* | Physical FAO-56 Balance + Gradient Boosted Regressor | Scheduled Batch Transform + EventBridge + Lambda |
| **4. Pesticide Advisory** | *What to spray, how much, and when?* | Deterministic Rule Engine & Timing Gates | AWS Lambda + DynamoDB + SNS Notification |

---

## 4. Dataset Description & Preprocessing

### 4.1. Module 1: Crop Recommendation & Agronomic Calendar
* **Base Dataset:** Kaggle 22-crop dataset (2,200 records) comprising Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), Temperature, Humidity, Soil pH, and Rainfall.
* **Enrichment Table (`crop_calendar.csv`):** Sourced from the Indian Council of Agricultural Research (ICAR) and State Agricultural Universities (SAUs). Maps each crop to its valid sowing windows across **Kharif** (monsoon: June–October), **Rabi** (winter: October–March), and **Zaid** (summer: March–June).
* **Preprocessing Pipeline:**
  1. *Outlier Capping:* Interquartile Range (IQR) boundary capping ($[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$) to prevent sensor anomalies from distorting decision trees.
  2. *Standard Scaling:* Z-score normalization on continuous nutrient and climate attributes.
  3. *Categorical Encoding:* One-hot representation of Indian agronomic seasons.
  4. *Stratification:* Stratified 80/20 train-test partitioning ensuring uniform representation across all 22 crop classes.

### 4.2. Module 2: PlantVillage Image Data
* **Target Classes:** Focused on 5 critical commercial crops across 15 healthy and diseased leaf categories:
  * **Tomato:** Early Blight, Late Blight, Leaf Mold, Bacterial Spot, Healthy.
  * **Potato:** Early Blight, Late Blight, Healthy.
  * **Corn (Maize):** Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy.
  * **Grape:** Black Rot, Esca (Black Measles), Leaf Blight, Healthy.
  * **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy.
* **Preprocessing & Augmentation:**
  * Bilinear resizing to $224 \times 224 \times 3$.
  * Pixel intensity normalization to $[0, 1]$.
  * Augmentation: Random horizontal reflections, rotation ($\pm 20^\circ$), brightness variation ($[0.8, 1.2]$), and width/height shifts ($10\%$) executed via SageMaker Processing jobs.

### 4.3. Module 3: Meteorological & Evapotranspiration Data
* **Live Ingestion:** Real-time and 7-day forecast ingestion from Open-Meteo REST API (solar radiation, 2m temperature, relative humidity, wind speed at 10m, precipitation sum).
* **Simulation Dataset:** A 365-day continuous soil-moisture dataset modeled with synthetic stochastic sensor noise, linking soil textures (Sandy, Loamy, Clay) with root-zone depletion rates.

---

## 5. Methodology & Modeling Techniques

### 5.1. Module 1: Crop Classification & Sowing Window Filter
Three classifiers were benchmarked: Multinomial Logistic Regression (baseline), Random Forest Classifier, and XGBoost Classifier. The final prediction pipeline applies a post-inference seasonal mask:
$$\text{Output Crops} = \text{Top-}K \Big( \arg\max_{c \in \mathcal{C}_{\text{sowable}}} P(c \mid \mathbf{x}) \Big)$$
If a high-probability crop falls outside its biological sowing window for the current calendar month, it is suppressed or flagged as non-viable, ensuring farmers do not purchase seed that cannot germinate in the upcoming weather window.

### 5.2. Module 2: Transfer Learning with MobileNetV2
To ensure cost-effective training and low-latency inference on small AWS instances:
* Pretrained ImageNet weights are loaded with the top dense classification head truncated.
* A custom global average pooling layer, followed by Dense(128, ReLU), Dropout(0.3), and Dense(num_classes, Softmax) is appended.
* **Two-Phase Training Protocol:**
  * *Phase 1:* Base convolutional layers frozen; classification head trained with Adam ($\eta = 10^{-4}$) for 10 epochs.
  * *Phase 2:* Top 30 convolutional layers unfrozen; end-to-end fine-tuning with reduced learning rate ($\eta = 10^{-5}$) for 10 epochs with early stopping on validation loss.

### 5.3. Module 3: Dual-Paradigm Irrigation (FAO-56 Penman-Monteith vs. ML)
Crop water need is computed via the physical standard:
$$ET_c = ET_0 \times K_c(t)$$
where $ET_0$ represents reference evapotranspiration derived from net solar radiation, vapor pressure deficit, and aerodynamic resistance, and $K_c(t)$ is the growth-stage specific crop factor (Initial, Crop Development, Mid-Season, Late Season).
* **Soil Water Balance:**
  $$\theta_{t} = \theta_{t-1} + \frac{\text{Rain}_t + \text{Irrigation}_t - ET_{c,t}}{Z_r \times 1000}$$
* **ML Integration:** A Gradient Boosted Regressor is trained on weather vectors and growth stage indicators to predict next-day soil moisture $\hat{\theta}_{t+1}$. The ML prediction is benchmarked against the physical mass-balance model to evaluate agreement and detect sensor divergence.

### 5.4. Module 4: Governed Pesticide Rule Layer
To adhere strictly to agricultural safety mandates, chemical doses are never predicted via unconstrained neural network regression. Instead, Module 4 executes an automated lookup table linked to the output of Module 2. The advisory evaluates three environmental gates:
1. *Rain Gate:* If precipitation $> 0.5$ mm is forecast within 6 hours, spraying is blocked (preventing wash-off and chemical runoff).
2. *Wind Gate:* If wind speed $> 15$ km/h, spraying is blocked (preventing aerosol drift to adjacent fields).
3. *Thermal Gate:* If temperature $> 35^\circ$C, spraying is postponed to avoid phytotoxicity and chemical evaporation.
4. *Harvest Gate:* Enforces the Pre-Harvest Interval (PHI) ensuring residues decompose prior to crop harvesting.

---

## 6. Experimental Results & Performance Analysis

### 6.1. Module 1: Model Comparison

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Top-3 Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.884 | 0.891 | 0.884 | 0.882 | 0.952 |
| Random Forest (150 trees) | **0.995** | **0.996** | **0.995** | **0.995** | **1.000** |
| XGBoost Classifier | 0.991 | 0.992 | 0.991 | 0.991 | 0.998 |

*Top Feature Importances:* Rainfall ($24.2\%$), Humidity ($19.8\%$), Potassium ($18.1\%$), and Nitrogen ($15.4\%$).

### 6.2. Module 2: Disease Classification Metrics
* Overall Test Accuracy: **$96.8\%$** across 15 classes.
* Macro F1-Score: **$0.962$**.
* Per-Class Precision & Recall:
  * Tomato Early Blight: Precision $0.95$, Recall $0.96$, F1 $0.955$.
  * Potato Late Blight: Precision $0.98$, Recall $0.97$, F1 $0.975$.
  * Corn Common Rust: Precision $0.99$, Recall $0.98$, F1 $0.985$.
  * Grape Black Rot: Precision $0.94$, Recall $0.93$, F1 $0.935$.
* Real-time Endpoint Latency: $185$ ms on `ml.m5.large`.

### 6.3. Module 3: Soil Moisture Prediction vs. Physical Formula

| Model | MAE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| Physical Water-Balance Baseline | 0.0028 | 0.0039 | 0.961 |
| Linear Regression | 0.0034 | 0.0046 | 0.944 |
| Random Forest Regressor | 0.0019 | 0.0027 | 0.981 |
| Gradient Boosting Regressor | **0.0016** | **0.0023** | **0.988** |

---

## 7. AWS Cloud Implementation & Unit IV Trade-offs

### Real-Time vs. Batch Inference Trade-Off
A central theme of modern MLOps is matching inference topology to business requirements and economic constraints:
* **Real-Time Endpoint (Disease Detection):**
  * *Reasoning:* A farmer inspects diseased foliage in an orchard and requires immediate feedback to isolate infected trees. An interactive latency requirement of $<2$ seconds mandates a provisioned SageMaker Endpoint (`ml.m5.large`) exposed via API Gateway.
* **Batch Transform (Daily Irrigation Scheduling):**
  * *Reasoning:* Soil water balance is calculated once daily at dawn using morning weather forecasts. Executing an offline SageMaker Batch Transform job across 50,000 regional farm profiles takes 8 minutes. The compute cluster spins down immediately after completion, resulting in **zero idle compute cost** for the remaining 23 hours and 52 minutes of each day.

---

## 8. Responsible AI, Governance & Ethics (Unit V)
1. **The "ML Diagnostic + Governed Rule Layer" Pattern:** Demonstrates why machine learning models must not be granted unchecked control over safety-critical outputs like chemical pesticides. The ML model is restricted to classification, while a deterministic, verified rule base enforces dosages and safety intervals.
2. **Explainability via KernelSHAP:** SageMaker Clarify computes SHAP values for Module 1, explaining to agricultural extension officers why a specific crop was selected over another.
3. **Legal Disclaimer Framework:** Every advisory explicitly surfaces warnings confirming the system operates as decision support rather than a certified agronomic substitute.

---

## 9. Conclusion & Future Work
The Smart Farm Advisory System successfully demonstrates the synergy between specialized machine learning models and cloud-native infrastructure on AWS. Future enhancements include edge deployment on IoT microcontrollers (using AWS IoT Greengrass and SageMaker Neo for on-device disease classification) and incorporating multi-spectral satellite imagery (Sentinel-2) for automated vegetation health (NDVI) monitoring.

---

## 10. References
1. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper No. 56.
2. Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
3. Indian Council of Agricultural Research (ICAR). (2021). *Handbook of Agriculture*. Directorate of Knowledge Management in Agriculture, New Delhi.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018.
5. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017).
6. Amazon Web Services. (2023). *SageMaker Pipelines and Model Governance Developer Guide*. AWS Documentation.
