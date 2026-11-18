# 🌾 Smart Farm Advisory System

> An end-to-end machine-learning platform that delivers **crop recommendation, plant disease detection, irrigation scheduling,** and **pesticide advisory** — from raw data to cloud-deployed inference.

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [Repository Structure](#repository-structure)
3. [Setup & Installation](#setup--installation)
4. [Usage Guide](#usage-guide)
5. [AWS Deployment](#aws-deployment)
6. [Architecture](#architecture)
7. [Disclaimer](#disclaimer)
8. [License](#license)

---

## Module Overview

| # | Module | Description | Key Algorithm |
|---|--------|-------------|---------------|
| 1 | **Crop Recommendation** | Suggests the optimal crop based on soil nutrients (N, P, K), pH, rainfall, temperature, and humidity. | Random Forest / XGBoost |
| 2 | **Disease Detection** | Classifies plant leaf images into healthy or one of 37 disease categories using transfer learning. | ResNet-50 CNN (TensorFlow/Keras) |
| 3 | **Irrigation Scheduling** | Predicts when and how much to irrigate by combining weather forecasts with soil & crop parameters. | XGBoost Regressor |
| 4 | **Pesticide Advisory** | Recommends suitable pesticides and application rates given the detected disease & environmental context. | Gradient Boosting Classifier |

---

## Repository Structure

```
Project/
├── data/
│   ├── raw/                 # Original datasets & crop calendar
│   ├── processed/           # Cleaned, feature-engineered data
│   └── interim/             # Intermediate artefacts
├── models/                  # Serialised model files (.joblib, .h5)
├── reports/
│   └── figures/             # Saved plots & confusion matrices
├── logs/                    # Runtime log files
├── src/
│   ├── utils/
│   │   ├── config.py        # Central configuration & constants
│   │   └── common.py        # Shared helpers (I/O, plotting, logging)
│   ├── module1_crop/        # Crop recommendation pipeline
│   ├── module2_disease/     # Disease detection pipeline
│   ├── module3_irrigation/  # Irrigation scheduling pipeline
│   └── module4_pesticide/   # Pesticide advisory pipeline
├── notebooks/               # Exploratory & demo notebooks
├── requirements.txt         # Python dependencies
└── README.md                # ← You are here
```

---

## Setup & Installation

### Prerequisites

- **Python** ≥ 3.10
- **pip** (or `conda`)
- *Optional*: an AWS account with SageMaker access for cloud deployment

### 1. Clone the repository

```bash
git clone <repository-url>
cd Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download data

| Dataset | Source | Destination |
|---------|--------|-------------|
| Crop Recommendation | [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | `data/raw/` |
| PlantVillage Leaf Images | [Kaggle — PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) | `data/raw/plant_village/` |
| Crop Calendar | Provided / custom | `data/raw/crop_calendar.csv` |
| Weather Data | Auto-fetched via Open-Meteo API | `data/raw/weather/` |

---

## Usage Guide

### Module 1 — Crop Recommendation

```bash
# Data pre-processing
python -m src.module1_crop.preprocess

# Train model
python -m src.module1_crop.train

# Evaluate
python -m src.module1_crop.evaluate

# Predict for new input
python -m src.module1_crop.predict --N 90 --P 42 --K 43 --temperature 20.8 --humidity 82 --ph 6.5 --rainfall 202.9
```

### Module 2 — Disease Detection

```bash
# Prepare image dataset (resize, split, augment)
python -m src.module2_disease.preprocess

# Train CNN
python -m src.module2_disease.train

# Evaluate & generate confusion matrix
python -m src.module2_disease.evaluate

# Predict from a single leaf image
python -m src.module2_disease.predict --image path/to/leaf.jpg
```

### Module 3 — Irrigation Scheduling

```bash
# Fetch weather data & merge with soil/crop features
python -m src.module3_irrigation.preprocess

# Train irrigation model
python -m src.module3_irrigation.train

# Generate weekly irrigation schedule
python -m src.module3_irrigation.schedule --crop rice --region "Maharashtra"
```

### Module 4 — Pesticide Advisory

```bash
# Prepare advisory dataset
python -m src.module4_pesticide.preprocess

# Train advisory model
python -m src.module4_pesticide.train

# Get recommendation for a detected disease
python -m src.module4_pesticide.recommend --disease "Tomato_Late_blight" --severity High
```

---

## AWS Deployment

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SFAS_AWS_REGION` | AWS region | `ap-south-1` |
| `SFAS_S3_BUCKET_DATA` | S3 bucket for datasets | `smart-farm-data` |
| `SFAS_S3_BUCKET_MODELS` | S3 bucket for trained models | `smart-farm-models` |
| `SFAS_S3_BUCKET_ARTIFACTS` | S3 bucket for pipeline artefacts | `smart-farm-artifacts` |
| `SFAS_SAGEMAKER_ROLE` | SageMaker execution IAM role ARN | — |
| `SFAS_SAGEMAKER_INSTANCE` | SageMaker training instance type | `ml.m5.xlarge` |

### Deployment Steps

1. **Upload data to S3**
   ```bash
   aws s3 sync data/raw/ s3://$SFAS_S3_BUCKET_DATA/raw/
   ```

2. **Train on SageMaker**
   ```bash
   python -m src.deploy.sagemaker_train --module crop
   ```

3. **Deploy endpoint**
   ```bash
   python -m src.deploy.sagemaker_deploy --module crop --instance ml.m5.large
   ```

4. **Invoke endpoint**
   ```python
   import boto3, json
   client = boto3.client("sagemaker-runtime")
   response = client.invoke_endpoint(
       EndpointName="smart-farm-crop-endpoint",
       ContentType="application/json",
       Body=json.dumps({"N": 90, "P": 42, "K": 43, "temperature": 20.8,
                        "humidity": 82, "ph": 6.5, "rainfall": 202.9}),
   )
   print(json.loads(response["Body"].read()))
   ```

---

## Architecture

> See [`reports/figures/architecture_diagram.png`](reports/figures/architecture_diagram.png) for the full system architecture.

```
┌───────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
│  Raw Data │──▸│ Preprocess │──▸│  Train Model │──▸│   Evaluate   │
└───────────┘   └────────────┘   └──────┬───────┘   └──────┬───────┘
                                        │                  │
                                        ▼                  ▼
                                 ┌──────────────┐   ┌────────────┐
                                 │  Save Model  │   │  Reports   │
                                 └──────┬───────┘   └────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                    ┌──────────┐ ┌───────────┐ ┌───────────┐
                    │  Local   │ │ SageMaker │ │ REST API  │
                    │ Predict  │ │ Endpoint  │ │  Gateway  │
                    └──────────┘ └───────────┘ └───────────┘
```

---

## Disclaimer

> [!CAUTION]
> **Pesticide Advisory — Important Notice**
>
> The pesticide recommendations generated by Module 4 are **machine-learning-based suggestions only** and must **not** be treated as professional agronomic or regulatory advice. Always:
>
> - Consult a certified agronomist or local agricultural extension officer before applying any pesticide.
> - Follow the manufacturer's label instructions regarding dosage, dilution, protective equipment, and pre-harvest intervals.
> - Comply with regional pesticide regulations and approved product lists.
>
> The authors assume **no liability** for crop damage, health risks, or environmental impact arising from the use of these recommendations.

---

## 🌐 Web Interface & Vercel Deployment

The frontend dashboard can be launched locally or deployed to **Vercel** with zero configuration:

### Option A: Deploy with Vercel CLI
```bash
npm i -g vercel
vercel
```

### Option B: Deploy via GitHub / Vercel Dashboard
1. Push this repository to GitHub.
2. In the [Vercel Dashboard](https://vercel.com/new), select **Import Git Repository**.
3. The included [vercel.json](file:///c:/Users/Aditi/OneDrive/Desktop/CLG/SEM-5/MLAO/Project/vercel.json) will automatically configure clean URLs, security headers, and route traffic to the interactive agricultural dashboard.

### Option C: Local Preview
Simply open `index.html` (or `frontend/index.html`) directly in any web browser.

---

## License

This project is developed for academic purposes as part of the **SEM-5 MLAO** course.

---

*Built with ❤️ for smarter, data-driven agriculture.*
