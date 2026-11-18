# ☁️ AWS Cloud Architecture: Smart Farm Advisory System

## 1. System Overview

The **Smart Farm Advisory System (SFAS)** leverages an enterprise-grade, serverless, and cloud-native architecture on Amazon Web Services (AWS) to deliver intelligent agricultural decision support.

```
       [ Farmer / Web UI ]
               │
               ▼
     [ Amazon API Gateway ]
               │
        ┌──────┴─────────────────────────┐
        ▼                                ▼
[ Lambda: API Handler ]        [ Real-Time Endpoint ]
        │                      (MobileNetV2 Disease)
        ▼                                ▲
[ DynamoDB Tables ]                      │
  • FarmProfiles                         │
  • IrrigationAdvisories                 │
  • PesticideAdvisories                  │
        ▲                                │
        │ Daily                          │
[ Lambda: Irrigation ]                   │
        ▲                                │
        │ 06:00 AM IST                   │
[ Amazon EventBridge ]                   │
        │                                │
        ▼                                │
[ Amazon SNS ] (SMS/Email)               │
                                         │
 ══════════════ MLOps Layer ═════════════╪══════════════════════════════
                                         │
 [ Amazon S3: Data Lake ]                │
   ├── raw/                              │
   ├── processed/                        │
   └── models/                           │
        │                                │
        ▼                                │
 [ SageMaker Processing ]                │
   ├── Tabular Preprocessing             │
   └── Image Preprocessing               │
        │                                │
        ▼                                │
 [ SageMaker Pipelines (DAG) ]           │
   ├── HPO / Training (Crop & Disease)   │
   ├── Evaluation (F1 >= 0.85)           │
   └── Model Registry ───────────────────┘
        │
        ▼
 [ SageMaker Model Monitor & Clarify ]
   ├── Data & Prediction Drift (CloudWatch)
   └── SHAP Feature Explainability
```

---

## 2. Component Mapping & Justifications

| AWS Service | Role in SFAS | Architectural Justification |
| :--- | :--- | :--- |
| **Amazon S3** | Data Lake & Artifact Storage | Versioned, durable object storage separating raw input datasets, processed features, trained model weights, and inference logs. |
| **SageMaker Processing** | Feature Engineering | Managed containers for CPU/GPU-intensive tasks (image resizing/augmentation, tabular IQR capping, scaling) isolated from model training. |
| **SageMaker Training & HPO** | Model Training & Tuning | Distributed training with automated Hyperparameter Optimization (Bayesian search) over tree depth, learning rate, and estimators. |
| **SageMaker Pipelines** | CI/CD MLOps Automation | Directed Acyclic Graph (DAG) coordinating steps from preprocessing to evaluation and conditional registration into Model Registry. |
| **SageMaker Model Registry** | Model Governance | Central repository managing model versions, lineage tracking, and manual approval gates before deployment. |
| **SageMaker Real-Time Endpoint** | Disease Detection Serving | Low-latency (<2s) on-demand GPU/CPU endpoint for instant field diagnosis when farmers upload diseased leaf photos. |
| **SageMaker Batch Transform** | Daily Irrigation Prediction | High-throughput offline batch processing over thousands of registered farms; zero idle cost as cluster terminates after run. |
| **Amazon EventBridge** | Workflow Scheduling | Cron scheduler triggering the daily 06:00 AM IST irrigation calculation run. |
| **AWS Lambda** | Serverless Execution | Event-driven microservices: API routing, daily advisory dispatch, pesticide rule enforcement. |
| **Amazon DynamoDB** | Operational Data Store | Low-latency NoSQL database storing farm profiles, weather history, and generated advisories with pay-per-request billing. |
| **Amazon SNS** | Notification Service | Dispatches immediate SMS alerts and push notifications for urgent pest infestations or irrigation requirements. |
| **Amazon API Gateway** | API Exposure | Secure, managed HTTPS entry point with CORS support, throttling, and authentication for frontend integration. |
| **SageMaker Clarify & Monitor** | Explainability & Drift | Calculates SHAP attribution values for crop suitability and tracks feature/concept drift over time. |

---

## 3. Core Syllabus Architectural Concepts

### A. Real-Time vs. Batch Inference Trade-Off (Unit IV)
* **Module 2 (Disease Detection) → Real-Time Endpoint**:
  * *Context*: A farmer standing in an orchard spots yellowing or necrotic spots on a leaf.
  * *Requirement*: Sub-second or immediate turnaround (<2s) to identify pathogens before moving to other plants.
  * *Trade-off*: Requires a constantly warm, provisioned endpoint (`ml.m5.large`). Higher idle cost is justified by interactive user necessity.
* **Module 3 (Irrigation Scheduling) → Batch Transform**:
  * *Context*: Farm watering schedules are determined based on daily cumulative evapotranspiration ($ET_0$) and morning weather forecasts.
  * *Requirement*: High throughput across 10,000+ regional farms. Farm moisture does not change rapidly within minutes.
  * *Trade-off*: EventBridge triggers a scheduled batch job at 06:00 AM IST. EC2 instances spin up, process the entire farm catalog in parallel, output predictions to S3 and DynamoDB, and immediately shut down. Zero idle infrastructure cost during the other 23.5 hours of the day.

### B. Governed Rule Layer Over ML Prediction (Unit V)
* **Module 4 (Pesticide Advisory)**:
  * Pure ML models hallucinate or generate unsafe non-linear extrapolations when predicting chemical dosages. An over-dosage destroys crops and risks human poisoning; under-dosage breeds chemical resistance.
  * **Solution**: The ML model predicts *only* the disease classification ($P(disease|image)$). A strictly verified, deterministic **Rule Engine** (built from official ICAR and State Agricultural University Package of Practices) governs the dosage, safe weather windows (skipping spray if wind > 15 km/h or rain is imminent), and enforces Pre-Harvest Intervals (PHI).

---

## 4. Security, Resilience & Cost Optimization

1. **IAM Principle of Least Privilege**: Lambda functions have scoped permissions (`dynamodb:PutItem`, `sns:Publish`, `sagemaker:InvokeEndpoint`) restricted to resource ARNs.
2. **KMS Encryption**: All S3 buckets and DynamoDB tables use customer-managed AWS KMS keys for encryption at rest.
3. **Cost Optimization**:
   - Spot instances configured for SageMaker Training and Processing jobs (saving up to 70%).
   - Auto-scaling configured on SageMaker Real-time Endpoints (scaling to zero or minimum instances during night hours).
   - DynamoDB On-Demand capacity avoids paying for unused read/write units.
