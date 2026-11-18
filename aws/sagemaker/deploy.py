"""
AWS SageMaker Model Deployment: Real-time Endpoint vs Batch Transform.

Unit IV Concept Demonstration:
1. Real-Time Endpoint (Module 2: Disease Detection):
   - Justification: Interactive, on-demand query. A farmer in the field notices
     a diseased leaf, takes a photograph, and expects an immediate (<2 second) diagnosis
     to make quick containment decisions.
   - Mechanism: SageMaker Real-Time Endpoint with AutoScaling on ml.t2.medium / ml.m5.large.

2. Batch Transform (Module 3: Daily Irrigation Advisory):
   - Justification: Scheduled, non-interactive high-throughput workload. Farms do not need
     irrigation recalculated second-by-second. Daily weather forecast data arrives once or twice
     a day (early morning). Running a batch transform job over all registered farms in a region
     is significantly cheaper (zero idle endpoint cost, spins down when complete).
   - Mechanism: SageMaker Batch Transform Job triggered daily by EventBridge + Lambda.
"""

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.sklearn import SKLearnModel


def deploy_disease_realtime_endpoint(
    model_s3_uri: str,
    role: str,
    endpoint_name: str = "smartfarm-disease-detector-endpoint",
    instance_type: str = "ml.m5.large",
):
    """
    Deploys MobileNetV2 disease detection model as a persistent, low-latency endpoint.
    """
    print(f"Deploying Real-Time Endpoint for Disease Detection: {endpoint_name}...")
    tf_model = TensorFlowModel(
        model_data=model_s3_uri,
        role=role,
        framework_version="2.11",
    )

    predictor = tf_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    print(f"Real-Time Endpoint online: {endpoint_name}")
    return predictor


def run_irrigation_batch_transform(
    model_s3_uri: str,
    input_s3_data: str,
    output_s3_data: str,
    role: str,
    instance_type: str = "ml.m5.large",
    instance_count: int = 1,
):
    """
    Executes an on-demand/scheduled Batch Transform job for daily irrigation predictions.
    Computes recommendations for all farms, uploads output CSV/JSON to S3, and terminates.
    """
    print(f"Starting Batch Transform Job for Daily Irrigation...")
    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=role,
        framework_version="1.2-1",
        entry_point="aws/sagemaker/training/irrigation_training.py",
    )

    transformer = sklearn_model.transformer(
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_s3_data,
        assemble_with="Line",
        accept="text/csv",
    )

    transformer.transform(
        data=input_data_s3,
        content_type="text/csv",
        split_type="Line",
        wait=False,
    )
    print(f"Batch Transform Job submitted: {transformer.latest_transform_job.name}")
    return transformer


if __name__ == "__main__":
    print("Deployment utilities loaded. See functions for real-time vs batch deployment.")
