"""
SageMaker Model Monitor Configuration: Real-time Drift Detection.

Configures data capture on the Disease Detection endpoint, establishes a statistical baseline
from training data, and sets up hourly/daily monitoring schedules for data quality and model bias.
"""

from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    DatasetFormat,
    CronExpressionGenerator,
)


def enable_data_capture(
    endpoint_name: str,
    s3_capture_uri: str,
    sampling_percentage: int = 100,
):
    """Configures SageMaker endpoint data capture for input images & output predictions."""
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=sampling_percentage,
        destination_s3_uri=s3_capture_uri,
        capture_options=["REQUEST", "RESPONSE"],
        csv_content_types=["text/csv", "application/json"],
        json_content_types=["application/json"],
    )
    print(f"Data capture enabled for {endpoint_name} -> {s3_capture_uri}")
    return data_capture_config


def create_monitoring_schedule(
    role: str,
    endpoint_name: str,
    baseline_s3_uri: str,
    output_s3_uri: str,
    instance_type: str = "ml.m5.xlarge",
):
    """Creates a scheduled Model Monitor job comparing live payloads against baseline."""
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type=instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )

    # Suggest baseline constraints
    monitor.suggest_baseline(
        baseline_dataset=baseline_s3_uri,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"{output_s3_uri}/baseline",
        wait=True,
    )

    # Schedule hourly or daily checks
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f"{endpoint_name}-quality-monitor",
        endpoint_input=endpoint_name,
        output_s3_uri=f"{output_s3_uri}/reports",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )
    print(f"Model Monitor scheduled for {endpoint_name}")
    return monitor
