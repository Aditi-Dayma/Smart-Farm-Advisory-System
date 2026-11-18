"""
SageMaker Clarify: Model Explainability (SHAP) & Fairness Configuration.

Computes KernelSHAP feature importance values for Module 1 (Crop Suitability)
to explain why a specific crop was suggested based on Nitrogen, Phosphorus, Potassium,
soil pH, and precipitation, and detects bias across climate/soil profiles.
"""

from sagemaker import clarify


def configure_clarify_processor(
    role: str,
    instance_type: str = "ml.m5.xlarge",
    instance_count: int = 1,
):
    """Initializes SageMaker Clarify Processor."""
    clarify_processor = clarify.SageMakerClarifyProcessor(
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=None,
    )
    return clarify_processor


def run_explainability_job(
    clarify_processor,
    data_s3_uri: str,
    model_name: str,
    output_s3_uri: str,
    target_column: str = "label",
):
    """Executes SHAP feature attribution analysis."""
    data_config = clarify.DataConfig(
        s3_data_input_path=data_s3_uri,
        s3_output_path=output_s3_uri,
        label=target_column,
        dataset_type="text/csv",
    )

    model_config = clarify.ModelConfig(
        model_name=model_name,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        content_type="text/csv",
        accept_type="text/csv",
    )

    # SHAP configuration
    shap_config = clarify.SHAPConfig(
        baseline=[[50.0, 50.0, 50.0, 25.0, 70.0, 6.5, 100.0, 1.0, 0.0, 0.0]],
        num_samples=100,
        agg_strategy="mean_abs",
    )

    print("Submitting SageMaker Clarify Explainability Job...")
    clarify_processor.run_explainability(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    print(f"SHAP explanation report generated at: {output_s3_uri}")
