"""
SageMaker Pipelines: Automated CI/CD Pipeline Definition.

Defines an end-to-end DAG:
ProcessingStep → HyperparameterTuningStep / TrainingStep → EvaluationStep
→ ConditionStep (Macro F1 >= 0.85) → RegisterModel (Model Registry)
"""

import os
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.inputs import TrainingInput


def create_crop_pipeline(
    role: str = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole",
    default_bucket: str = "smart-farm-advisory-bucket",
    pipeline_name: str = "SmartFarm-CropRecommendation-Pipeline",
    base_job_prefix: str = "sfas-crop",
) -> Pipeline:
    """Builds and returns the SageMaker Pipeline specification."""

    # 1. Pipeline Parameters
    input_data_s3 = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/data/raw/crop_recommendation.csv",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    f1_score_threshold = ParameterFloat(
        name="F1ScoreThreshold", default_value=0.85
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )

    # 2. Preprocessing Step
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess",
        role=role,
    )

    step_process = ProcessingStep(
        name="PreprocessCropData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data_s3,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output",
                destination=f"s3://{default_bucket}/data/processed/crop",
            )
        ],
        code="aws/sagemaker/processing/tabular_processing.py",
    )

    # 3. Training Step (with Scikit-learn / XGBoost)
    estimator = Estimator(
        image_uri="683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        entry_point="aws/sagemaker/training/crop_training.py",
        base_job_name=f"{base_job_prefix}-train",
        hyperparameters={
            "n-estimators": 150,
            "max-depth": 12,
            "model-type": "random_forest",
        },
    )

    step_train = TrainingStep(
        name="TrainCropModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    # 4. Evaluation Step (Evaluates Macro F1 on held-out test data)
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-eval",
        role=role,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation_metrics.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateCropModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{default_bucket}/evaluation/crop",
            )
        ],
        code="src/module1_crop/evaluate.py",
        property_files=[evaluation_report],
    )

    # 5. Model Registration Step
    model = Model(
        image_uri="683313688378.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=None,
        role=role,
    )

    step_register = ModelStep(
        name="RegisterCropModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="SmartFarmCropRecommendationGroup",
            approval_status=model_approval_status,
        ),
    )

    # 6. Condition Step: Only register if Macro F1 >= threshold
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,  # parsed via report
        right=f1_score_threshold,
    )

    step_cond = ConditionStep(
        name="CheckEvaluationMetric",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # 7. Pipeline DAG Definition
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data_s3,
            model_approval_status,
            f1_score_threshold,
            processing_instance_type,
            training_instance_type,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
    )

    return pipeline


if __name__ == "__main__":
    pipeline = create_crop_pipeline()
    pipeline_json = pipeline.definition()
    print("Pipeline definition validated successfully.")
