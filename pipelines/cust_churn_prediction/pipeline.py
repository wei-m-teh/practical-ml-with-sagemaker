"""Example workflow pipeline script for customer churn prediction pipeline.
                                                                                 . -ModelStep
                                                                                .
    Process-> DataQualityCheck/DataBiasCheck -> Train -> Evaluate -> Condition .
                                                  |                              .
                                                  |                                . -(stop)
                                                  |
                                                   -> CreateModel-> ModelBiasCheck/ModelExplainabilityCheck
                                                           |
                                                           |
                                                            -> BatchTransform -> ModelQualityCheck

Implements a get_pipeline(**kwargs) method.
"""
import boto3
import pandas as pd
import sagemaker
import os
import json
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig,
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
    ParameterBoolean
)
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.inputs import CreateModelInput, TransformInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.transformer import Transformer
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ScriptProcessor
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.drift_check_baselines import DriftCheckBaselines

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    s3_client = boto3.resource('s3')
    pipeline_name = f"StreamingServiceChurnModelPipeline"
    sagemaker_session = sagemaker.session.Session()
    region = sagemaker_session.boto_region_name
    pipeline_session = PipelineSession()
    default_bucket = sagemaker_session.default_bucket()

    model_package_group_name = f"StreamingServiceModelPackageGroup"
    s3_processing_input_prefix = "data/kkbox-customer-churn-model/raw"
    s3_code_prefix = "data/kkbox-customer-churn-model/code"
    s3_preprocessing_output_prefix = "data/kkbox-customer-churn-model/processed"
    s3_batch_transform_output_prefix = "data/kkbox-customer-churn-model/batch-transformed"
    s3_logs_prefix = "data/kkbox-customer-churn-model/logs"
    s3_model_prefix = "data/kkbox-customer-churn-model/output"
    s3_quality_prefx = "data/kkbox-customer-churn-model/quality"
    s3_bias_prefx = "data/kkbox-customer-churn-model/bias"
    s3_explainability_prefix = "data/kkbox-customer-churn-model/explainability"
    s3_model_evaluation_prefix = "data/kkbox-customer-churn-model/evaluation"
    auc_score_threshold = 0.75
    base_job_prefix = "kkbox-customer-churn-model"
    model_package_group_name = "kkbox-customer-churn-model-packages"
    pyspark_cluster_instance_type = "ml.m5.4xlarge"
    pyspark_cluster_instance_count = 4
    cache_config = CacheConfig(enable_caching=True, expire_after="P1M")
    
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount",default_value=4)
    processing_instance_type = ParameterString(name="ProcessingInstanceType",default_value="ml.m5.4xlarge")
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=True)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=True)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value="")
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value="")
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value="")
    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value=True)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=True)
    training_instance_type = ParameterString(name="TrainingInstanceType",default_value="ml.m5.2xlarge")
    inference_instance_type = ParameterString(name="InferenceInstanceType",default_value="ml.m5.2xlarge")
    input_data_prefix = ParameterString(name="InputDataPrefix",default_value=s3_processing_input_prefix,)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value=True)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=True)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value="")
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value="")
    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=True)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=True)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedStatistics", default_value="")
    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=True)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=True)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedStatistics", default_value="")

    # Upload processing script to S3
    s3_client.Bucket(default_bucket).upload_file(f"{BASE_DIR}/preprocess.py", f"{s3_code_prefix}/preprocess.py")
    
    # PreProcessing Step
    # In this step, we'll use a Pyspark SageMaker processing job to perform the feature engineering functionality. 
    # SageMaker Processing job will use an ephimeral cluster to run the given pyspark script and automatically 
    # shutsdown when the script complete.
    
    spark_processor = PySparkProcessor(
        base_job_name="sm-spark",
        framework_version="3.1",
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        max_runtime_in_seconds=12000,
        sagemaker_session = pipeline_session
    )

    processor_args = spark_processor.run(
        submit_app=f"s3://{default_bucket}/{s3_code_prefix}/preprocess.py",
        arguments=[
            "--s3_input_bucket",
            default_bucket,
            "--s3_input_key_prefix",
            input_data_prefix,
            "--s3_output_bucket",
            default_bucket,
            "--s3_output_key_prefix",
            s3_preprocessing_output_prefix,
        ],
        spark_event_logs_s3_uri=f"s3://{default_bucket}/{s3_logs_prefix}/spark_event_logs",
        logs=False,
    )

    step_process = ProcessingStep(name="PreProcess", step_args=processor_args, cache_config=cache_config)

    ### Calculating the Data Quality

    # `CheckJobConfig` is a helper function that's used to define the job configurations used by the `QualityCheckStep`.
    # By separating the job configuration from the step parameters, the same `CheckJobConfig` can be used across multiple
    # steps for quality checks.
    # The `DataQualityCheckConfig` is used to define the Quality Check job by specifying the dataset used to calculate
    # the baseline, in this case, the training dataset from the data processing step, the dataset format, in this case,
    # a csv file with no headers, and the output path for the results of the data quality check.
    
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=default_bucket, Prefix=s3_preprocessing_output_prefix)
    train_data_s3_uri_prefix = [ x['Key'] for x in response['Contents'] if f"{s3_preprocessing_output_prefix}/train" in x['Key']][0]
    train_data_s3_uri = os.path.join(f"s3://{default_bucket}", train_data_s3_uri_prefix)    

    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        sagemaker_session=sagemaker_session,
    )

    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=train_data_s3_uri,
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                s3_quality_prefx,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "dataqualitycheckstep",
            ],
        ),
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name,
        cache_config=cache_config,
        depends_on=[step_process]
    )

    #### Calculating the Data Bias

    # The job configuration from the previous step is used here and the `DataConfig` class is used to define how
    # the `ClarifyCheckStep` should compute the data bias. The training dataset is used again for the bias evaluation,
    # the column representing the label is specified through the `label` parameter, and a `BiasConfig` is provided.

    # In the `BiasConfig`, we specify a facet name (the column that is the focal point of the bias calculation),
    # the value of the facet that determines the range of values it can hold, and the threshold value for the label.
    # More details on `BiasConfig` can be found at
    # https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.clarify.BiasConfig

    headers = ['msno', 'is_churn', 'regist_trans', 'mst_frq_plan_days', \
           'revenue', 'regist_cancels', 'bd', 'tenure', 'num_25', \
           'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', \
           'total_secs', 'city', 'gender', 'registered_via', \
           'qtr_trans', 'mst_frq_pay_met', 'is_auto_renew']

    data_bias_analysis_cfg_output_path = (
        f"s3://{default_bucket}/{s3_quality_prefx}/databiascheckstep/analysis_cfg"
    )

    data_bias_data_config = DataConfig(
        s3_data_input_path=train_data_s3_uri,
        s3_output_path=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                s3_quality_prefx,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "databiascheckstep",
            ],
        ),
        label=1,
        excluded_columns=[0],
        dataset_type="text/csv",
        headers=headers,
        s3_analysis_config_output_path=data_bias_analysis_cfg_output_path,
    )


    data_bias_config = BiasConfig(
        label_values_or_threshold=[1], facet_name="gender", facet_values_or_threshold=[0], group_name="bd")

    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=data_bias_config,
    )

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_data_bias,
        register_new_baseline=register_new_baseline_data_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_data_bias,
        model_package_group_name=model_package_group_name,
        cache_config=cache_config,
        depends_on=[data_quality_check_step]
    )
    model_path = f"s3://{default_bucket}/{s3_model_prefix}"

    model_path = f"s3://{default_bucket}/{s3_model_prefix}"

    hyperparameters = {
        "max_depth":5,
        "eta":0.2,
        "gamma":4,
        "min_child_weight":6,
        "subsample":0.7,
        "n_estimators":50,
        "region" : region,
        "sm_experiment" : ExecutionVariables.PIPELINE_NAME,
        "sm_run" : ExecutionVariables.PIPELINE_EXECUTION_ID}

    xgb_train = XGBoost(entry_point = "pipelines/cust_churn_prediction/train.py", 
                        framework_version='1.5-1',
                        hyperparameters=hyperparameters,
                        role=role,
                        instance_count=1,
                        instance_type=training_instance_type,
                        volume_size =10,
                        output_path=model_path,
                        sagemaker_session=pipeline_session)

    train_args = xgb_train.fit(
        inputs={
                "train": TrainingInput(
                    s3_data=f"s3://{default_bucket}/{s3_preprocessing_output_prefix}/train",
                    content_type="text/csv"
                ),
                "test": TrainingInput(
                    s3_data=f"s3://{default_bucket}/{s3_preprocessing_output_prefix}/test",
                    content_type="text/csv"
                ),
            },
    )

    step_train = TrainingStep(
        name="TrainModel",
        step_args=train_args,
        cache_config=cache_config,
        depends_on=[step_process])

    
    model = XGBoostModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
        entry_point="pipelines/cust_churn_prediction/inference.py",
        framework_version="1.5-1"
    )
    step_create_model = ModelStep(
        name="CreateModel",
        step_args=model.create(instance_type=inference_instance_type),
    )

    model_client_config = { "InvocationsTimeoutInSeconds" : 10, "InvocationsMaxRetries" : 3 }

    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}/{s3_batch_transform_output_prefix}",
    )

    step_transform = TransformStep(
        name="BatchTransform",
        transformer=transformer,
        inputs=TransformInput(
            data=f"s3://{default_bucket}/{s3_preprocessing_output_prefix}/test",
            input_filter="$[2:]",
            join_source="Input",
            output_filter="$[1,-1]",
            content_type="text/csv",
            split_type="Line",
            model_client_config = model_client_config
        ),
        cache_config=cache_config,
        depends_on=[step_train]
    )
    ### Check the Model Quality

    # In this `QualityCheckStep` we calculate the baselines for statistics and constraints using the
    # predictions that the model generates from the test dataset (output from the TransformStep). We define
    # the problem type as 'Regression' in the `ModelQualityCheckConfig` along with specifying the columns
    # which represent the input and output. Since the dataset has no headers, `_c0`, `_c1` are auto-generated
    # header names that should be used in the `ModelQualityCheckConfig`.

    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                s3_quality_prefx,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "modelqualitycheckstep",
            ],
        ),
        problem_type="BinaryClassification",
        inference_attribute="_c1",  # use auto-populated headers since we don't have headers in the dataset
        ground_truth_attribute="_c0",  # use auto-populated headers since we don't have headers in the dataset
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name,
        cache_config=cache_config,
        depends_on = [step_transform]
    )
    ### Check for Model Bias

    # Similar to the Data Bias check step, a `BiasConfig` is defined and Clarify is used to calculate
    # the model bias using the training dataset and the model.
    
    model_bias_analysis_cfg_output_path = (
        f"s3://{default_bucket}/{s3_bias_prefx}/modelbiascheckstep/analysis_cfg"
    )

    model_bias_data_config = DataConfig(
        s3_data_input_path=f"s3://{default_bucket}/{s3_preprocessing_output_prefix}/train",
        s3_output_path=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                s3_bias_prefx,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "modelbiascheckstep",
            ],
        ),
        s3_analysis_config_output_path=model_bias_analysis_cfg_output_path,
        label=1,
        dataset_type="text/csv",
        excluded_columns=[0]
    )

    model_config = ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )

    # We are using this bias config to configure Clarify to detect bias based on the first feature in the featurized vector for Sex
    model_bias_config = BiasConfig(label_values_or_threshold=[1], facet_name=[16], facet_values_or_threshold=[[1]])

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=model_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig(),
    )

    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_bias,
        register_new_baseline=register_new_baseline_model_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
        model_package_group_name=model_package_group_name,
        cache_config=cache_config,
        depends_on=[model_quality_check_step]
    )    
    
    ### Check Model Explainability

    # SageMaker Clarify uses a model-agnostic feature attribution approach, which you can used to understand
    # why a model made a prediction after training and to provide per-instance explanation during inference. The implementation
    # includes a scalable and efficient implementation of SHAP, based on the concept of a Shapley value from the field of
    # cooperative game theory that assigns each feature an importance value for a particular prediction.

    # For Model Explainability, Clarify requires an explainability configuration to be provided. In this example, we
    # use `SHAPConfig`. For more information of `explainability_config`, visit the Clarify documentation at
    # https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html.

    model_explainability_analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        default_bucket, s3_explainability_prefix, "modelexplainabilitycheckstep", "analysis_cfg"
    )

    model_explainability_data_config = DataConfig(
        s3_data_input_path=f"s3://{default_bucket}/{s3_preprocessing_output_prefix}/validation",
        s3_output_path=Join(
            on="/",
            values=[
                "s3:/",
                default_bucket,
                s3_explainability_prefix,
                ExecutionVariables.PIPELINE_EXECUTION_ID,
                "modelexplainabilitycheckstep",
            ],
        ),
        s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
        label=1,
        excluded_columns=[0],
        dataset_type="text/csv",
    )
    shap_config = SHAPConfig(seed=123, num_samples=10, num_clusters=2)
    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_explainability,
        register_new_baseline=register_new_baseline_model_explainability,
        supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
        model_package_group_name=model_package_group_name,
        cache_config=cache_config,
        depends_on=[model_bias_check_step]
    )
    
    #Upload the evaluation script to S3
    s3_client.Bucket(default_bucket).upload_file("pipelines/cust_churn_prediction/evaluate.py", f"{s3_code_prefix}/evaluate.py")

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    # define model evaluation step to evaluate the trained model
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="model-evaluation",
        role=role,
        sagemaker_session=pipeline_session,
    )
    eval_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/{s3_preprocessing_output_prefix}/validation",
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation",\
                                 destination=f"s3://{default_bucket}/{s3_model_evaluation_prefix}"),
            ],
        code=f"s3://{default_bucket}/{s3_code_prefix}/evaluate.py",
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        step_args=eval_args,
        property_files=[evaluation_report],
        depends_on=[step_create_model]
    ) 
    
    # Model Metrics
    # Define the metrics to be registered with the model in the Model Registry
    
    model_metrics = ModelMetrics(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        bias_pre_training=MetricsSource(
            s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        bias_post_training=MetricsSource(
            s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
    )

    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        bias_pre_training_constraints=MetricsSource(
            s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        bias_config_file=FileSource(
            s3_uri=model_bias_check_config.monitoring_analysis_config_uri,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        bias_post_training_constraints=MetricsSource(
            s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        explainability_constraints=MetricsSource(
            s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        explainability_config_file=FileSource(
            s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
            content_type="application/json",
        ),
    )
    
    ### Register the model

    # The two parameters in `RegisterModel` that hold the metrics calculated by the `ClarifyCheckStep` and
    # `QualityCheckStep` are `model_metrics` and `drift_check_baselines`.

    # `drift_check_baselines` - these are the baseline files that will be used for drift checks in
    # `QualityCheckStep` or `ClarifyCheckStep` and model monitoring jobs that are set up on endpoints hosting this model.
    # `model_metrics` - these should be the latest baslines calculated in the pipeline run. This can be set
    # using the step property `CalculatedBaseline`

    # The intention behind these parameters is to give users a way to configure the baselines associated with
    # a model so they can be used in drift checks or model monitoring jobs. Each time a pipeline is executed, users can
    # choose to update the `drift_check_baselines` with newly calculated baselines. The `model_metrics` can be used to
    # register the newly calculated baslines or any other metrics associated with the model.

    # Every time a baseline is calculated, it is not necessary that the baselines used for drift checks are updated to
    # the newly calculated baselines. In some cases, users may retain an older version of the baseline file to be used
    # for drift checks and not register new baselines that are calculated in the Pipeline run.

    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines
    )

    step_register = ModelStep(name="RegisterModel", step_args=register_args, depends_on=[model_explainability_check_step])

    # Define a FailStep to stop an Amazon SageMaker Model Building Pipelines execution when a desired condition or state 
    # is not achieved and to mark that pipeline's execution as failed. The FailStep also allows you to enter a custom error message, 
    # indicating the cause of the pipeline's execution failure.
    
    step_fail = FailStep(
        name="EvalScoreFail",
        error_message=Join(on=" ", values=["Execution failed due to AUC Score >", auc_score_threshold]),
    )
    
    # Condition Check
    # We define a condition check to determine whether to stop the pipeline execution when the evaluation scores (AUC) 
    # did not meet the threshold specified.
    
    cond_gte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.auc_score.value",
        ),
        right=auc_score_threshold,
    )
    step_cond = ConditionStep(
        name="CheckEvaluationScore",
        conditions=[cond_gte],
        if_steps=[step_transform, model_quality_check_step, model_bias_check_step, model_explainability_check_step,step_register],
        else_steps=[step_fail]
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            skip_check_data_quality,
            register_new_baseline_data_quality,
            supplied_baseline_constraints_data_bias,
            supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints_data_quality,
            skip_check_data_bias,
            register_new_baseline_data_bias,
            training_instance_type,
            inference_instance_type,
            skip_check_model_quality,
            register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality,
            supplied_baseline_constraints_model_quality,
            skip_check_model_bias,
            register_new_baseline_model_bias,
            supplied_baseline_constraints_model_bias,
            skip_check_model_explainability,
            register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability,
            input_data_prefix,
            model_approval_status,
            auc_score_threshold,
        ],
        steps=[step_process, 
            data_quality_check_step,
            data_bias_check_step,
            step_train, 
            step_create_model,
            step_eval, 
            step_cond
        ]
    ) 

    return pipeline
