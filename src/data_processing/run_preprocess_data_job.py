from sagemaker import get_execution_role
from sagemaker.processing import FrameworkProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from datetime import datetime
import logging
import argparse
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
env = os.getenv("ENVIRONMENT") # i.e. dev, stage

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_name", type=str, required=True, help="Name of Sagemaker preprocessing job")
    parser.add_argument("--region", type=str, required=True, help="AWS region for processing job")
    parser.add_argument("--input_s3_bucket", type=str, required=True, help="Name of S3 bucket where input data is stored")
    parser.add_argument("--output_s3_bucket", type=str, required=True, help="Name of S3 bucket where processed output data will be stored")
    parser.add_argument("--data_split_ratio", type=float, default=0.1, help="AWS region for processing job")
    parser.add_argument("--keep_eoCollections", action="store_true", default=False, help="Name of S3 bucket where processed output data will be stored")

    return parser.parse_args()

def authenticate():
    # TODO: make accessible from Github Actions
    role = get_execution_role()
    return role

def run_sagemaker_job(job_name, role, region, input_s3, output_s3, data_split_ratio, keep_eocollections):
    # using sklearn image
    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.4-2",
        py_version="py3",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )

    inputs = [
            ProcessingInput(
                source=f"s3://{input_s3}/",
                destination="/opt/ml/processing/input/data"
            )
        ]

    outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/output/train",
                destination=f"s3://{output_s3}/semantic_search_se/data/"
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/test",
                destination=f"s3://{output_s3}/semantic_search_se/data/"
            )
        ]
    
    arguments = [
            "--input-data-dir", "/opt/ml/processing/input/data",
            "--output-path", "/opt/ml/processing/output/",
            "--train-test-split-ratio", str(data_split_ratio),
            "--random-state", "42",]
    if keep_eocollections:
        arguments.append("--keep-eoCollections")
    
    processor.run(
        code="preprocess_data.py",
        source_dir="src/data_processing",
        inputs=inputs,
        outputs=outputs,
        arguments=arguments,
        job_name=job_name,
        wait=True, # for notebook to wait until process finishes to stop running cell
        logs=True # display logs
    )


def main():
    logger.info("Kickstarting preprocessing job")
    args = parse_args()

    complete_job_name = f"{args.job_name}-{env}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # authenticate
    logger.info("Authenticating")
    role = authenticate()
    logger.info("Authentication complete.")

    run_sagemaker_job(complete_job_name, role, args.region, args.input_s3_bucket, args.output_s3_bucket, args.data_split_ratio, args.keep_eoCollections)
    
    logger.info("Finished preprocessing job.")

if __name__ == "__main__":
    main()