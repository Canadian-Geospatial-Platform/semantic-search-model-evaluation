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
env = os.getenv("ENV") # e.g. stage

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sagemaker_job_name", type=str, help="Name of Sagemaker preprocessing job")
    parser.add_argument("--input_s3_bucket", type=str, help="Name of S3 bucket where input data is stored")
    parser.add_argument("--output_s3_bucket", type=str, help="Name of S3 bucket where processed output data will be stored")
    parser.add_argument("--data_split_ratio", type=float, default=0.1, help="Ratio for splitting training and test data")
    parser.add_argument("--keep_eoCollections", action="store_true", default=False, help="Whether to keep EO collections in the processed data")

    return parser.parse_args()

def authenticate():
    role = get_execution_role()
    return role

def run_sagemaker_job(job_name, role, input_s3, output_s3, data_split_ratio, keep_eocollections):
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
                source="/opt/ml/processing/output/",
                destination=f"s3://{output_s3}/semantic_search_se/data/" if not keep_eocollections else f"s3://{output_s3}/semantic_search_se/data/with_eocollections/"
            ),
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

def get_args_if_not_set(args):
    for arg_name, arg_value in vars(args).items():
        if arg_value is None:
            logger.info(f"Argument '{arg_name}' not set. Attempting to retrieve from environment variables.")
            arg_value_from_env = os.getenv(arg_name.upper())
            if arg_value_from_env is not None:
                logger.info(f"Successfully retrieved '{arg_name}' from environment variable.")
                setattr(args, arg_name, arg_value_from_env)
            else:
                logger.error(f"Environment variable for argument '{arg_name}' not found. Please set the argument or the corresponding environment variable and try again.")
                exit(1)
        else:
            logger.info(f"Argument '{arg_name}' is set to: {arg_value}")

    return args

def main():
    logger.info("Kickstarting preprocessing job")
    args = parse_args()
    args = get_args_if_not_set(args)

    complete_job_name = f"{args.sagemaker_job_name}-{env}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # authenticate
    logger.info("Authenticating")
    role = authenticate()
    logger.info("Authentication complete.")

    run_sagemaker_job(complete_job_name, role, args.input_s3_bucket, args.output_s3_bucket, args.data_split_ratio, args.keep_eoCollections)
    
    logger.info("Finished preprocessing job.")

if __name__ == "__main__":
    main()