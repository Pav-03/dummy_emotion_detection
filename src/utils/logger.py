import logging
import os
import boto3
from datetime import datetime

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Set up a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # prevent adding multiple handlers to the logger

    if logger.handlers:
        return logger
    
    # Console handler for logging debug and higher level messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # File Handler for logging errors

    if log_file is None:
        os.makedirs('logs', exist_ok=True)
        log_file = f'logs/{name}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def upload_logs_to_s3(bucket_name: str = None, prefix: str = "logs"):
    """
    Upload all log files to S3.
    Call this at the END of your pipeline.
    
    S3 path: s3://bucket/logs/2026/02/28/model_building.log
    """
    try:
        bucket_name = bucket_name or os.environ.get(
            "MLFLOW_S3_BUCKET", "dummy-emotion-detection-mlops"
        )
        s3_client = boto3.client('s3')
        today = datetime.now().strftime("%Y/%m/%d")

        log_dir = 'logs'
        if not os.path.exists(log_dir):
            print("No logs directory found.")
            return

        uploaded = 0
        for log_file in os.listdir(log_dir):
            if log_file.endswith('.log'):
                local_path = os.path.join(log_dir, log_file)
                s3_key = f"{prefix}/{today}/{log_file}"

                s3_client.upload_file(local_path, bucket_name, s3_key)
                uploaded += 1
                print(f"Uploaded: {log_file} → s3://{bucket_name}/{s3_key}")

        print(f"Total {uploaded} log files uploaded to S3!")

    except Exception as e:
        print(f"Failed to upload logs to S3: {e}")