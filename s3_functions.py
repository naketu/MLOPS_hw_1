import boto3
import os
import logging
import pickle
from io import BytesIO
import pandas as pd
from botocore.exceptions import ClientError

# setting up a logger
logger = logging.getLogger("s3_logger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("s3_logger.log", mode="a")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# setting up a logger S3 minio
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = os.getenv("BUCKET_NAME", "ml-models")


# setting up a bucket
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="rus-1",
    )
    
    # Create bucket if it doesn't exist
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        logger.info(f"Bucket '{BUCKET_NAME}' created or already exists")
    
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            logger.error(f"Error creating bucket: {e}")
    
    S3_AVAILABLE = True
    logger.info(f"Connected to S3/Minio at {MINIO_ENDPOINT}")

except Exception as e:
    S3_AVAILABLE = False
    logger.warning(f"S3/Minio connection failed: {e}. Operating in memory-only mode.")
    s3_client = None

def get_model_ids_from_s3() -> list:
    
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="models/",
    )

    if "Contents" not in response:
        return []
    
    return [
        obj["Key"].replace("models/model_", "").replace(".pkl", "")
        for obj in response["Contents"]
    ]

def save_model_to_s3(model_id, model) -> bool:
    """
    Сохранить модель в S3/Minio.
    Возвращает True если успешно, False иначе.
    """
    if not S3_AVAILABLE or s3_client is None:
        logger.warning(f"S3 not available, skipping save for model {model_id}")
        return False
    
    try:
        # Сериализовать модель в pickle
        buffer = BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        
        # Загрузить в S3
        s3_key = f"models/model_{int(model_id)}.pkl"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=buffer.getvalue()
        )
        
        logger.info(f"Model {model_id} successfully saved to S3 at {s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model {model_id} to S3: {e}")
        return False

def load_model_from_s3(model_id: int):
    """
    Загрузить модель из S3/Minio в память
    """
    if not S3_AVAILABLE or s3_client is None:
        logger.warning(f"S3 not available, cannot load model {int(model_id)}")
        return False
    
    try:
        s3_key = f"models/model_{model_id}.pkl"
        
        # Скачать модель из S3
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        model_data = response["Body"].read()
        
        # Десериализовать модель
        buffer = BytesIO(model_data)
        model = pickle.load(buffer)\
        
        logger.info(f"Model {model_id} successfully loaded from S3")
        return model

        
    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Model {model_id} not found in S3")
        return False
    
    except Exception as e:
        logger.error(f"Error loading model {model_id} from S3: {e}")
        return False

def delete_model_from_s3(model_id: int) -> bool:
    """
    Удалить модель из S3/Minio.
    Возвращает True если успешно, False иначе.
    """
    if not S3_AVAILABLE or s3_client is None:
        logger.warning(f"S3 not available, skipping S3 delete for model {model_id}")
        return False
    
    try:
        s3_key = f"models/model_{model_id}.pkl"
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Model {model_id} successfully deleted from S3")
        return True
    except Exception as e:
        logger.error(f"Error deleting model {model_id} from S3: {e}")
        return False

def save_dataset_to_s3(dataset_name: str, data: pd.DataFrame) -> bool:
    """
    Сохранить датасет в S3/Minio в формате parquet.
    """
    if not S3_AVAILABLE or s3_client is None:
        logger.warning(f"S3 not available, skipping dataset save")
        return False
    
    try:
        buffer = BytesIO()
        data.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        s3_key = f"datasets/{dataset_name}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=buffer.getvalue()
        )
        
        logger.info(f"Dataset '{dataset_name}' saved to S3 at {s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving dataset to S3: {e}")
        return False