import pyodbc 
import os, re
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import boto3
from botocore.config import Config
from zipfile import ZipFile
import polars as pl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("downloads_020824.log"),
        logging.StreamHandler()
    ]
)

dbConnect = pyodbc.connect(
    driver="{/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.3.so.3.1}",
    server="122.169.98.113",
    database="TendersDocument",
    user="sumit",
    password="sumit@123",
    TrustServerCertificate="yes",
)

# AWS credentials and configuration
aws_access_key_id ='FAKE'
aws_secret_access_key = 'FAKE'
aws_region_name = 'FAKE'
config = Config(
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    },
    max_pool_connections=50
)
bucketName = "tendertigerdocs"

# Initialize S3 resource
s3_resource = boto3.resource(
    service_name='s3',
    region_name=aws_region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    config=config
)

SQLq = "SELECT * FROM tbl_DailyDocumentData"
plDF = pl.read_database(query=SQLq, connection=dbConnect)
logging.info(f"DataFrame loaded with {len(plDF)} rows")

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

# Function to download files from S3
def download_from_s3(bucket, s3_path, local_path):
    try:
        s3_resource.Bucket(bucket).download_file(s3_path.strip(), local_path)
        logging.info(f"Downloaded {s3_path} to {local_path}")
    except Exception as e:
        logging.error(f"Error downloading {s3_path}: {e}")

# Iterate over rows in DataFrame
for row in plDF.iter_rows():
    tcno = str(row[0])
    docpath = row[1].strip("'")

    if docpath is None or docpath == "''":
        logging.warning("Skipping NoneType docpath.")
        continue
    
    # Check if the file has a .html extension
    if docpath.lower().endswith('.html'):
        logging.info(f"Skipping {docpath} as it is an HTML file.")
        continue

    local_dir = f"/data/docFolders6/{tcno}"
    local_file_path = os.path.join(local_dir, os.path.basename(docpath))
    logging.info(f"Local file path: {local_file_path}, Local directory: {local_dir}")
    create_dir(local_dir)
    download_from_s3(bucketName, docpath, local_file_path)

logging.info("Documents downloaded and organized into respective folders.")
