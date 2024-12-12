### 22-08-2024 MODIFIED SCRIPT
import os
import logging
import requests
import boto3
from botocore.config import Config
import pandas as pd
from botocore.exceptions import ClientError
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# AWS credentials and configuration
aws_access_key_id = 'FAKE'
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

def get_log_filename():
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_dir = '/data/s3logs'
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f's3_download_{date_str}.log')


# Set up logging with daily log files
logging.basicConfig(filename=get_log_filename(), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


url = "https://tendersearchai.tendertiger.co.in/api/tender/GetDocumentsGroupedBy"
# Your authentication token
token = "Securedocs#ttneo2@24!$esk@pi"
headers = {
    "Token": f"{token}"
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    if data == []:
        logging.error("Found zero records")
else:
    print(f"Failed to fetch data: {response.status_code}")


# Prepare the list to store flattened data
flattened_data = []

# Iterate over the entries in the JSON data
for entry in data:
    tcNo = entry.get('tcNo')
    isalready = entry.get('isalready')
    
    # Check if 'files' key exists and iterate over the files
    if 'files' in entry:
        for file in entry['files']:
            file_data = {
                'tcNo': tcNo,
                'isalready': isalready,
                'originalDocPath': file.get('originalDocPath'),
                'docname': file.get('docname'),
                'docType': file.get('docType'),
                'docmentdate': file.get('docmentdate'),
                'fileext': file.get('fileext')
            }
            flattened_data.append(file_data)

# Convert the list of flattened data to a DataFrame
df = pd.DataFrame(flattened_data)

# Function to download a file from S3
def download_from_s3(row):
    try:
        tc_no = str(row['tcNo'])
        file_path = row['originalDocPath']
        folder_path = os.path.join(base_dir, tc_no)

        # Ensure the directory exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_path = os.path.join(folder_path, row['docname'])

        # Check if the save path (file) already exists
        if os.path.exists(save_path):
            logging.info(f"File already exists: {save_path}")
        else:
            logging.info(f"Preparing to download file: {file_path}")
            s3_resource.Bucket(bucketName).download_file("TenderTiger_FileDocs/" + file_path, save_path)
            logging.info(f"Successfully downloaded {file_path} to {save_path}")
    except s3_resource.meta.client.exceptions.NoSuchKey:
        logging.error(f"Error: The file {file_path} does not exist in bucket {bucketName}")
    except Exception as e:
        logging.error(f"Error downloading {file_path}: {e}")

# Directory where you want to save the downloaded files
base_dir = r"/data/dailydocument/"
os.makedirs(base_dir, exist_ok=True)
# Using ThreadPoolExecutor for parallel downloading
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(download_from_s3, row) for index, row in df.iterrows()]
    for future in as_completed(futures):
        future.result()  # This will raise any exception caught during execution

