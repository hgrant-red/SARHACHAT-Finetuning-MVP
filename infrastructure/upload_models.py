# upload_models.py
import os
import boto3
from huggingface_hub import snapshot_download
from botocore.config import Config
from botocore.exceptions import ClientError

# Fetch secrets from environment variables instead of hardcoding
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "replace_me_locally")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "replace_me_locally")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ WARNING: Please set HF_TOKEN as an environment variable before running.")

# 1. MINIO SETUP
s3_client = boto3.client(
    's3',
    endpoint_url="http://minio.minio.svc.cluster.local:9000", 
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)
bucket = "models"
base_s3_path = "mistral-base-fp8"
lora_s3_path = f"{base_s3_path}/nurse-adapter" # NOTE: nested inside base!

try:
    s3_client.head_bucket(Bucket=bucket)
except ClientError:
    s3_client.create_bucket(Bucket=bucket)

# 2. DOWNLOAD BASE MODEL
print("⏳ Downloading base model...")
local_base_path = snapshot_download("RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic", token=HF_TOKEN)

# 3. UPLOAD BASE MODEL
print(f"🚀 Uploading Base Model to s3://{bucket}/{base_s3_path}")
for root, dirs, files in os.walk(local_base_path):
    for file in files:
        local_file = os.path.join(root, file)
        s3_key = f"{base_s3_path}/{os.path.relpath(local_file, local_base_path)}".replace("\\", "/")
        s3_client.upload_file(local_file, bucket, s3_key)

# 4. UPLOAD LORA ADAPTER 
local_lora_path = "./sarhachat-lora-vA-10epochs" 
print(f"🚀 Uploading LoRA Adapter to s3://{bucket}/{lora_s3_path}")
for root, dirs, files in os.walk(local_lora_path):
    for file in files:
        local_file = os.path.join(root, file)
        s3_key = f"{lora_s3_path}/{os.path.relpath(local_file, local_lora_path)}".replace("\\", "/")
        s3_client.upload_file(local_file, bucket, s3_key)

print("✅ ENTERPRISE SYNC COMPLETE!")