import os
import boto3
from huggingface_hub import snapshot_download
from botocore.config import Config
from botocore.exceptions import ClientError

# --- ENVIRONMENT CONFIGURATION ---
# Fetch secrets and endpoints from environment variables instead of hardcoding cluster URLs
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
# Defaulting to localhost so it doesn't leak internal cluster DNS
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000") 
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "models")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL_REPO = os.getenv("BASE_MODEL_REPO", "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic")
LORA_DIR = os.getenv("LORA_DIR", "./sarhachat-lora-vA-10epochs")

if not HF_TOKEN:
    print("⚠️ WARNING: Please set HF_TOKEN as an environment variable before running to download gated models.")
if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    print("⚠️ WARNING: MinIO/S3 credentials not found in environment. Script may fail if the bucket requires auth.")

# 1. S3/MINIO SETUP
print(f"🔌 Connecting to S3 Endpoint: {S3_ENDPOINT_URL}")
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL, 
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

base_s3_path = "mistral-base-fp8"
lora_s3_path = f"{base_s3_path}/nurse-adapter" # NOTE: nested inside base!

try:
    s3_client.head_bucket(Bucket=BUCKET_NAME)
except ClientError:
    print(f"🪣 Bucket '{BUCKET_NAME}' not found. Attempting to create it...")
    s3_client.create_bucket(Bucket=BUCKET_NAME)

# 2. DOWNLOAD BASE MODEL
print(f"⏳ Downloading base model: {BASE_MODEL_REPO}...")
local_base_path = snapshot_download(BASE_MODEL_REPO, token=HF_TOKEN)

# 3. UPLOAD BASE MODEL
print(f"🚀 Uploading Base Model to s3://{BUCKET_NAME}/{base_s3_path}")
for root, dirs, files in os.walk(local_base_path):
    for file in files:
        local_file = os.path.join(root, file)
        s3_key = f"{base_s3_path}/{os.path.relpath(local_file, local_base_path)}".replace("\\", "/")
        s3_client.upload_file(local_file, BUCKET_NAME, s3_key)

# 4. UPLOAD LORA ADAPTER 
print(f"🚀 Uploading LoRA Adapter from {LORA_DIR} to s3://{BUCKET_NAME}/{lora_s3_path}")
if os.path.exists(LORA_DIR):
    for root, dirs, files in os.walk(LORA_DIR):
        for file in files:
            local_file = os.path.join(root, file)
            s3_key = f"{lora_s3_path}/{os.path.relpath(local_file, LORA_DIR)}".replace("\\", "/")
            s3_client.upload_file(local_file, BUCKET_NAME, s3_key)
    print("✅ ENTERPRISE SYNC COMPLETE!")
else:
    print(f"🚨 ERROR: LoRA directory '{LORA_DIR}' not found. Did you run the training script first?")