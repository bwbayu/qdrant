from google.cloud import storage
import os
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

GCS_CRED_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

def upload_file(source, bucket_name: str, dest_path: str, content_type: str = None) -> str:
    """
    Upload a file or in-memory buffer to Google Cloud Storage.

    Args:
        source (str | BytesIO): 
            - If str: local file path to upload.
            - If BytesIO: in-memory file-like object (e.g., an image buffer).
        bucket_name (str): Name of the target GCS bucket.
        dest_path (str): Destination path (object key) inside the bucket.

    Returns:
        str: Public URL of the uploaded file in GCS.

    Notes:
        - If GOOGLE_APPLICATION_CREDENTIALS is set, it uses that JSON key file.
        - If not, it falls back to default GCP credentials (e.g., from environment).
        - Automatically handles Windows backslashes in `dest_path` by replacing with "/".
        - If uploading from BytesIO, sets `content_type="image/jpeg"`. 
          For other file types, adjust the `content_type` accordingly.
    """
    # Initialize GCS client (with explicit service account if provided)
    if GCS_CRED_JSON:
        client_gcs = storage.Client.from_service_account_json(GCS_CRED_JSON)
    else:
        client_gcs = storage.Client()

    bucket = client_gcs.bucket(bucket_name)
    blob_name = dest_path.replace("\\", "/")  # normalize path separators
    blob = bucket.blob(blob_name)

    if isinstance(source, str):  # Local file path
        blob.upload_from_filename(source)
    elif isinstance(source, BytesIO):  # In-memory buffer
        blob.upload_from_file(source, content_type=content_type or "application/octet-stream")
    else:
        raise TypeError("source must be str (file path) or BytesIO")

    return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
