from dotenv import load_dotenv
load_dotenv()
import os
import json
from pipelines.cognee.preprocessing import process_document
from pipelines.cognee.utils.pipeline_utils import get_doc_id_from_filename
from pipelines.cognee.utils.check_processed_docs import load_registry, save_registry
from cognee_community_vector_adapter_qdrant import register
from pipelines.cognee.utils.upload_to_gcs import upload_file
from cognee import add, cognify

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

async def create_knowledge(raw_dir, clean_dir, upload_metadata = False, job_id = None):
    """
    Process PDF documents to extract metadata, store knowledge, and update registry.

    Steps performed:
    1. Load the registry of already processed documents (to avoid duplicates).
    2. Iterate over all PDFs in `raw_dir`.
    3. For each PDF:
       - Generate a document ID.
       - Skip if it is already processed.
       - Load metadata if it exists in `clean_dir`, otherwise process the PDF to create metadata.
       - Optionally upload metadata to Google Cloud Storage (if `upload_metadata=True`).
       - Send extracted text into Cognee (`add`) and generate knowledge (`cognify`).
    4. Update the registry with the new document’s metadata file and processing status.

    Args:
        raw_dir (str): Path to the folder containing raw PDF files.
        clean_dir (str): Path to the folder where processed metadata should be stored.
        upload_metadata (bool, optional): Whether to upload metadata JSON to GCS. Defaults to False.
        job_id (str, optional): Job identifier for organizing uploads in GCS. Defaults to None.

    Returns:
        None
    """
    # get json registry data
    registry = load_registry()

    for filename in os.listdir(raw_dir):
        # get pdf file
        if not filename.endswith(".pdf"):
            continue
        
        # create pdf path
        pdf_path = os.path.join(raw_dir, filename)
        doc_id = get_doc_id_from_filename(pdf_path)
        
        # check if file already processed or not
        if doc_id in registry:
            print(f"Skip {filename}, already processed")
            continue
        
        # check if metadata already exists in clean_dir
        metadata_filename = f"metadata_{doc_id}.json"
        metadata_path = os.path.join(clean_dir, metadata_filename)

        if os.path.exists(metadata_path):
            print(f"Metadata exists for {filename}, loading...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            try:
                # processing new file
                print(f"Processing {filename}")
                metadata = process_document(pdf_path)
                # write data
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                # upload metadata
                if upload_metadata == True and (job_id != None or job_id != ""):
                    print("Upload metadata")
                    upload_file(metadata_path, BUCKET_NAME, f"metadata/{job_id}/{metadata_filename}")
            except Exception as e:
                print(f"Failed {filename}: {e}")
                continue

        # create knowledge base
        for data in metadata:
            if data.get("text"):
                # create dataset for cognee
                await add(data=data['text'])
                print(f"Add: Finish created dataset for {data['slide_id']}")
                # create knowledge using cognee
                await cognify()
                print(f"Cognify: Finish created knowledge for {data['slide_id']}")

        # update registry
        registry[doc_id] = {
            "metadata_file": metadata_filename,
            "status": "done"
        }

        save_registry(registry)