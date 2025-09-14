import os
import json
from preprocessing import process_document
from utils.pipeline_utils import get_doc_id_from_filename
from utils.check_processed_docs import load_registry, save_registry
from cognee_community_vector_adapter_qdrant import register
from cognee import add, cognify

RAW_DIR = "data/raw/logif"
CLEAN_DIR = "data/clean/logif"


async def create_knowledge():
    # get json registry data
    registry = load_registry()

    for filename in os.listdir(RAW_DIR):
        # get pdf file
        if not filename.endswith(".pdf"):
            continue
        
        # create pdf path
        pdf_path = os.path.join(RAW_DIR, filename)
        doc_id = get_doc_id_from_filename(pdf_path)
        
        # check if file already processed or not
        if doc_id in registry:
            print(f"Skip {filename}, already processed")
            continue
        
        # check if metadata already exists in clean_dir
        metadata_filename = f"metadata_{doc_id}.json"
        metadata_path = os.path.join(CLEAN_DIR, metadata_filename)
        if os.path.exists(metadata_path):
            print(f"Metadata exists for {filename}, loading...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            try:
                # processing new file
                print(f"Processing {filename}")
                metadata = process_document(pdf_path)
                out_file = os.path.join(CLEAN_DIR, metadata_filename)
                # write data
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed {filename}: {e}")
                continue
        
        all_text = "\n".join([data.get("text", "") for data in metadata if data.get("text")])

        if all_text:
            await add(data=all_text)
            print(f"Add: Finish created dataset for {doc_id}")
            await cognify()
            print(f"Cognify: Finish created knowledge for {doc_id}")

        # # create knowledge base
        # for data in metadata:
        #     if data.get("text"):
        #         await add(data=data['text'])
        #         print(f"Add: Finish created dataset for {data['slide_id']}")
        #         await cognify()
        #         print(f"Cognify: Finish created knowledge for {data['slide_id']}")

        # update registry
        registry[doc_id] = {
            "metadata_file": metadata_filename,
            "status": "done"
        }
        save_registry(registry)